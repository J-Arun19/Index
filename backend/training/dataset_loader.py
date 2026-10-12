import os

# Configure OpenBLAS and numerical libraries thread limits before importing numpy/cv2/torch
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import csv
import cv2
import time
import random
import queue
import torch
import numpy as np
import multiprocessing as mp
from collections import Counter
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.video_preprocessor import VideoPreprocessor
from backend.training.dataset_exceptions import (
    DatasetDecodeError,
    VideoDecodeError,
    ImageDecodeError,
    SamplePreprocessingError,
    is_fatal_error,
)

# Disable OpenCV OpenCL globally to enforce CPU execution & reduce log noise
cv2.ocl.setUseOpenCL(False)
if hasattr(cv2, "setNumThreads"):
    cv2.setNumThreads(1)
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

logger = setup_logger("DeepfakeDataset")


class DatasetStatsTracker:
    """
    High-throughput, multiprocessing-safe stats tracker.
    Hot path (successful samples) does zero IPC / locks.
    Failure accounting uses lock-based mp.Value and non-blocking mp.Queue only on the rare failure path.
    """
    def __init__(self):
        self._decode_failures = mp.Value('q', 0)
        self._retry_count = mp.Value('q', 0)
        self._skipped_samples = mp.Value('q', 0)
        self._failed_video_queue = mp.Queue(maxsize=5000)
        self._unique_failed_videos = set()

    def record_decode_failure(self, sample_path: Optional[str] = None):
        with self._decode_failures.get_lock():
            self._decode_failures.value += 1
        if sample_path:
            try:
                self._failed_video_queue.put_nowait(str(sample_path))
            except (queue.Full, Exception):
                pass

    def record_retry(self):
        with self._retry_count.get_lock():
            self._retry_count.value += 1

    def record_skipped(self):
        with self._skipped_samples.get_lock():
            self._skipped_samples.value += 1

    def _drain_queue(self):
        while True:
            try:
                item = self._failed_video_queue.get_nowait()
                self._unique_failed_videos.add(item)
            except (queue.Empty, Exception):
                break

    def get_stats(self, successful_samples: int = 0) -> dict:
        self._drain_queue()
        failures = self._decode_failures.value
        retries = self._retry_count.value
        skipped = self._skipped_samples.value
        total_attempted = successful_samples + skipped
        fail_rate = (failures / max(1, total_attempted)) * 100.0 if total_attempted > 0 else 0.0
        return {
            "total_samples_attempted": total_attempted,
            "successful_samples": successful_samples,
            "decode_failures": failures,
            "retry_count": retries,
            "skipped_samples": skipped,
            "unique_failed_videos": len(self._unique_failed_videos),
            "unique_failed_video_list": sorted(list(self._unique_failed_videos)),
            "failure_rate_pct": fail_rate
        }


_last_reject_log_time = 0.0
_reject_count = 0
_REJECT_LOG_INTERVAL_SEC = 5.0


def _log_rejection(reason: str):
    """Rate-limited logging helper for safe_collate_fn rejections."""
    global _last_reject_log_time, _reject_count
    _reject_count += 1
    now = time.time()
    if _reject_count <= 3 or (now - _last_reject_log_time) >= _REJECT_LOG_INTERVAL_SEC:
        logger.warning(f"safe_collate_fn rejected sample ({reason}) [total rejected in worker: {_reject_count}]")
        _last_reject_log_time = now


def safe_collate_fn(batch: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Collate function that defensively filters out invalid, corrupt, or inconsistent samples.
    - Accepts only valid (image_tensor, label) pairs.
    - Rejects None entries, malformed tuples/lists, and missing labels.
    - Rejects non-tensor image values and tensors with invalid dimensions or NaNs/Infs.
    - Enforces batch-wide shape homogeneity: rejects any tensor whose shape deviates from the dominant batch shape.
    - Normalizes labels to scalar torch.long tensors.
    - Never passes invalid items to torch.utils.data.dataloader.default_collate.
    - If all samples in the batch fail or are rejected, returns (None, None).
    - Never fabricates replacement data or zero-filled tensors.
    """
    if not batch:
        return None, None

    candidate_items = []

    for item in batch:
        # 1. Reject None item
        if item is None:
            _log_rejection("item is None")
            continue

        # 2. Reject malformed tuples/lists (must be a 2-element sequence)
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            item_type = type(item).__name__
            item_len = len(item) if hasattr(item, "__len__") else "N/A"
            _log_rejection(f"malformed item: expected 2-element sequence, got {item_type} of length {item_len}")
            continue

        img, label = item

        # 3. Reject None entries for image or label
        if img is None:
            _log_rejection("image is None")
            continue
        if label is None:
            _log_rejection("label is None")
            continue

        # 4. Reject non-tensor image values
        if not isinstance(img, torch.Tensor):
            _log_rejection(f"image is not a torch.Tensor (got {type(img).__name__})")
            continue

        # 5. Reject tensors with unexpected dimensions or empty sizes
        # Image tensors must be 3D (C, H, W); video tensors 4D (C, T, H, W)
        if img.ndim not in (3, 4) or img.numel() == 0 or any(dim <= 0 for dim in img.shape):
            _log_rejection(f"image tensor has unexpected dimensions: {tuple(img.shape)}")
            continue

        # Reject tensors with NaN or Inf
        if torch.isnan(img).any() or torch.isinf(img).any():
            _log_rejection("image tensor contains NaN or Inf values")
            continue

        # 6. Validate and normalize label: must be scalar integer or 0D/1D single-element tensor
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                _log_rejection(f"label tensor has non-scalar shape: {tuple(label.shape)}")
                continue
            try:
                lbl_val = int(label.item())
            except (ValueError, TypeError) as e:
                _log_rejection(f"label tensor cannot be cast to scalar int: {e}")
                continue
            norm_label = torch.tensor(lbl_val, dtype=torch.long)
        elif isinstance(label, (int, np.integer)):
            norm_label = torch.tensor(int(label), dtype=torch.long)
        else:
            _log_rejection(f"label is not an int or scalar tensor (got {type(label).__name__})")
            continue

        # Ensure image tensor is float32 and contiguous
        if img.dtype != torch.float32:
            img = img.to(torch.float32)
        if not img.is_contiguous():
            img = img.contiguous()

        candidate_items.append((img, norm_label))

    if not candidate_items:
        return None, None

    # 7. Ensure all image tensors in a batch have identical shape before default_collate
    # Determine the dominant (majority) shape among valid candidates
    shapes = [item[0].shape for item in candidate_items]
    shape_counts = Counter(shapes)
    dominant_shape = shape_counts.most_common(1)[0][0]

    valid_items = []
    for img, norm_label in candidate_items:
        if img.shape == dominant_shape:
            valid_items.append((img, norm_label))
        else:
            _log_rejection(f"tensor shape {tuple(img.shape)} deviates from dominant batch shape {tuple(dominant_shape)}")

    if not valid_items:
        return None, None

    try:
        return torch.utils.data.dataloader.default_collate(valid_items)
    except Exception as e:
        if is_fatal_error(e):
            raise
        logger.warning(f"safe_collate_fn: default_collate failed on candidate batch ({e}). Safely skipping batch.")
        return None, None


class DeepfakeDataset(Dataset):
    """Unified PyTorch Dataset for lazy loading image and video deepfake samples from manifest."""
    
    def __init__(
        self,
        split: str,
        media_type: str,
        manifest_path: Optional[str] = None,
        preprocessor: Optional[any] = None,
        max_retries: int = 5,
        use_cache: bool = True,
        stats_tracker: Optional[DatasetStatsTracker] = None
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            media_type (str): One of 'image', 'video'.
            manifest_path (str, optional): Path to dataset_manifest.csv.
            preprocessor (optional): Custom preprocessor instance.
            max_retries (int): Maximum number of retries to load other samples if a sample is corrupted.
            use_cache (bool): Whether to cache loaded/preprocessed samples in memory.
            stats_tracker (optional): Shared DatasetStatsTracker instance.
        """
        self.split = split.lower()
        self.media_type = media_type.lower()
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.is_pre_cached = False
        self.stats = stats_tracker if stats_tracker is not None else DatasetStatsTracker()
        
        if manifest_path is None:
            manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
            
        self.manifest_path = manifest_path
        
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Split must be 'train', 'val', or 'test'. Got: {split}")
        if self.media_type not in ("image", "video"):
            raise ValueError(f"MediaType must be 'image' or 'video'. Got: {media_type}")
            
        # Initialize appropriate preprocessor if not provided
        self.preprocessor = preprocessor
        if self.preprocessor is None:
            if self.media_type == "image":
                self.preprocessor = ImagePreprocessor()
            else:
                self.preprocessor = VideoPreprocessor()
                
        # Load and filter manifest
        self.samples = self._load_and_filter_manifest()
        self.cached_tensors = None
        self.cached_images = None
        logger.info(f"Initialized {self.split} {self.media_type} dataset with {len(self.samples)} samples (use_cache={self.use_cache}).")
        
        # Log class counts
        class_counts = {0: 0, 1: 0}
        for s in self.samples:
            class_counts[s["label"]] += 1
        logger.info(f"Class distribution: REAL (0): {class_counts[0]}, FAKE (1): {class_counts[1]}")

    @property
    def failed_sample_count(self) -> int:
        """Backward compatibility for decode failure count."""
        return self.stats._decode_failures.value

    @property
    def retry_count(self) -> int:
        """Backward compatibility for retry count."""
        return self.stats._retry_count.value

    @property
    def skipped_samples(self) -> int:
        """Backward compatibility for skipped sample count."""
        return self.stats._skipped_samples.value

    def get_failure_stats(self, successful_samples: int = 0) -> dict:
        """Returns the failure, retry, and skip statistics dictionary."""
        return self.stats.get_stats(successful_samples=successful_samples)

    def pre_cache_samples(self):
        """Pre-caches all samples to memory to avoid repeated disk reading and OpenCV video decoding/seeking."""
        if not self.use_cache:
            logger.info(f"Pre-caching skipped because use_cache=False for split '{self.split}' ({self.media_type}).")
            return
        logger.info(f"Pre-caching {len(self.samples)} samples for split '{self.split}' ({self.media_type})...")
        self.cached_tensors = {}
        self.cached_images = {}
        
        if self.media_type == "video":
            # For video, process each file sequentially
            total = len(self.samples)
            for i, sample in enumerate(self.samples):
                if i % 50 == 0:
                    logger.info(f"Pre-cached {i}/{total} videos...")
                file_path = sample["file_path"]
                try:
                    tensor = self.preprocessor.preprocess_video(file_path)
                    self.cached_tensors[i] = tensor
                except Exception as e:
                    if is_fatal_error(e):
                        raise
                    logger.warning(f"Error pre-caching video {file_path}: {e}")
            self.is_pre_cached = True
            logger.info(f"Completed pre-caching. Cached {len(self.cached_tensors)}/{len(self.samples)} video samples.")
            return

        # For image: group video frame extractions by file path
        video_groups = {}
        image_samples = []
        
        for idx, sample in enumerate(self.samples):
            origin = sample.get("sample_origin", "physical_image")
            if origin == "video_frame":
                video_groups.setdefault(sample["file_path"], []).append((idx, sample["frame_index"]))
            else:
                image_samples.append((idx, sample))
                
        # Load physical images
        for idx, sample in image_samples:
            try:
                img_rgb = self.preprocessor.get_cropped_resized_rgb(sample["file_path"])
                self.cached_images[idx] = img_rgb
            except Exception as e:
                if is_fatal_error(e):
                    raise
                logger.warning(f"Error pre-caching image sample {sample['file_path']}: {e}")
                
        # Load video frames efficiently with guaranteed VideoCapture cleanup
        total_videos = len(video_groups)
        for i, (file_path, frame_requests) in enumerate(video_groups.items()):
            if i % 100 == 0:
                logger.info(f"Pre-cached {i}/{total_videos} videos...")
            cap = None
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    logger.warning(f"Failed to open video file: {file_path}")
                    continue
                
                # Sort frame requests to read them sequentially (which is MUCH faster than random seeking!)
                frame_requests_sorted = sorted(frame_requests, key=lambda x: x[1])
                
                current_frame = 0
                for idx, frame_idx in frame_requests_sorted:
                    if frame_idx < current_frame:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        current_frame = frame_idx
                    
                    while current_frame < frame_idx:
                        cap.grab()
                        current_frame += 1
                        
                    ret, frame = cap.read()
                    current_frame += 1
                    
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame index {frame_idx} from video: {file_path}")
                        frame = None
                        continue
                        
                    img_rgb = self.preprocessor.get_cropped_resized_rgb(frame)
                    self.cached_images[idx] = img_rgb
                    frame = None
            except Exception as e:
                if is_fatal_error(e):
                    raise
                logger.warning(f"Error pre-caching video frames from {file_path}: {e}")
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                
        self.is_pre_cached = True
        logger.info(f"Completed pre-caching. Cached {len(self.cached_images)}/{len(self.samples)} samples.")

    def _load_and_filter_manifest(self) -> List[Dict]:
        """Reads the CSV manifest and filters rows matching the target split and media type."""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}. Please run generate_manifest.py first.")
            
        filtered_samples = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"].lower() == self.split and row["media_type"].lower() == self.media_type:
                    filtered_samples.append({
                        "dataset_source": row["dataset_source"],
                        "file_path": row["file_path"],
                        "label": int(row["label"]),
                        "manipulation_type": row["manipulation_type"],
                        "compression": row["compression"],
                        "sample_origin": row.get("sample_origin", "video" if row["media_type"] == "video" else "physical_image"),
                        "frame_index": int(row.get("frame_index", -1))
                    })
        return filtered_samples

    def get_class_weights(self) -> torch.Tensor:
        """Computes inverse frequency class weights to handle class imbalance."""
        class_counts = {0: 0, 1: 0}
        for s in self.samples:
            class_counts[s["label"]] += 1
            
        total = len(self.samples)
        if total == 0:
            return torch.tensor([1.0, 1.0], dtype=torch.float32)
            
        w_real = total / (2.0 * max(1, class_counts[0]))
        w_fake = total / (2.0 * max(1, class_counts[1]))
        return torch.tensor([w_real, w_fake], dtype=torch.float32)

    def _extract_video_frame(self, file_path: str, frame_idx: int) -> np.ndarray:
        """Helper to extract a specific frame from a video file using OpenCV, strictly guaranteeing VideoCapture release."""
        if not os.path.exists(file_path):
            raise VideoDecodeError(f"Video file does not exist: {file_path}", sample_path=file_path, frame_index=frame_idx)
            
        cap = None
        try:
            try:
                cap = cv2.VideoCapture(file_path)
            except Exception as e:
                if is_fatal_error(e):
                    raise
                raise VideoDecodeError(f"OpenCV VideoCapture exception for {file_path}: {e}", sample_path=file_path, frame_index=frame_idx) from e
                
            if not cap.isOpened():
                raise VideoDecodeError(f"Failed to open video file (isOpened returned False): {file_path}", sample_path=file_path, frame_index=frame_idx)
                
            if frame_idx >= 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                except cv2.error as e:
                    raise VideoDecodeError(f"OpenCV seek error for frame {frame_idx} in {file_path}: {e}", sample_path=file_path, frame_index=frame_idx) from e
                
            try:
                ret, frame = cap.read()
            except Exception as e:
                if is_fatal_error(e):
                    raise
                raise VideoDecodeError(f"OpenCV frame read exception for frame {frame_idx} in {file_path}: {e}", sample_path=file_path, frame_index=frame_idx) from e
                
            if not ret or frame is None or getattr(frame, "size", 0) == 0:
                raise VideoDecodeError(f"Failed to read valid frame index {frame_idx} from video: {file_path}", sample_path=file_path, frame_index=frame_idx)

            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3 or frame.shape[2] != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                raise VideoDecodeError(f"Extracted frame has invalid dimensions {getattr(frame, 'shape', None)} from video: {file_path}", sample_path=file_path, frame_index=frame_idx)
                
            return frame
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Lazy loads, preprocesses, and returns a single item from the dataset with narrow, explicit error recovery.
        Never crashes the DataLoader worker on recoverable decode failures; skips sample if retries exhausted.
        """
        is_train = (self.split == "train")
        
        if self.media_type == "image":
            if self.is_pre_cached and self.cached_images is not None and idx in self.cached_images:
                img_rgb = self.cached_images[idx]
                tensor = self.preprocessor.apply_augmentation_and_normalization(img_rgb, is_training=is_train)
                return tensor, torch.tensor(self.samples[idx]["label"], dtype=torch.long)
        else:
            if self.is_pre_cached and self.cached_tensors is not None and idx in self.cached_tensors:
                return self.cached_tensors[idx], torch.tensor(self.samples[idx]["label"], dtype=torch.long)
            
        retries = 0
        current_idx = idx
        
        while retries <= self.max_retries:
            sample = self.samples[current_idx]
            file_path = sample["file_path"]
            label = sample["label"]
            origin = sample["sample_origin"]
            frame_idx = sample["frame_index"]
            
            frame = None
            img_rgb = None
            try:
                if self.media_type == "image":
                    if origin == "video_frame":
                        frame = self._extract_video_frame(file_path, frame_idx)
                        img_rgb = self.preprocessor.get_cropped_resized_rgb(frame)
                        del frame
                        frame = None # Explicitly free full-resolution frame
                    else:
                        img_rgb = self.preprocessor.get_cropped_resized_rgb(file_path)
                    
                    tensor = self.preprocessor.apply_augmentation_and_normalization(img_rgb, is_training=is_train)
                    del img_rgb
                    img_rgb = None # Explicitly free RGB array

                    # Validate tensor shape and integrity before returning
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
                        raise SamplePreprocessingError(f"Preprocessed tensor has invalid shape: {getattr(tensor, 'shape', None)}", sample_path=file_path)
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        raise SamplePreprocessingError("Preprocessed tensor contains NaN/Inf values", sample_path=file_path)
                else:
                    try:
                        tensor = self.preprocessor.preprocess_video(file_path)
                    except (ValueError, FileNotFoundError, cv2.error) as e:
                        raise VideoDecodeError(f"Video preprocessing failed for {file_path}: {e}", sample_path=file_path) from e

                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or tensor.shape[1] != 3 or tensor.shape[0] <= 0:
                        raise VideoDecodeError(f"Preprocessed video tensor has invalid shape: {getattr(tensor, 'shape', None)}", sample_path=file_path)
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        raise VideoDecodeError("Preprocessed video tensor contains NaN/Inf values", sample_path=file_path)
                
                return tensor, torch.tensor(int(label), dtype=torch.long)
                
            except Exception as e:
                # Clean up intermediate allocations cleanly
                if frame is not None:
                    del frame
                    frame = None
                if img_rgb is not None:
                    del img_rgb
                    img_rgb = None

                # Fatal programming/system errors must terminate training immediately
                if is_fatal_error(e):
                    import gc
                    gc.collect()
                    raise

                # Normalize recoverable decode/preprocess exceptions into DatasetDecodeError
                if isinstance(e, DatasetDecodeError):
                    decode_err = e
                elif isinstance(e, (cv2.error, ValueError, IOError, OSError)):
                    decode_err = SamplePreprocessingError(f"Recoverable sample load error for {file_path}: {e}", sample_path=file_path)
                else:
                    # Arbitrary unexpected RuntimeError or other exception must not be swallowed
                    raise

                err_sample_path = getattr(decode_err, "sample_path", None) or file_path
                self.stats.record_decode_failure(err_sample_path)
                err_msg = str(decode_err)
                
                if retries < self.max_retries:
                    self.stats.record_retry()
                    logger.warning(
                        f"Sample load/decode error [sample_idx={current_idx}, retry={retries + 1}/{self.max_retries}]: {err_msg}. "
                        f"Retrying with another random sample..."
                    )
                    retries += 1
                    current_idx = random.randint(0, len(self.samples) - 1)
                    del decode_err
                    del e
                    # Run garbage collection on failure to immediately reclaim memory
                    import gc
                    gc.collect()
                else:
                    # All retries exhausted for this sample: record skip and return None without crashing
                    self.stats.record_skipped()
                    logger.warning(
                        f"All {self.max_retries} retries exhausted for requested sample index {idx} "
                        f"(last attempted {current_idx}: {file_path}). Skipping sample safely."
                    )
                    del decode_err
                    del e
                    import gc
                    gc.collect()
                    return None, None
