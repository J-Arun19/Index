import os
import csv
import cv2
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.video_preprocessor import VideoPreprocessor

logger = setup_logger("DeepfakeDataset")

class DeepfakeDataset(Dataset):
    """Unified PyTorch Dataset for lazy loading image and video deepfake samples from manifest."""
    
    def __init__(
        self,
        split: str,
        media_type: str,
        manifest_path: Optional[str] = None,
        preprocessor: Optional[any] = None,
        max_retries: int = 5
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            media_type (str): One of 'image', 'video'.
            manifest_path (str, optional): Path to dataset_manifest.csv.
            preprocessor (optional): Custom preprocessor instance.
            max_retries (int): Maximum number of retries to load other samples if a sample is corrupted.
        """
        self.split = split.lower()
        self.media_type = media_type.lower()
        self.max_retries = max_retries
        
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
        self.cached_tensors = {}
        logger.info(f"Initialized {self.split} {self.media_type} dataset with {len(self.samples)} samples.")
        
        # Log class counts
        class_counts = {0: 0, 1: 0}
        for s in self.samples:
            class_counts[s["label"]] += 1
        logger.info(f"Class distribution: REAL (0): {class_counts[0]}, FAKE (1): {class_counts[1]}")

    def pre_cache_samples(self):
        """Pre-caches all samples to memory to avoid repeated disk reading and OpenCV video decoding/seeking."""
        logger.info(f"Pre-caching {len(self.samples)} samples for split '{self.split}' ({self.media_type})...")
        self.cached_tensors = {}
        
        if self.media_type == "video":
            # For video, just process each file sequentially
            total = len(self.samples)
            for i, sample in enumerate(self.samples):
                if i % 50 == 0:
                    logger.info(f"Pre-cached {i}/{total} videos...")
                file_path = sample["file_path"]
                try:
                    tensor = self.preprocessor.preprocess_video(file_path)
                    self.cached_tensors[i] = tensor
                except Exception as e:
                    logger.warning(f"Error pre-caching video {file_path}: {e}")
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
                tensor = self.preprocessor.preprocess_image(sample["file_path"])
                self.cached_tensors[idx] = tensor
            except Exception as e:
                logger.warning(f"Error pre-caching image sample {sample['file_path']}: {e}")
                
        # Load video frames efficiently
        total_videos = len(video_groups)
        for i, (file_path, frame_requests) in enumerate(video_groups.items()):
            if i % 100 == 0:
                logger.info(f"Pre-cached {i}/{total_videos} videos...")
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    logger.warning(f"Failed to open video file: {file_path}")
                    continue
                
                # Sort frame requests to read them sequentially (which is MUCH faster than random seeking!)
                frame_requests_sorted = sorted(frame_requests, key=lambda x: x[1])
                
                current_frame = 0
                for idx, frame_idx in frame_requests_sorted:
                    # Seek or read forward to the requested frame
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
                        continue
                        
                    tensor = self.preprocessor.preprocess_image(frame)
                    self.cached_tensors[idx] = tensor
                cap.release()
            except Exception as e:
                logger.warning(f"Error pre-caching video frames from {file_path}: {e}")
                
        logger.info(f"Completed pre-caching. Cached {len(self.cached_tensors)}/{len(self.samples)} samples.")

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

    def _extract_video_frame(self, file_path: str, frame_idx: int) -> any:
        """Helper to extract a specific frame from a video file using OpenCV."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {file_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame index {frame_idx} from video: {file_path}")
        return frame

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lazy loads, preprocesses, and returns a single item from the dataset."""
        if hasattr(self, 'cached_tensors') and idx in self.cached_tensors:
            return self.cached_tensors[idx], torch.tensor(self.samples[idx]["label"], dtype=torch.long)
            
        retries = 0
        current_idx = idx
        
        while retries <= self.max_retries:
            sample = self.samples[current_idx]
            file_path = sample["file_path"]
            label = sample["label"]
            origin = sample["sample_origin"]
            frame_idx = sample["frame_index"]
            
            try:
                if self.media_type == "image":
                    if origin == "video_frame":
                        # Load frame on-the-fly
                        frame = self._extract_video_frame(file_path, frame_idx)
                        tensor = self.preprocessor.preprocess_image(frame)
                    else:
                        tensor = self.preprocessor.preprocess_image(file_path)
                else:
                    tensor = self.preprocessor.preprocess_video(file_path)
                
                # Cache the successfully processed tensor
                if hasattr(self, 'cached_tensors'):
                    self.cached_tensors[current_idx] = tensor
                    
                return tensor, torch.tensor(label, dtype=torch.long)
                
            except Exception as e:
                logger.warning(
                    f"Failed to load/preprocess sample at index {current_idx} ({file_path}): {str(e)}. "
                    f"Retrying with another random sample..."
                )
                retries += 1
                current_idx = random.randint(0, len(self.samples) - 1)
                
        raise RuntimeError(
            f"Failed to load any valid sample after {self.max_retries} retries. "
            f"Last attempted index: {current_idx}."
        )
