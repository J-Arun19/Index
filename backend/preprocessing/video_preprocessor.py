import os
import cv2
import torch
import numpy as np
from typing import Optional, List, Tuple
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector
from backend.training.dataset_exceptions import VideoDecodeError

# Disable OpenCV OpenCL globally to enforce CPU execution & suppress log flooding
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

logger = setup_logger("VideoPreprocessor")

class VideoPreprocessor:
    """Preprocesses video files by uniformly sampling frames, detecting/cropping faces, and standardizing sequence length."""
    
    def __init__(self, face_detector: Optional[FaceDetector] = None, max_frames: Optional[int] = None):
        self.detector = face_detector or FaceDetector()
        self.max_frames = max_frames if max_frames is not None else PipelineConfig.MAX_FRAMES_PER_VIDEO
        self.target_size = PipelineConfig.IMAGE_SIZE

    def preprocess_video(self, file_path: str, save_processed: bool = False, max_frames: Optional[int] = None) -> torch.Tensor:
        """
        Loads a video file, extracts uniformly sampled frames, preprocesses them, 
        and standardizes the sequence length to target max_frames (default 8).
        
        Args:
            file_path (str): Path to the video file.
            save_processed (bool): If True, saves intermediate resized/cropped frames.
            max_frames (int, optional): Number of frames to sample (defaults to self.max_frames).
            
        Returns:
            torch.Tensor: Preprocessed video tensor of shape (max_frames, 3, H, W)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file does not exist: {file_path}")
            
        # Validate format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in PipelineConfig.ALLOWED_VIDEO_EXT:
            raise ValueError(f"Unsupported video file extension '{ext}' for file: {file_path}")
            
        target_len = max_frames if max_frames is not None else self.max_frames
        cap = None
        frame_tensors = []
        last_valid_frame_tensor = None
        last_face_box = None
        prev_idx = None
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise VideoDecodeError(f"Failed to open video file (isOpened=False): {file_path}", sample_path=file_path)
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine target frame indices using uniform sampling across available duration
            if total_frames > 0:
                target_indices = np.linspace(0, total_frames - 1, target_len, dtype=int).tolist()
            else:
                # Fallback if frame count cannot be read
                target_indices = [i * PipelineConfig.FRAME_INTERVAL for i in range(target_len)]
                
            for idx in target_indices:
                # Short video optimization: if duplicate index requested and previous frame was valid
                if idx == prev_idx and last_valid_frame_tensor is not None:
                    frame_tensors.append(last_valid_frame_tensor.clone())
                    continue
                prev_idx = idx

                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                except Exception:
                    pass

                ret, frame = cap.read()
                if not ret or frame is None or getattr(frame, "size", 0) == 0:
                    # Sequential fallback read
                    ret, frame = cap.read()
                    if not ret or frame is None or getattr(frame, "size", 0) == 0:
                        # Pad with the last valid frame tensor if available
                        if last_valid_frame_tensor is not None:
                            frame_tensors.append(last_valid_frame_tensor.clone())
                        continue

                # Downscale high-resolution frames before face cropping to prevent memory pressure
                h, w = frame.shape[:2]
                if max(h, w) > 640:
                    scale = 640.0 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # Face detection with temporal consistency fallback
                faces = self.detector.detect_faces(frame)
                if faces:
                    last_face_box = faces[0]
                    processed_frame = self.detector.crop_face(frame, faces[0], margin_pct=0.08)
                elif last_face_box is not None:
                    # Reuse last known face bounding box from same video
                    processed_frame = self.detector.crop_face(frame, last_face_box, margin_pct=0.08)
                else:
                    processed_frame = frame

                # Resize to standard size (224, 224)
                processed_resized = cv2.resize(processed_frame, self.target_size)

                # Save processed copy if requested
                if save_processed:
                    filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{idx}.jpg"
                    processed_path = os.path.join(PipelineConfig.PROCESSED_DIR, "videos", filename)
                    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                    cv2.imwrite(processed_path, processed_resized)

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2RGB)

                # Normalize with ImageNet mean and std
                frame_norm = frame_rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                frame_norm = (frame_norm - mean) / std

                # Rearrange dimensions to CHW (3, 224, 224)
                frame_transposed = np.transpose(frame_norm, (2, 0, 1))

                frame_tensor = torch.tensor(frame_transposed, dtype=torch.float32)
                frame_tensors.append(frame_tensor)
                last_valid_frame_tensor = frame_tensor

                frame = None
                processed_frame = None
                processed_resized = None
                frame_rgb = None
                frame_norm = None
                frame_transposed = None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        if not frame_tensors:
            raise VideoDecodeError(f"No frames could be extracted from video file: {file_path}", sample_path=file_path)

        # Pad if short video produced fewer than target_len frames
        while len(frame_tensors) < target_len:
            frame_tensors.append(frame_tensors[-1].clone())

        # Truncate if more frames than target_len
        if len(frame_tensors) > target_len:
            frame_tensors = frame_tensors[:target_len]

        # Stack into tensor of shape (target_len, 3, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)
        return video_tensor
