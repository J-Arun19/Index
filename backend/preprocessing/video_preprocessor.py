import os
import cv2
import torch
import numpy as np
from typing import Optional, List
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("VideoPreprocessor")

class VideoPreprocessor:
    """Preprocesses video files by sampling frames, detecting/cropping faces, and standardizing sequence length."""
    
    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.detector = face_detector or FaceDetector()
        self.target_size = PipelineConfig.IMAGE_SIZE

    def preprocess_video(self, file_path: str, save_processed: bool = False) -> torch.Tensor:
        """
        Loads a video file, extracts frames efficiently using seeking, preprocesses them, 
        and standardizes the sequence length to match MAX_FRAMES_PER_VIDEO.
        
        Args:
            file_path (str): Path to the video file.
            save_processed (bool): If True, saves intermediate resized/cropped frames.
            
        Returns:
            torch.Tensor: Preprocessed video tensor of shape (MAX_FRAMES_PER_VIDEO, 3, H, W)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file does not exist: {file_path}")
            
        # Validate format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in PipelineConfig.ALLOWED_VIDEO_EXT:
            raise ValueError(f"Unsupported video file extension '{ext}' for file: {file_path}")
            
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {file_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_len = PipelineConfig.MAX_FRAMES_PER_VIDEO
        
        # Determine target frame indices to decode directly
        if total_frames > 0:
            target_indices = np.linspace(0, total_frames - 1, target_len, dtype=int).tolist()
        else:
            # Fallback if frame count cannot be read
            target_indices = [i * PipelineConfig.FRAME_INTERVAL for i in range(target_len)]
            
        frame_tensors = []
        last_valid_frame_tensor = None
        
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # If seek/read fails, try to fall back to the next sequential frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If fallback fails, pad with the last valid frame tensor
                    if last_valid_frame_tensor is not None:
                        frame_tensors.append(last_valid_frame_tensor.clone())
                    continue
                    
            # Detect and crop face
            faces = self.detector.detect_faces(frame)
            if faces:
                processed_frame = self.detector.crop_face(frame, faces[0])
            else:
                processed_frame = frame
                
            # Resize
            processed_resized = cv2.resize(processed_frame, self.target_size)
            
            # Save processed copy if requested
            if save_processed:
                filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{idx}.jpg"
                processed_path = os.path.join(PipelineConfig.PROCESSED_DIR, "videos", filename)
                os.makedirs(os.path.dirname(processed_path), exist_ok=True)
                cv2.imwrite(processed_path, processed_resized)
                
            # Convert BGR (OpenCV standard) to RGB
            frame_rgb = cv2.cvtColor(processed_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize: Convert pixel values to [0.0, 1.0] float
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization stats
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frame_norm = (frame_norm - mean) / std
            
            # Re-arrange dimensions from HWC to CHW for PyTorch
            frame_transposed = np.transpose(frame_norm, (2, 0, 1))
            
            # Convert to tensor and store
            frame_tensor = torch.tensor(frame_transposed, dtype=torch.float32)
            frame_tensors.append(frame_tensor)
            last_valid_frame_tensor = frame_tensor
            
        cap.release()
        
        if not frame_tensors:
            raise ValueError(f"No frames could be extracted from video file: {file_path}")
            
        # Pad if we couldn't load enough frames
        while len(frame_tensors) < target_len:
            frame_tensors.append(frame_tensors[-1].clone())
            
        # Stack frames into a single tensor: shape (MAX_FRAMES_PER_VIDEO, 3, H, W)
        video_tensor = torch.stack(frame_tensors, dim=0)
        return video_tensor
