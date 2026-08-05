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
        Loads a video file, extracts frames, preprocesses them, and standardizes 
        the sequence length to match MAX_FRAMES_PER_VIDEO.
        
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
            
        frame_tensors = []
        frame_idx = 0
        extracted_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames at configurable intervals
            if frame_idx % PipelineConfig.FRAME_INTERVAL == 0:
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
                    filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_frame_{extracted_count}.jpg"
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
                frame_tensors.append(torch.tensor(frame_transposed, dtype=torch.float32))
                extracted_count += 1
                
            frame_idx += 1
            
        cap.release()
        
        if not frame_tensors:
            raise ValueError(f"No frames could be extracted from video file: {file_path}")
            
        # Standardize the sequence length to MAX_FRAMES_PER_VIDEO (padding or truncating)
        target_len = PipelineConfig.MAX_FRAMES_PER_VIDEO
        n_frames = len(frame_tensors)
        
        if n_frames == target_len:
            final_tensors = frame_tensors
        elif n_frames > target_len:
            # Downsample frames evenly across the video duration
            indices = np.linspace(0, n_frames - 1, target_len, dtype=int)
            final_tensors = [frame_tensors[i] for i in indices]
            logger.info(f"Video {file_path} had {n_frames} frames. Downsampled to {target_len}.")
        else:
            # Pad by repeating the last valid frame
            final_tensors = list(frame_tensors)
            while len(final_tensors) < target_len:
                final_tensors.append(frame_tensors[-1].clone())
            logger.info(f"Video {file_path} had {n_frames} frames. Padded to {target_len}.")
            
        # Stack frames into a single tensor: shape (MAX_FRAMES_PER_VIDEO, 3, H, W)
        video_tensor = torch.stack(final_tensors, dim=0)
        return video_tensor
