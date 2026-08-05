import os
import cv2
import torch
import numpy as np
from typing import Optional
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("ImagePreprocessor")

class ImagePreprocessor:
    """Preprocesses images by cropping face areas, resizing, normalizing, and converting to PyTorch tensors."""
    
    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.detector = face_detector or FaceDetector()
        self.target_size = PipelineConfig.IMAGE_SIZE

    def preprocess_image(self, file_path: str, save_processed: bool = False) -> torch.Tensor:
        """
        Loads and processes an image file.
        
        Args:
            file_path (str): Path to the image file.
            save_processed (bool): If True, saves intermediate resized/cropped image.
            
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (3, H, W)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file does not exist: {file_path}")
            
        # Validate format
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in PipelineConfig.ALLOWED_IMAGE_EXT:
            raise ValueError(f"Unsupported image file extension '{ext}' for file: {file_path}")
            
        # Load image via OpenCV
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to read image file: {file_path}")
            
        # Locate faces and crop if found
        faces = self.detector.detect_faces(img)
        if faces:
            # We crop to the first detected face for binary classification
            logger.info(f"Face detected in {file_path}. Cropping face region.")
            img = self.detector.crop_face(img, faces[0])
        else:
            logger.warning(f"No face detected in {file_path}. Processing full image.")
            
        # Resize to Target Size (e.g. 224 x 224)
        img_resized = cv2.resize(img, self.target_size)
        
        # Save processed copy to disk if requested
        if save_processed:
            filename = os.path.basename(file_path)
            processed_path = os.path.join(PipelineConfig.PROCESSED_DIR, "images", filename)
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            cv2.imwrite(processed_path, img_resized)
            logger.info(f"Saved preprocessed image to {processed_path}")
            
        # Convert BGR (OpenCV standard) to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize: Convert pixel values to [0.0, 1.0] float
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Apply standard ImageNet normalization statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Re-arrange dimensions from HWC to CHW (Channels, Height, Width) for PyTorch
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Create PyTorch FloatTensor
        return torch.tensor(img_transposed, dtype=torch.float32)
