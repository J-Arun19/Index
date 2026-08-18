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

    def get_cropped_resized_rgb(self, file_path_or_array) -> np.ndarray:
        """
        Loads an image, detects faces, crops to the first detected face if available, 
        and resizes to target_size, returning the RGB representation as a numpy array.
        """
        import numpy as np
        
        if isinstance(file_path_or_array, np.ndarray):
            img = file_path_or_array.copy()
            file_path = "numpy_array.jpg"
        else:
            file_path = file_path_or_array
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file does not exist: {file_path}")
                
            # Validate format
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in PipelineConfig.ALLOWED_IMAGE_EXT:
                raise ValueError(f"Unsupported image file extension '{ext}' for file: {file_path}")
                
            # Load image via OpenCV
            img = cv2.imread(file_path)
            
        if img is None:
            raise ValueError(f"Failed to read image data from: {file_path}")
            
        # Locate faces and crop if found
        faces = self.detector.detect_faces(img)
        if faces:
            logger.debug(f"Face detected in {file_path}. Cropping face region.")
            img = self.detector.crop_face(img, faces[0])
        else:
            logger.debug(f"No face detected in {file_path}. Processing full image.")
            
        # Resize to Target Size (e.g. 224 x 224)
        img_resized = cv2.resize(img, self.target_size)
        
        # Convert BGR (OpenCV standard) to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        return img_rgb

    def apply_augmentation_and_normalization(self, img_rgb: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """
        Applies mild training-time augmentations (if is_training is True), normalizes
        to [0, 1], applies ImageNet stats, and returns a PyTorch tensor.
        """
        import numpy as np
        
        img_processed = img_rgb.copy()
        
        # Apply training augmentations if requested
        if is_training and getattr(PipelineConfig, 'USE_AUGMENTATION', True):
            # 1. Horizontal flip (50% probability)
            if np.random.rand() > 0.5:
                img_processed = cv2.flip(img_processed, 1)
            
            # 2. Mild color/brightness variation (50% probability)
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # contrast adjustment
                beta = np.random.uniform(-10, 10)           # brightness adjustment
                img_processed = np.clip(alpha * img_processed.astype(np.float32) + beta, 0, 255).astype(np.uint8)
                
            # 3. JPEG compression artifact simulation (30% probability)
            if np.random.rand() > 0.7:
                quality = np.random.randint(70, 95)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                result, encimg = cv2.imencode('.jpg', img_processed, encode_param)
                if result:
                    img_processed = cv2.imdecode(encimg, 1)
                    
            # 4. Slight Gaussian Blur (20% probability)
            if np.random.rand() > 0.8:
                ksize = np.random.choice([3, 5])
                img_processed = cv2.GaussianBlur(img_processed, (ksize, ksize), 0)
        
        # Normalize: Convert pixel values to [0.0, 1.0] float
        img_normalized = img_processed.astype(np.float32) / 255.0
        
        # Apply standard ImageNet normalization statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Re-arrange dimensions from HWC to CHW (Channels, Height, Width) for PyTorch
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Create PyTorch FloatTensor
        return torch.tensor(img_transposed, dtype=torch.float32)

    def preprocess_image(self, file_path_or_array, save_processed: bool = False, is_training: bool = False) -> torch.Tensor:
        """
        Loads and processes an image file or pre-loaded image array.
        
        Args:
            file_path_or_array (str or np.ndarray): Path to the image file or a numpy image array.
            save_processed (bool): If True, saves intermediate resized/cropped image.
            is_training (bool): If True, applies training data augmentations.
            
        Returns:
            torch.Tensor: Preprocessed image tensor with shape (3, H, W)
        """
        img_rgb = self.get_cropped_resized_rgb(file_path_or_array)
        
        # Save processed copy to disk if requested
        if save_processed:
            # Reconvert to BGR for opencv writing
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if isinstance(file_path_or_array, str):
                filename = os.path.basename(file_path_or_array)
            else:
                filename = "numpy_array.jpg"
            processed_path = os.path.join(PipelineConfig.PROCESSED_DIR, "images", filename)
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            cv2.imwrite(processed_path, img_bgr)
            logger.info(f"Saved preprocessed image to {processed_path}")
            
        return self.apply_augmentation_and_normalization(img_rgb, is_training=is_training)

