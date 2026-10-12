import os
import cv2
import torch
import numpy as np
from typing import Optional
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector
from backend.training.dataset_exceptions import ImageDecodeError, SamplePreprocessingError

# Disable OpenCV OpenCL globally to enforce CPU execution & suppress log noise
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

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
        if isinstance(file_path_or_array, np.ndarray):
            if file_path_or_array is None or getattr(file_path_or_array, "size", 0) == 0:
                raise SamplePreprocessingError("Provided image numpy array is empty", sample_path="numpy_array.jpg")
            img = file_path_or_array
            file_path = "numpy_array.jpg"
        else:
            file_path = str(file_path_or_array)
            if not os.path.exists(file_path):
                raise ImageDecodeError(f"Image file does not exist: {file_path}", sample_path=file_path)
                
            # Validate format
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in PipelineConfig.ALLOWED_IMAGE_EXT:
                raise ImageDecodeError(f"Unsupported image file extension '{ext}' for file: {file_path}", sample_path=file_path)
                
            # Load image via OpenCV
            try:
                img = cv2.imread(file_path)
            except cv2.error as e:
                raise ImageDecodeError(f"OpenCV error reading image file {file_path}: {e}", sample_path=file_path) from e
            
        if img is None or getattr(img, "size", 0) == 0:
            raise ImageDecodeError(f"Failed to read valid image data from: {file_path}", sample_path=file_path)
            
        # Locate faces and crop if found
        orig_img = img
        faces = self.detector.detect_faces(img)
        if faces:
            logger.debug(f"Face detected in {file_path}. Cropping face region.")
            cropped = self.detector.crop_face(img, faces[0])
            if cropped is not None and getattr(cropped, "size", 0) > 0 and cropped.shape[0] > 0 and cropped.shape[1] > 0:
                img = cropped
            else:
                img = orig_img
        else:
            logger.debug(f"No face detected in {file_path}. Processing full image.")
            
        if img is None or getattr(img, "size", 0) == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            raise SamplePreprocessingError(f"Empty image array cannot be resized: {file_path}", sample_path=file_path)

        # Resize to Target Size (e.g. 224 x 224) and Convert BGR to RGB
        try:
            img_resized = cv2.resize(img, self.target_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            return img_rgb
        except cv2.error as e:
            raise SamplePreprocessingError(f"OpenCV resize/color transform error for {file_path}: {e}", sample_path=file_path) from e

    def simulate_jpeg_compression(self, img_rgb: np.ndarray, quality: int) -> np.ndarray:
        """
        Simulates JPEG compression artifacts while strictly guaranteeing RGB channel consistency.
        Converts RGB -> BGR for OpenCV encoding, and decodes back BGR -> RGB.
        """
        if img_rgb is None or getattr(img_rgb, "size", 0) == 0:
            return img_rgb
        try:
            q = int(np.clip(quality, 10, 100))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
            if result and encimg is not None:
                dec_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
                if dec_bgr is not None:
                    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"simulate_jpeg_compression error: {e}. Falling back to uncompressed RGB.")
        return img_rgb

    def apply_forensic_augmentations(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Applies mild, forensic-preserving augmentations specifically designed for deepfake detection:
        1. Random horizontal flip (p=0.50)
        2. Mild scale/crop jitter (scale 0.90-1.00, p=0.40)
        3. Mild rotation (-8 to +8 deg, reflect padding, p=0.35)
        4. Photometric variation (brightness/contrast +/-12%, saturation +/-8%, p=0.40)
        5. Safe RGB JPEG compression (quality 65-95, p=0.35)
        6. Mild Gaussian blur (k=3 or 5, p=0.20)
        7. Mild additive Gaussian sensor noise (sigma 2-6, p=0.25)
        
        Strictly avoids forensic-destructive operations (no cutout, no extreme blur, no heavy compression).
        """
        img = img_rgb.copy()
        h, w = img.shape[:2]

        # 1. Random horizontal flip (p=0.50)
        if np.random.rand() < 0.50:
            img = cv2.flip(img, 1)

        # 2. Mild scale/crop jitter (p=0.40, scale 0.90 - 1.00)
        if np.random.rand() < 0.40:
            scale = np.random.uniform(0.90, 1.00)
            ch, cw = int(h * scale), int(w * scale)
            dy = np.random.randint(0, max(1, h - ch + 1))
            dx = np.random.randint(0, max(1, w - cw + 1))
            cropped = img[dy:dy + ch, dx:dx + cw]
            img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. Mild rotation (p=0.35, -8 to +8 deg with reflect border to avoid black frames)
        if np.random.rand() < 0.35:
            angle = np.random.uniform(-8.0, 8.0)
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 4. Photometric variation (p=0.40: saturation +/-8%, contrast +/-12%, brightness +/-12%)
        if np.random.rand() < 0.40:
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                sat_factor = np.random.uniform(0.92, 1.08)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            except Exception:
                pass
            alpha = np.random.uniform(0.88, 1.12)
            beta = np.random.uniform(-30.0, 30.0)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 5. Safe RGB JPEG compression simulation (p=0.35, Q=65-95)
        if np.random.rand() < 0.35:
            q = int(np.random.randint(65, 96))
            img = self.simulate_jpeg_compression(img, quality=q)

        # 6. Mild Gaussian blur (p=0.20, k=3 or 5, sigma=0.5-1.0)
        if np.random.rand() < 0.20:
            ksize = int(np.random.choice([3, 5]))
            sigma = float(np.random.uniform(0.5, 1.0))
            img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

        # 7. Mild additive Gaussian sensor noise (p=0.25, sigma=2.0-6.0)
        if np.random.rand() < 0.25:
            sigma_noise = float(np.random.uniform(2.0, 6.0))
            noise = np.random.normal(0, sigma_noise, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def apply_augmentation_and_normalization(self, img_rgb: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """
        Applies forensic-preserving training augmentations (if is_training is True), normalizes
        to [0, 1], applies ImageNet stats, and returns a PyTorch tensor.
        Validation and test paths (is_training=False) strictly bypass all augmentations.
        """
        # Apply training augmentations only when is_training is True and USE_AUGMENTATION is enabled
        if is_training and getattr(PipelineConfig, 'USE_AUGMENTATION', True):
            img_processed = self.apply_forensic_augmentations(img_rgb)
        else:
            img_processed = img_rgb

        # Convert to float tensor, permute HWC to CHW, scale and normalize using vectorized tensor operations
        tensor = torch.from_numpy(img_processed).permute(2, 0, 1).to(torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor / 255.0 - mean) / std

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

