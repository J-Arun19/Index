import cv2
import numpy as np
from typing import List, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger("FaceDetector")

class FaceDetector:
    """Uses OpenCV's Haar Cascade classifier to locate and crop faces in images/frames."""
    
    def __init__(self):
        # Load the default frontal face cascade from OpenCV
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            logger.error(f"Failed to load Haar Cascade from path: {cascade_path}")
        else:
            logger.info("Successfully loaded Haar Cascade Face Detector.")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in the given image.
        
        Args:
            image (np.ndarray): Image in BGR (OpenCV format) or RGB.
            
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes as (x, y, w, h).
        """
        if image is None:
            logger.warning("Empty image provided to face detection.")
            return []
            
        try:
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with standard parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Return list of coordinates
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        except Exception as e:
            logger.error(f"Error during face detection execution: {str(e)}")
            return []

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crops the face region from the image based on the bounding box coordinates.
        
        Args:
            image (np.ndarray): Original image.
            bbox (Tuple[int, int, int, int]): Face bounding box (x, y, w, h).
            
        Returns:
            np.ndarray: Cropped face image region.
        """
        if image is None:
            return None
            
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        # Guard coordinates to stay within image boundaries
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_w, x + w)
        y_end = min(img_h, y + h)
        
        # Crop and return
        return image[y_start:y_end, x_start:x_end]
