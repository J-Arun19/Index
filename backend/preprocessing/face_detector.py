import os
import cv2
import numpy as np
from typing import List, Tuple
from backend.utils.logger import setup_logger

# Explicitly disable OpenCV OpenCL to enforce CPU-only execution and prevent CL_OUT_OF_RESOURCES errors
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

logger = setup_logger("FaceDetector")

class FaceDetector:
    """Uses OpenCV's Haar Cascade classifier to locate and crop faces in images/frames."""
    
    def __init__(self):
        # Enforce CPU execution path in OpenCV for Haar face detection
        cv2.ocl.setUseOpenCL(False)
        assert cv2.ocl.useOpenCL() == False, "Failed to disable OpenCV OpenCL execution path!"
        self._face_cascade = None

    @property
    def face_cascade(self):
        if getattr(self, '_face_cascade', None) is None:
            cv2.ocl.setUseOpenCL(False)
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self._face_cascade = cv2.CascadeClassifier(cascade_path)
                    if self._face_cascade.empty():
                        logger.error(f"Failed to load Haar Cascade from path: {cascade_path}")
                        self._face_cascade = False
                    else:
                        logger.info(f"Successfully loaded Haar Cascade Face Detector (CPU-only, OpenCL enabled: {cv2.ocl.useOpenCL()}).")
                except Exception as e:
                    logger.warning(f"Failed to initialize Haar Cascade: {e}")
                    self._face_cascade = False
            else:
                self._face_cascade = False
        return self._face_cascade if self._face_cascade is not False else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_face_cascade'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._face_cascade = None

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

        if self.face_cascade is None:
            return []
            
        try:
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Downscale large image for significantly faster Haar Cascade execution on CPU
            h, w = gray.shape
            max_dim = 320
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                gray_small = gray
            
            # Detect faces with standard parameters
            faces = self.face_cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Scale coordinates back to original size if resized
            if scale != 1.0:
                scaled_faces = []
                for (x, y, w_box, h_box) in faces:
                    scaled_faces.append((
                        int(x / scale),
                        int(y / scale),
                        int(w_box / scale),
                        int(h_box / scale)
                    ))
                return scaled_faces
            
            # Return list of coordinates
            return [(int(x), int(y), int(w_box), int(h_box)) for (x, y, w_box, h_box) in faces]
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
