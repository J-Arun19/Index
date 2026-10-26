import os
import cv2
import numpy as np
from typing import List, Tuple
from backend.utils.logger import setup_logger

# Explicitly disable OpenCV OpenCL to enforce CPU-only execution and prevent CL_OUT_OF_RESOURCES errors
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
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

    @property
    def face_cascade_alt2(self):
        if getattr(self, '_face_cascade_alt2', None) is None:
            cv2.ocl.setUseOpenCL(False)
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                try:
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
                    self._face_cascade_alt2 = cv2.CascadeClassifier(cascade_path)
                    if self._face_cascade_alt2.empty():
                        self._face_cascade_alt2 = False
                except Exception:
                    self._face_cascade_alt2 = False
            else:
                self._face_cascade_alt2 = False
        return self._face_cascade_alt2 if self._face_cascade_alt2 is not False else None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_face_cascade'] = None
        state['_face_cascade_alt2'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._face_cascade = None
        self._face_cascade_alt2 = None

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

        if self.face_cascade is None and self.face_cascade_alt2 is None:
            return []
            
        try:
            # Convert to grayscale for Haar Cascade
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:
                gray = image
            else:
                return []
            
            # Downscale large image for fast Haar Cascade execution while preserving facial details
            h, w = gray.shape
            max_dim = 640
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                gray_small = cv2.resize(gray, (int(w * scale), int(h * scale)))
            else:
                gray_small = gray
            
            faces = ()
            # Stage 1: Detect faces with primary cascade
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray_small,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(24, 24)
                )
            
            # Stage 2: Sensitive alt2 cascade fallback for challenging lighting or non-frontal angles
            if len(faces) == 0 and self.face_cascade_alt2 is not None:
                faces = self.face_cascade_alt2.detectMultiScale(
                    gray_small,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(22, 22)
                )

            # Stage 3: Adaptive CLAHE equalization for low-contrast/backlit video frames
            if len(faces) == 0 and min(h, w) >= 50:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray_small)
                detector_to_use = self.face_cascade_alt2 if self.face_cascade_alt2 is not None else self.face_cascade
                if detector_to_use is not None:
                    faces = detector_to_use.detectMultiScale(
                        gray_clahe,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(20, 20)
                    )
            
            # Scale coordinates back to original size if resized
            candidate_boxes = []
            if scale != 1.0:
                for (x, y, w_box, h_box) in faces:
                    candidate_boxes.append((
                        int(x / scale),
                        int(y / scale),
                        int(w_box / scale),
                        int(h_box / scale)
                    ))
            else:
                candidate_boxes = [(int(x), int(y), int(w_box), int(h_box)) for (x, y, w_box, h_box) in faces]

            # Filter out non-face false positives (background, clothing, furniture)
            valid_boxes = [b for b in candidate_boxes if self.is_valid_face_region(image, b)]
            if not valid_boxes and candidate_boxes:
                valid_boxes = [b for b in candidate_boxes if 0.65 <= b[2] / max(1, b[3]) <= 1.35 and b[2] >= 28]

            # Apply Non-Maximum Suppression to remove overlapping duplicate boxes
            valid_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            nms_boxes = []
            for b in valid_boxes:
                bx, by, bw, bh = b
                overlap = False
                for nb in nms_boxes:
                    nx, ny, nw, nh = nb
                    xA = max(bx, nx)
                    yA = max(by, ny)
                    xB = min(bx + bw, nx + nw)
                    yB = min(by + bh, ny + nh)
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    union = bw * bh + nw * nh - inter
                    iou = inter / float(union) if union > 0 else 0.0
                    inter_b = inter / float(bw * bh) if bw * bh > 0 else 0.0
                    if iou > 0.20 or inter_b > 0.25:
                        overlap = True
                        break
                if not overlap:
                    nms_boxes.append(b)

            return nms_boxes
        except Exception as e:
            logger.error(f"Error during face detection execution: {str(e)}")
            return []

    @staticmethod
    def is_valid_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validates that a candidate bounding box contains an authentic human face structure,
        filtering out false positives on background objects, clothing, furniture, and noise.
        Uses skin chrominance locus in YCrCb/HSV and anatomical aspect ratio bounds.
        """
        if image is None:
            return False
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        if w < 24 or h < 24:
            return False
        aspect = w / float(max(1, h))
        if aspect < 0.60 or aspect > 1.40:
            return False
            
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_w, x + w)
        y_end = min(img_h, y + h)
        crop = image[y_start:y_end, x_start:x_end]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return False
            
        # YCrCb skin chrominance locus (universal across Fitzpatrick scale I-VI)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_ycrcb = (cr >= 128) & (cr <= 178) & (cb >= 70) & (cb <= 135)
        skin_ratio = float(np.mean(skin_ycrcb))
        
        # HSV skin locus fallback
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        skin_hsv = ((h_ch <= 28) | (h_ch >= 165)) & (s_ch >= 18) & (v_ch >= 30)
        hsv_ratio = float(np.mean(skin_hsv))
        
        # Genuine human faces contain substantial skin coverage (typically > 15-20%)
        # Background objects (walls, blue jeans, dark suits, carpets) have < 1-2%
        return max(skin_ratio, hsv_ratio) >= 0.15

    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.08) -> np.ndarray:
        """
        Crops the face region from the image based on the bounding box coordinates,
        with a calibrated contextual margin (default 8%) to ensure facial boundaries
        and blending seams are visible for forensic inspection.
        
        Args:
            image (np.ndarray): Original image.
            bbox (Tuple[int, int, int, int]): Face bounding box (x, y, w, h).
            margin_pct (float): Fractional padding to include hairline and jawline.
            
        Returns:
            np.ndarray: Cropped face image region.
        """
        if image is None:
            return None
            
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        
        mx = int(w * margin_pct)
        my = int(h * margin_pct)
        
        # Guard coordinates to stay within image boundaries
        x_start = max(0, x - mx)
        y_start = max(0, y - my)
        x_end = min(img_w, x + w + mx)
        y_end = min(img_h, y + h + my)
        
        # Crop and return
        return image[y_start:y_end, x_start:x_end]
