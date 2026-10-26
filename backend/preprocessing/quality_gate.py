import os
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("QualityGate")


class MediaEligibility:
    SUPPORTED = "SUPPORTED"
    LOW_QUALITY = "LOW_QUALITY"
    UNSUPPORTED = "UNSUPPORTED"


class QualityGateResult:
    def __init__(
        self,
        is_suitable: bool,
        rejection_reason: Optional[str] = None,
        issues: Optional[List[str]] = None,
        resolution: Optional[Tuple[int, int]] = None,
        blur_score: float = 0.0,
        face_detected: bool = False,
        face_count: int = 0,
        primary_face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_dimensions: Optional[Tuple[int, int]] = None,
        eligibility: Optional[str] = None,
        media_classification: Optional[str] = None,
    ):
        self.is_suitable = is_suitable
        self.rejection_reason = rejection_reason
        self.issues = issues or []
        self.resolution = resolution or (0, 0)
        self.blur_score = blur_score
        self.face_detected = face_detected
        self.face_count = face_count
        self.primary_face_bbox = primary_face_bbox
        self.face_dimensions = face_dimensions
        if eligibility:
            self.eligibility = eligibility
        elif is_suitable:
            self.eligibility = MediaEligibility.SUPPORTED
        else:
            self.eligibility = MediaEligibility.LOW_QUALITY

        if media_classification:
            self.media_classification = media_classification
        elif is_suitable:
            self.media_classification = "Facial Media"
        else:
            self.media_classification = "Low Quality Facial Media"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_suitable": self.is_suitable,
            "eligibility": self.eligibility,
            "media_classification": self.media_classification,
            "status": "PASSED" if self.is_suitable else ("UNSUPPORTED" if self.eligibility == MediaEligibility.UNSUPPORTED else "REJECTED"),
            "rejection_reason": self.rejection_reason,
            "issues": self.issues,
            "resolution": {"width": self.resolution[0], "height": self.resolution[1]},
            "blur_score": round(self.blur_score, 2),
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "primary_face_bbox": self.primary_face_bbox,
            "face_dimensions": (
                {"width": self.face_dimensions[0], "height": self.face_dimensions[1]}
                if self.face_dimensions
                else None
            ),
        }


class ImageQualityGate:
    """
    Validates whether input images meet the fidelity requirements for forensic analysis.
    Prevents low-resolution, blurred, corrupted, or face-less inputs from triggering
    unjustified or confident verdicts.
    """

    MIN_IMAGE_WIDTH = 100
    MIN_IMAGE_HEIGHT = 100
    MIN_FACE_WIDTH = 50
    MIN_FACE_HEIGHT = 50
    BLUR_THRESHOLD = 25.0  # Laplacian variance below this indicates excessive blur

    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.detector = face_detector or FaceDetector()

    def evaluate(
        self,
        image_or_path: Any,
        require_face: bool = True,
    ) -> QualityGateResult:
        """
        Evaluates an image file path or numpy image array.

        Args:
            image_or_path: File path string or BGR numpy array.
            require_face: If True, abstains if no usable face is detected.

        Returns:
            QualityGateResult with detailed suitability metrics.
        """
        issues = []
        img = None

        # 1. Decode image if path provided
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                return QualityGateResult(
                    is_suitable=False,
                    rejection_reason=f"File not found: {image_or_path}",
                    issues=["File does not exist on disk."],
                )
            if os.path.getsize(image_or_path) == 0:
                return QualityGateResult(
                    is_suitable=False,
                    rejection_reason="Image file is empty (0 bytes).",
                    issues=["Empty file."],
                )
            try:
                img = cv2.imread(image_or_path)
            except Exception as e:
                return QualityGateResult(
                    is_suitable=False,
                    rejection_reason=f"Corrupt or unreadable image file: {str(e)}",
                    issues=[str(e)],
                )
        elif isinstance(image_or_path, np.ndarray):
            img = image_or_path
        else:
            return QualityGateResult(
                is_suitable=False,
                rejection_reason="Invalid image input type (expected file path or numpy array).",
                issues=["Invalid type."],
            )

        if img is None or img.size == 0:
            return QualityGateResult(
                is_suitable=False,
                rejection_reason="Failed to decode image data into valid pixel matrix.",
                issues=["Decoding failed."],
            )

        h, w = img.shape[:2]
        resolution = (w, h)

        # 2. Check resolution
        if w < self.MIN_IMAGE_WIDTH or h < self.MIN_IMAGE_HEIGHT:
            issues.append(
                f"Image resolution ({w}x{h}) is below the required minimum ({self.MIN_IMAGE_WIDTH}x{self.MIN_IMAGE_HEIGHT})."
            )

        # 3. Check face detection
        faces = self.detector.detect_faces(img)
        face_count = len(faces)
        primary_bbox = None
        face_dims = None

        if face_count > 0:
            primary_bbox = faces[0]
            fx, fy, fw, fh = primary_bbox
            face_dims = (fw, fh)
            if fw < self.MIN_FACE_WIDTH or fh < self.MIN_FACE_HEIGHT:
                issues.append(
                    f"Detected facial region is too small ({fw}x{fh}px) for forensic artifact verification "
                    f"(minimum required: {self.MIN_FACE_WIDTH}x{self.MIN_FACE_HEIGHT}px)."
                )
        elif require_face:
            issues.append(
                "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face."
            )

        # 4. Check blur via Laplacian variance (accounting for natural background bokeh)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            full_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            face_blur = None
            if face_count > 0 and primary_bbox:
                fx, fy, fw, fh = primary_bbox
                fx_c = max(0, fx)
                fy_c = max(0, fy)
                fw_c = min(fw, w - fx_c)
                fh_c = min(fh, h - fy_c)
                if fw_c > 10 and fh_c > 10:
                    face_gray = gray[fy_c : fy_c + fh_c, fx_c : fx_c + fw_c]
                    face_blur = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
            
            # For facial media, sharpness of the facial region determines forensic validity
            effective_blur = max(full_blur, face_blur) if face_blur is not None else full_blur
            blur_score = effective_blur
            
            if effective_blur < 15.0:
                issues.append(
                    f"Excessive image blur detected (sharpness score {effective_blur:.1f} < 15.0)."
                )
        except Exception as e:
            blur_score = 0.0
            issues.append(f"Sharpness evaluation error: {str(e)}")

        is_suitable = len(issues) == 0
        rejection_reason = "; ".join(issues) if issues else None

        if face_count == 0 and require_face:
            eligibility = MediaEligibility.UNSUPPORTED
            media_classification = "Document / Non-face image"
        elif not is_suitable:
            eligibility = MediaEligibility.LOW_QUALITY
            media_classification = "Low Quality Facial Media"
        else:
            eligibility = MediaEligibility.SUPPORTED
            media_classification = "Facial Media"

        return QualityGateResult(
            is_suitable=is_suitable,
            rejection_reason=rejection_reason,
            issues=issues,
            resolution=resolution,
            blur_score=blur_score,
            face_detected=(face_count > 0),
            face_count=face_count,
            primary_face_bbox=primary_bbox,
            face_dimensions=face_dims,
            eligibility=eligibility,
            media_classification=media_classification,
        )
