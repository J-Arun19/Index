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
    ILLUSTRATION = "ILLUSTRATION"


class MediaTypeAssessment:
    PHOTOGRAPHIC = "PHOTOGRAPHIC"
    ILLUSTRATION_OR_CARTOON = "ILLUSTRATION_OR_CARTOON"
    UNCERTAIN = "UNCERTAIN"


class MediaTypeDetector:
    """
    Evaluates whether an image is Photographic (natural optical camera capture),
    an Illustration / Cartoon / Anime / 2D Art, or Uncertain based on
    multi-signal statistical texture, cel-shading flatness, and anatomical facial geometry.
    """
    PHOTOGRAPHIC = MediaTypeAssessment.PHOTOGRAPHIC
    ILLUSTRATION_OR_CARTOON = MediaTypeAssessment.ILLUSTRATION_OR_CARTOON
    UNCERTAIN = MediaTypeAssessment.UNCERTAIN

    def __init__(self, detector: Optional[FaceDetector] = None):
        self.detector = detector or FaceDetector()

    def evaluate(
        self,
        img_bgr: np.ndarray,
        face_records: Optional[List[Dict[str, Any]]] = None,
        vision_assessment: Optional[str] = None,
    ) -> Dict[str, Any]:
        if img_bgr is None or img_bgr.size == 0:
            return {
                "media_type": self.UNCERTAIN,
                "confidence": 0.0,
                "reasons": ["Invalid image array"],
                "scores": {},
                "explanation": "Media type could not be evaluated.",
            }

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        scores: Dict[str, Any] = {}
        reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Cel-Shading & Flat Color Fills (Texture Uniformity)
        # ---------------------------------------------------------------------
        kernel_size = 5
        mean_gray = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        sq_gray = cv2.blur((gray.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        local_var = np.maximum(0, sq_gray - mean_gray ** 2)
        flat_pixel_ratio = float(np.mean(local_var < 3.5))
        scores["flat_pixel_ratio"] = round(flat_pixel_ratio, 4)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        very_low_grad_ratio = float(np.mean(grad_mag < 2.0))
        high_grad_ratio = float(np.mean(grad_mag > 50.0))
        scores["very_low_grad_ratio"] = round(very_low_grad_ratio, 4)
        scores["high_grad_ratio"] = round(high_grad_ratio, 4)

        # ---------------------------------------------------------------------
        # 2. Color Palette Quantization & Entropy
        # ---------------------------------------------------------------------
        small = cv2.resize(img_bgr, (128, 128)).astype(np.int32)
        q16 = small >> 4
        flat_colors = q16[:, :, 0] * 256 + q16[:, :, 1] * 16 + q16[:, :, 2]
        _, counts = np.unique(flat_colors, return_counts=True)
        probs = counts / counts.sum()
        color_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        scores["color_entropy"] = round(color_entropy, 3)

        # ---------------------------------------------------------------------
        # 3. Facial Morphology & Stylization Geometry
        # ---------------------------------------------------------------------
        is_stylized_face = False
        face_stylization_signals: List[str] = []
        if face_records is None:
            if hasattr(self.detector, "detect_faces_with_landmarks"):
                face_records = self.detector.detect_faces_with_landmarks(img_bgr)
            else:
                faces = self.detector.detect_faces(img_bgr)
                face_records = [{"bbox": f} for f in faces]

        if face_records and len(face_records) > 0:
            primary = face_records[0]
            fx, fy, fw, fh = primary.get("bbox", (0, 0, 0, 0))
            lm = primary.get("landmarks", {})
            if fw > 20 and fh > 20:
                leye = lm.get("left_eye_box", [0, 0, 0, 0])
                reye = lm.get("right_eye_box", [0, 0, 0, 0])
                mouth = lm.get("mouth_box", [0, 0, 0, 0])
                eye_w = max(leye[2], reye[2])
                eye_h = max(leye[3], reye[3])

                eye_w_ratio = eye_w / max(1.0, float(fw))
                eye_h_ratio = eye_h / max(1.0, float(fh))
                mouth_w = float(mouth[2])
                eyes_to_mouth_ratio = eye_w / max(1.0, mouth_w)

                scores["eye_w_ratio"] = round(eye_w_ratio, 3)
                scores["eye_h_ratio"] = round(eye_h_ratio, 3)
                scores["eyes_to_mouth_ratio"] = round(eyes_to_mouth_ratio, 3)

                # Human biological invariants vs 2D stylized art:
                # Human eye width is 18-27% of face width, eye height is 10-22% of face height.
                # In anime/cartoon art, eye width > 30% of face width or eye height > 28% of face height,
                # and eye width is > 50% of mouth width.
                if eye_w_ratio > 0.31 or eye_h_ratio > 0.30:
                    face_stylization_signals.append(
                        f"Disproportionately large stylized eye geometry (eye width ratio: {eye_w_ratio:.2f}, eye height ratio: {eye_h_ratio:.2f}) characteristic of 2D anime/illustration."
                    )
                if eyes_to_mouth_ratio > 0.55:
                    face_stylization_signals.append(
                        f"Non-photographic eye-to-mouth ratio ({eyes_to_mouth_ratio:.2f}) typical of stylized character illustration."
                    )

                # Cheek skin texture: real human skin has non-zero biological micro-texture/pores;
                # anime has flat cel-shading fill.
                cheeks_box = lm.get("cheeks_box")
                if cheeks_box:
                    cx, cy, cw, ch = cheeks_box
                    cx_c = max(0, cx)
                    cy_c = max(0, cy)
                    cw_c = min(cw, w - cx_c)
                    ch_c = min(ch, h - cy_c)
                    if cw_c > 10 and ch_c > 10:
                        cheek_crop = gray[cy_c : cy_c + ch_c, cx_c : cx_c + cw_c]
                        c_mean = cv2.blur(cheek_crop.astype(np.float32), (5, 5))
                        c_sq = cv2.blur(cheek_crop.astype(np.float32) ** 2, (5, 5))
                        c_var = np.maximum(0, c_sq - c_mean ** 2)
                        cheek_flat_ratio = float(np.mean(c_var < 2.0))
                        scores["cheek_flat_ratio"] = round(cheek_flat_ratio, 3)
                        if cheek_flat_ratio > 0.40 and len(face_stylization_signals) > 0:
                            face_stylization_signals.append(
                                f"Cheek region exhibits flat cel-shading fill (flat ratio: {cheek_flat_ratio:.2f}) rather than organic skin texture."
                            )

        if len(face_stylization_signals) >= 1:
            is_stylized_face = True
            reasons.extend(face_stylization_signals)

        # ---------------------------------------------------------------------
        # 4. Inked Line Stroke Contours (Black Top-Hat / Outline Detection)
        # ---------------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        ink_line_pixels = black_hat > 8
        ink_line_ratio = float(np.mean(ink_line_pixels))
        scores["ink_line_ratio"] = round(ink_line_ratio, 4)

        # ---------------------------------------------------------------------
        # 5. Multi-modal / Vision Assessment (if available)
        # ---------------------------------------------------------------------
        vision_says_illustration = False
        if vision_assessment:
            v_text = str(vision_assessment).lower()
            if any(term in v_text for term in ["anime", "cartoon", "illustration", "illustrative", "drawing", "stylized art", "2d artwork"]):
                vision_says_illustration = True
                reasons.append("Multi-modal vision analysis flagged non-photographic illustrative medium.")

        # ---------------------------------------------------------------------
        # 6. Synthesize Verdict & Confidence
        # ---------------------------------------------------------------------
        illustration_evidence_count = 0
        if is_stylized_face:
            illustration_evidence_count += 3
        if scores.get("eye_w_ratio", 0) > 0.30:
            illustration_evidence_count += 2
        if scores.get("cheek_flat_ratio", 0) > 0.35:
            illustration_evidence_count += 1
        if vision_says_illustration:
            illustration_evidence_count += 3

        scores["illustration_evidence_count"] = illustration_evidence_count

        if illustration_evidence_count >= 3:
            verdict = self.ILLUSTRATION_OR_CARTOON
            confidence = min(0.98, 0.70 + 0.08 * illustration_evidence_count)
            explanation = (
                "This image appears to be a stylized illustration rather than a photographic image. "
                "The photographic deepfake detection models are not sufficient to determine whether this "
                "artwork was created by a human or generated by AI."
            )
        elif illustration_evidence_count == 0 and scores.get("eye_w_ratio", 0) < 0.28:
            verdict = self.PHOTOGRAPHIC
            confidence = 0.90
            explanation = "Media exhibits optical and anatomical characteristics consistent with photographic imagery."
        else:
            verdict = self.UNCERTAIN
            confidence = 0.50
            explanation = (
                "Media type could not be definitively determined as photographic or non-photographic. "
                "Evidence lies in the ambiguous margin."
            )

        return {
            "media_type": verdict,
            "confidence": round(confidence, 3),
            "reasons": reasons,
            "explanation": explanation,
            "scores": scores,
        }


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
        all_face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        facial_landmarks: Optional[Dict[str, Any]] = None,
        media_type: Optional[str] = None,
        media_type_details: Optional[Dict[str, Any]] = None,
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
        self.all_face_bboxes = all_face_bboxes or ([primary_face_bbox] if primary_face_bbox else [])
        self.facial_landmarks = facial_landmarks or {}
        self.media_type = media_type or MediaTypeAssessment.PHOTOGRAPHIC
        self.media_type_details = media_type_details or {}

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
            "media_type": self.media_type,
            "media_type_details": self.media_type_details,
            "status": "PASSED" if self.is_suitable else ("UNSUPPORTED" if self.eligibility == MediaEligibility.UNSUPPORTED else "REJECTED"),
            "rejection_reason": self.rejection_reason,
            "issues": self.issues,
            "resolution": {"width": self.resolution[0], "height": self.resolution[1]},
            "blur_score": round(self.blur_score, 2),
            "face_detected": self.face_detected,
            "face_count": self.face_count,
            "primary_face_bbox": self.primary_face_bbox,
            "all_face_bboxes": self.all_face_bboxes,
            "facial_landmarks": self.facial_landmarks,
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
        self.media_detector = MediaTypeDetector(detector=self.detector)

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

        # 3. Check face detection with anatomical landmark extraction
        face_records = []
        if hasattr(self.detector, "detect_faces_with_landmarks"):
            face_records = self.detector.detect_faces_with_landmarks(img)
            faces = [r["bbox"] for r in face_records]
        else:
            faces = self.detector.detect_faces(img)

        face_count = len(faces)
        primary_bbox = None
        face_dims = None
        facial_landmarks = {}

        if face_count > 0:
            primary_bbox = faces[0]
            fx, fy, fw, fh = primary_bbox
            face_dims = (fw, fh)
            if face_records and "landmarks" in face_records[0]:
                facial_landmarks = face_records[0]["landmarks"]
            if fw < self.MIN_FACE_WIDTH or fh < self.MIN_FACE_HEIGHT:
                issues.append(
                    f"Detected facial region is too small ({fw}x{fh}px) for forensic artifact verification "
                    f"(minimum required: {self.MIN_FACE_WIDTH}x{self.MIN_FACE_HEIGHT}px)."
                )
        elif require_face:
            issues.append(
                "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face."
            )

        # 4. Check media type (Photographic vs Illustration/Anime/Cartoon vs Uncertain)
        media_eval = self.media_detector.evaluate(img, face_records=face_records)
        media_type = media_eval.get("media_type", MediaTypeAssessment.PHOTOGRAPHIC)

        if media_type == MediaTypeAssessment.ILLUSTRATION_OR_CARTOON:
            issues.append(
                "This image appears to be a stylized illustration rather than a photographic image. "
                "The photographic deepfake detection models are not sufficient to determine whether this "
                "artwork was created by a human or generated by AI."
            )

        # 5. Check blur via Laplacian variance (accounting for natural background bokeh)
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

        if media_type == MediaTypeAssessment.ILLUSTRATION_OR_CARTOON:
            eligibility = MediaEligibility.UNSUPPORTED
            media_classification = "Illustration / Anime / Cartoon"
        elif face_count == 0 and require_face:
            eligibility = MediaEligibility.UNSUPPORTED
            media_classification = "Document / Non-face image"
        elif not is_suitable:
            eligibility = MediaEligibility.LOW_QUALITY
            media_classification = "Low Quality Facial Media"
        elif media_type == MediaTypeAssessment.UNCERTAIN:
            eligibility = MediaEligibility.SUPPORTED
            media_classification = "Facial Media (Uncertain Photographic Provenance)"
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
            all_face_bboxes=faces,
            facial_landmarks=facial_landmarks,
            media_type=media_type,
            media_type_details=media_eval,
        )
