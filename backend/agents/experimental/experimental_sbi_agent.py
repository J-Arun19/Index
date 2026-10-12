import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple

from backend.utils.logger import setup_logger
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.preprocessing.face_detector import FaceDetector
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.models.experimental.sbi_detector import SBIDetector

logger = setup_logger("ExperimentalSBIImageAgent")

class ExperimentalSBIImageAgent:
    """
    Experimental forensic image analysis agent using Self-Blended Images (SBI).
    Runs strictly in isolation from production pipelines.
    Preserves all safety gates:
      - Unsupported non-face media -> NOT_SUITABLE
      - Low-quality or ambiguous media -> UNCERTAIN
      - Boundary-aware face extraction with official 25% margin
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        self.face_detector = FaceDetector()
        self.detector = SBIDetector(weights_path=weights_path, device=self.device)

    def extract_sbi_face_crop(
        self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.25
    ) -> np.ndarray:
        """
        Extracts face crop with boundary margin as specified in official SBI codebase.
        Official SBI margin is w/4 and h/4 (margin_ratio = 0.25).
        This ensures facial mask perimeter boundaries (hairline, jawline) are preserved.
        """
        x, y, w, h = bbox
        H, W = img_bgr.shape[:2]

        w_margin = int(w * margin_ratio)
        h_margin = int(h * margin_ratio)

        x0 = max(0, x - w_margin)
        y0 = max(0, y - h_margin)
        x1 = min(W, x + w + w_margin)
        y1 = min(H, y + h + h_margin)

        return img_bgr[y0:y1, x0:x1]

    def analyze(
        self,
        file_path: str,
        require_face: bool = True,
        crop_mode: str = "sbi_margin",  # "sbi_margin" (official w/4), "tight_crop", "full_image"
    ) -> Dict[str, Any]:
        """
        Executes isolated forensic analysis on an image file.
        crop_mode:
          - "sbi_margin": official SBI face crop with 25% boundary expansion (intended).
          - "tight_crop": standard Haar bbox without expansion.
          - "full_image": whole image resized to 380x380 (exploratory).
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "PNG"

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            raise ValueError(f"Could not decode image at {file_path}")

        # 1. Quality Gate
        quality_res = self.quality_gate.evaluate(img_bgr, require_face=require_face)

        if getattr(quality_res, "eligibility", None) == "UNSUPPORTED":
            decision = self.decision_system.decide(
                raw_fake_prob=None,
                quality_result=quality_res,
                forensic_result={},
            )
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res={},
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=round(time.time() - start_time, 4),
                raw_backbone_score=None,
                crop_mode=crop_mode,
            )

        # 2. Forensic Analysis
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox)

        # Quality failure
        if not quality_res.is_suitable:
            decision = self.decision_system.decide(
                raw_fake_prob=0.50,
                quality_result=quality_res,
                forensic_result=forensic_res,
            )
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res=forensic_res,
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=round(time.time() - start_time, 4),
                raw_backbone_score=0.50,
                crop_mode=crop_mode,
            )

        # 3. SBI Inference
        raw_fake_prob = 0.50
        try:
            if crop_mode == "full_image":
                input_crop = img_bgr
            elif crop_mode == "tight_crop":
                if quality_res.face_detected and face_bbox:
                    x, y, w, h = face_bbox
                    input_crop = img_bgr[y:y+h, x:x+w]
                else:
                    input_crop = img_bgr
            else:
                # "sbi_margin" (official)
                if quality_res.face_detected and face_bbox:
                    input_crop = self.extract_sbi_face_crop(img_bgr, face_bbox, margin_ratio=0.25)
                else:
                    input_crop = img_bgr

            pred = self.detector.predict_crop(input_crop)
            raw_fake_prob = pred["fake_probability"]
        except Exception as e:
            logger.error(f"Inference error with SBI detector: {e}")
            raw_fake_prob = 0.50

        # 4. Safety Decision
        decision = self.decision_system.decide(
            raw_fake_prob=raw_fake_prob,
            quality_result=quality_res,
            forensic_result=forensic_res,
        )

        return self._format_response(
            decision=decision,
            quality_res=quality_res,
            forensic_res=forensic_res,
            filename=filename,
            size=size,
            fmt=fmt,
            processing_time=round(time.time() - start_time, 4),
            raw_backbone_score=raw_fake_prob,
            crop_mode=crop_mode,
        )

    def _format_response(
        self,
        decision: Any,
        quality_res: Any,
        forensic_res: Dict[str, Any],
        filename: str,
        size: int,
        fmt: str,
        processing_time: float,
        raw_backbone_score: Optional[float],
        crop_mode: str,
    ) -> Dict[str, Any]:
        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "format": fmt,
                "crop_mode": crop_mode,
                "experimental_backbone": "sbi_efficientnet_b4",
                "face_detected": getattr(quality_res, "face_detected", False),
            },
            "prediction": decision.prediction,
            "verdict": decision.verdict,
            "confidence": decision.confidence,
            "confidence_score": decision.confidence,
            "confidence_level": decision.confidence_level,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": raw_backbone_score,
            "uncertainty": decision.uncertainty,
            "is_hard_negative": decision.is_hard_negative,
            "reasons": [decision.reason] if decision.reason else [],
            "user_facing_statement": decision.user_facing_statement,
            "processing_time_seconds": processing_time,
            "evidence": decision.evidence,
        }
