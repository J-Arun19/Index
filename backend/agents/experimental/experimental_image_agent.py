import os
import cv2
import time
import torch
from typing import Optional, Dict, Any

from backend.utils.logger import setup_logger
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.models.experimental.vit_detector import ViTDetector
from backend.models.experimental.xception_detector import XceptionDetector
from backend.models.image_model import ImageModel

logger = setup_logger("ExperimentalImageAgent")

class ExperimentalImageAgent:
    """
    Experimental forensic image analysis agent.
    Runs in isolation from the production pipeline.
    Allows testing ViT, Xception, or Baseline ResNet-18 backbones
    with full QualityGate, ForensicEvidenceSuite, and SafetyDecisionSystem preservation.
    """
    def __init__(self, backbone: str = "vit", device: Optional[str] = None):
        self.backbone_name = backbone.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        
        self._vit_model: Optional[ViTDetector] = None
        self._xception_model: Optional[XceptionDetector] = None
        self._baseline_model: Optional[ImageModel] = None

    @property
    def vit_model(self) -> ViTDetector:
        if self._vit_model is None:
            self._vit_model = ViTDetector(device=self.device)
        return self._vit_model

    @property
    def xception_model(self) -> XceptionDetector:
        if self._xception_model is None:
            self._xception_model = XceptionDetector(device=self.device)
        return self._xception_model

    @property
    def baseline_model(self) -> ImageModel:
        if self._baseline_model is None:
            from backend.utils.config import PipelineConfig
            model = ImageModel(num_classes=2, pretrained=False)
            if os.path.exists(PipelineConfig.IMAGE_MODEL_PATH):
                model.load_weights(PipelineConfig.IMAGE_MODEL_PATH)
            model.eval()
            self._baseline_model = model
        return self._baseline_model

    def analyze(self, file_path: str, require_face: bool = True, use_face_crop_for_vit: bool = False) -> Dict[str, Any]:
        """
        Executes forensic verification on an image file.
        Preserves all safety gates:
          - Unsupported non-face media -> NOT_SUITABLE
          - Ambiguous/low-quality media -> UNCERTAIN
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

        # Non-face media rejection
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
                img_shape=img_bgr.shape,
                raw_backbone_score=None
            )

        # 2. Forensic Analysis
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox)

        # Quality failure (e.g. extreme blur)
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
                img_shape=img_bgr.shape,
                raw_backbone_score=0.50
            )

        # 3. Model Inference based on chosen experimental backbone
        raw_fake_prob = 0.50
        try:
            if self.backbone_name == "vit":
                if use_face_crop_for_vit and quality_res.face_detected and face_bbox:
                    x, y, w, h = face_bbox
                    crop = img_bgr[y:y+h, x:x+w]
                    pred = self.vit_model.predict_image(crop)
                else:
                    pred = self.vit_model.predict_image(img_bgr)
                raw_fake_prob = pred["fake_probability"]

            elif self.backbone_name == "xception":
                if quality_res.face_detected and face_bbox:
                    x, y, w, h = face_bbox
                    crop = img_bgr[y:y+h, x:x+w]
                    pred = self.xception_model.predict_frame(crop)
                else:
                    pred = self.xception_model.predict_frame(img_bgr)
                raw_fake_prob = pred["fake_probability"]

            else:
                # Baseline ImageModel
                from backend.preprocessing.image_preprocessor import ImagePreprocessor
                prep = ImagePreprocessor()
                tensor = prep.preprocess_image(file_path).unsqueeze(0)
                pred_dict = self.baseline_model.predict(tensor)
                raw_fake_prob = float(pred_dict["probabilities"][0, 1].item())

        except Exception as e:
            logger.error(f"Experimental inference failed for {filename} with backbone {self.backbone_name}: {e}")
            raw_fake_prob = 0.50

        # 4. Three-State Safety Decision Fusion
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
            img_shape=img_bgr.shape,
            raw_backbone_score=raw_fake_prob
        )

    def _format_response(
        self,
        decision,
        quality_res,
        forensic_res: Dict[str, Any],
        filename: str,
        size: int,
        fmt: str,
        processing_time: float,
        img_shape: tuple,
        raw_backbone_score: Optional[float]
    ) -> Dict[str, Any]:
        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "resolution": f"{img_shape[1]}x{img_shape[0]}",
                "format": fmt,
                "experimental_backbone": self.backbone_name,
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
            "evidence": decision.evidence,
            "reason": decision.reason,
            "summary": decision.user_facing_statement,
            "user_facing_statement": decision.user_facing_statement,
            "quality_gate": {
                "is_suitable": quality_res.is_suitable,
                "eligibility": getattr(quality_res, "eligibility", "ELIGIBLE"),
                "issues": quality_res.issues,
                "face_detected": quality_res.face_detected,
                "blur_score": round(quality_res.blur_score, 2),
            },
            "forensics": forensic_res,
            "processing_time": processing_time,
        }
