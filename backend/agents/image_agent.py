import os
import cv2
import time
import torch
from typing import Optional, Dict, Any

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.models.image_model import ImageModel
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState

logger = setup_logger("ImageAgent")


class ImageAgent:
    """
    Forensic image analysis agent.
    Combines input quality validation, genuine frequency/compression forensics,
    calibrated neural network inference, and a three-state decision engine.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PipelineConfig.IMAGE_MODEL_PATH
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        self.preprocessor = ImagePreprocessor()
        self._model: Optional[ImageModel] = None

    @property
    def model(self) -> ImageModel:
        """Lazily initializes and caches the neural network."""
        if self._model is None:
            model = ImageModel(num_classes=2, pretrained=False)
            if os.path.exists(self.weights_path):
                try:
                    model.load_weights(self.weights_path)
                    logger.info(f"ImageAgent: Loaded model weights from {self.weights_path}")
                except Exception as e:
                    logger.error(f"ImageAgent: Failed to load weights from {self.weights_path}: {e}")
            else:
                logger.warning(f"ImageAgent: No weights found at {self.weights_path}. Model uninitialized.")
            model.eval()
            self._model = model
        return self._model

    def analyze(self, file_path: str, require_face: bool = True) -> Dict[str, Any]:
        """
        Executes complete forensic verification on an image file.
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "PNG"

        # 1. Image Quality Gate
        img_bgr = cv2.imread(file_path)
        quality_res = self.quality_gate.evaluate(img_bgr, require_face=require_face)

        # Separate media suitability from authenticity:
        # If media is unsupported (no human face detected), bypass model and forensic analysis
        if getattr(quality_res, "eligibility", None) == "UNSUPPORTED":
            decision = self.decision_system.decide(
                raw_fake_prob=None,
                quality_result=quality_res,
                forensic_result={},
            )
            processing_time = round(time.time() - start_time, 4)
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res={},
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=processing_time,
                img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
            )

        # 2. Genuine Forensic Evidence
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox) if img_bgr is not None else {}

        # If quality gate failed (LOW QUALITY: blurry or face too small), abstain via decision system
        if not quality_res.is_suitable:
            decision = self.decision_system.decide(
                raw_fake_prob=0.50,
                quality_result=quality_res,
                forensic_result=forensic_res,
            )
            processing_time = round(time.time() - start_time, 4)
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res=forensic_res,
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=processing_time,
                img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
            )

        # 3. Model Inference (Raw Probability)
        raw_fake_prob = 0.50
        if os.path.exists(self.weights_path):
            try:
                tensor = self.preprocessor.preprocess_image(file_path)
                tensor = tensor.unsqueeze(0)  # (1, 3, H, W)
                pred_dict = self.model.predict(tensor)
                probs = pred_dict["probabilities"][0]
                raw_fake_prob = float(probs[1].item())
            except Exception as e:
                logger.error(f"Inference execution failed on {file_path}: {e}")
                raw_fake_prob = 0.50
        else:
            logger.warning("Using baseline inference as weights path does not exist.")
            raw_fake_prob = 0.50

        # 4. Three-State Safety Decision Fusion
        decision = self.decision_system.decide(
            raw_fake_prob=raw_fake_prob,
            quality_result=quality_res,
            forensic_result=forensic_res,
        )

        processing_time = round(time.time() - start_time, 4)
        return self._format_response(
            decision=decision,
            quality_res=quality_res,
            forensic_res=forensic_res,
            filename=filename,
            size=size,
            fmt=fmt,
            processing_time=processing_time,
            img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
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
    ) -> Dict[str, Any]:
        """Formats the structured forensic response."""
        is_not_suitable = decision.prediction == DecisionState.NOT_SUITABLE
        fft_data = forensic_res.get("fft_analysis", {})
        ela_data = forensic_res.get("ela_analysis", {})
        edge_data = forensic_res.get("edge_analysis", {})

        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "width": img_shape[1] if len(img_shape) > 1 else 0,
                "height": img_shape[0] if len(img_shape) > 0 else 0,
                "format": fmt,
            },
            "prediction": decision.prediction,
            "verdict": decision.verdict,
            "confidence": decision.confidence,
            "confidence_score": decision.confidence,
            "confidence_level": decision.confidence_level,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": decision.raw_model_score,
            "uncertainty": decision.uncertainty,
            "evidence": decision.evidence,
            "reason": decision.reason,
            "summary": decision.user_facing_statement,
            "user_facing_statement": decision.user_facing_statement,
            "recommendation": getattr(decision, "recommendation", None),
            "analysis_status": getattr(decision, "analysis_status", "Completed"),
            "face_detected": quality_res.face_detected,
            "media_category": getattr(decision, "media_category", "Facial Media"),
            "quality_gate": quality_res.to_dict(),
            "features": {
                "noise_inconsistency": 0.0 if is_not_suitable else forensic_res.get("composite_forensic_score", 0.15),
                "edge_boundary_artifacts": 0.0 if is_not_suitable else edge_data.get("edge_score", 0.10),
                "fft_spectral_anomaly": 0.0 if is_not_suitable else fft_data.get("fft_score", 0.10),
                "ela_compression_anomaly": 0.0 if is_not_suitable else ela_data.get("ela_score", 0.10),
                "metadata_anomaly": False,
                "blur_score": quality_res.blur_score,
            },
            "processing_time": processing_time,
        }
