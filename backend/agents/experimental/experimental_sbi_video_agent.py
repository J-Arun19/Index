import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from backend.utils.logger import setup_logger
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.models.experimental.sbi_detector import SBIDetector
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("ExperimentalSBIVideoAgent")

class ExperimentalSBIVideoAgent:
    """
    Experimental forensic video analysis agent using Self-Blended Images (SBI).
    Runs strictly in isolation from the production video pipeline.
    Performs multi-frame sampling, boundary-aware face tracking (25% margin),
    temporal consistency aggregation, and three-state safety decision fusion.
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        self.face_detector = FaceDetector()
        self.detector = SBIDetector(weights_path=weights_path, device=self.device)

    def extract_sbi_face_crop(
        self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.25
    ) -> np.ndarray:
        """
        Extracts face crop with official 25% boundary expansion (w/4, h/4)
        to capture perimeter blending artifacts.
        """
        x, y, w, h = bbox
        H, W = frame_bgr.shape[:2]
        w_margin = int(w * margin_ratio)
        h_margin = int(h * margin_ratio)

        x0 = max(0, x - w_margin)
        y0 = max(0, y - h_margin)
        x1 = min(W, x + w + w_margin)
        y1 = min(H, y + h + h_margin)
        return frame_bgr[y0:y1, x0:x1]

    def analyze(self, file_path: str, max_sample_frames: int = 8) -> Dict[str, Any]:
        """
        Analyzes a video file through multi-frame sampling, SBI boundary face inference,
        and temporal consistency safety fusion.
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "MP4"

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return self._empty_failure_response(filename, size, fmt, "Unable to open video stream.", round(time.time() - start_time, 4))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30.0
        duration = round(total_frames / fps, 2) if total_frames > 0 else 0.0

        if total_frames < 2:
            cap.release()
            return self._empty_failure_response(filename, size, fmt, "Video contains insufficient readable frames.", round(time.time() - start_time, 4))

        num_samples = min(max_sample_frames, total_frames)
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

        frames: List[np.ndarray] = []
        try:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames.append(frame)
        finally:
            cap.release()

        if not frames:
            return self._empty_failure_response(filename, size, fmt, "No readable frames could be decoded.", round(time.time() - start_time, 4))

        # Evaluate each sampled frame
        frame_fake_probs: List[float] = []
        frame_predictions: List[Dict[str, Any]] = []
        usable_faces = 0
        quality_issues: List[str] = []

        for frame_idx, frame in enumerate(frames):
            q_res = self.quality_gate.evaluate(frame, require_face=False)
            if not q_res.is_suitable:
                quality_issues.extend(q_res.issues)

            faces = self.face_detector.detect_faces(frame)
            if faces:
                usable_faces += 1
                crop = self.extract_sbi_face_crop(frame, faces[0], margin_ratio=0.25)
                face_found = True
            else:
                crop = frame
                face_found = False

            pred = self.detector.predict_crop(crop)
            fake_p = pred["fake_probability"]
            frame_fake_probs.append(fake_p)
            frame_predictions.append({
                "sample_idx": frame_idx,
                "face_detected": face_found,
                "fake_probability": fake_p,
                "predicted_label": pred["predicted_label"]
            })

        # Temporal aggregation
        mean_fake_prob = float(np.mean(frame_fake_probs))
        max_fake_prob = float(np.max(frame_fake_probs))
        min_fake_prob = float(np.min(frame_fake_probs))
        temporal_variance = float(np.std(frame_fake_probs))

        temporal_conflict = bool(
            (max_fake_prob > 0.70 and min_fake_prob < 0.25) or (temporal_variance > 0.22)
        )

        # Midpoint frame forensic analysis
        key_frame = frames[len(frames) // 2]
        forensic_res = self.forensics.evaluate(key_frame)

        # Safety Decision Fusion
        if usable_faces == 0:
            q_res_face = self.quality_gate.evaluate(frames[0], require_face=True)
            decision = self.decision_system.decide(
                raw_fake_prob=None,
                quality_result=q_res_face,
                forensic_result={},
            )
        else:
            q_res = self.quality_gate.evaluate(frames[0], require_face=False) if quality_issues else None
            decision = self.decision_system.decide(
                raw_fake_prob=mean_fake_prob,
                quality_result=q_res,
                forensic_result=forensic_res,
                temporal_variance=temporal_variance,
                temporal_conflict=temporal_conflict,
            )

        processing_time = round(time.time() - start_time, 4)

        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "fps": round(fps, 2),
                "duration_seconds": duration,
                "format": fmt,
                "total_frames": total_frames,
                "sampled_frames_count": len(frames),
                "faces_detected_in_frames": usable_faces,
                "experimental_backbone": "sbi_efficientnet_b4_video",
            },
            "prediction": decision.prediction,
            "verdict": decision.verdict,
            "confidence": decision.confidence,
            "confidence_score": decision.confidence,
            "confidence_level": decision.confidence_level,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": round(mean_fake_prob, 4),
            "uncertainty": decision.uncertainty,
            "temporal_metrics": {
                "temporal_variance": round(temporal_variance, 4),
                "temporal_conflict": temporal_conflict,
                "min_frame_fake_prob": round(min_fake_prob, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
            },
            "per_frame_predictions": frame_predictions,
            "is_hard_negative": decision.is_hard_negative,
            "reasons": [decision.reason] if decision.reason else [],
            "user_facing_statement": decision.user_facing_statement,
            "processing_time_seconds": processing_time,
            "evidence": decision.evidence,
        }

    def _empty_failure_response(self, filename: str, size: int, fmt: str, reason: str, dt: float) -> Dict[str, Any]:
        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "format": fmt,
                "total_frames": 0,
                "sampled_frames_count": 0,
                "faces_detected_in_frames": 0,
                "experimental_backbone": "sbi_efficientnet_b4_video",
            },
            "prediction": DecisionState.UNCERTAIN,
            "verdict": DecisionState.UNCERTAIN,
            "confidence": 50.0,
            "confidence_score": 50.0,
            "confidence_level": "Low",
            "fake_probability": 0.50,
            "real_probability": 0.50,
            "raw_model_score": 0.50,
            "uncertainty": 1.0,
            "temporal_metrics": {
                "temporal_variance": 0.0,
                "temporal_conflict": False,
                "min_frame_fake_prob": 0.50,
                "max_frame_fake_prob": 0.50,
            },
            "per_frame_predictions": [],
            "is_hard_negative": False,
            "reasons": [reason],
            "user_facing_statement": "Insufficient evidence to determine authenticity reliably.",
            "processing_time_seconds": dt,
            "evidence": [reason],
        }
