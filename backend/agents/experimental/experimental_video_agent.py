import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List

from backend.utils.logger import setup_logger
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.models.experimental.xception_detector import XceptionDetector
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("ExperimentalVideoAgent")

class ExperimentalVideoAgent:
    """
    Experimental forensic video analysis agent using DeepfakeBench Xception.
    Runs in isolation from the production video pipeline.
    Performs frame-level sampling, face tracking, temporal consistency aggregation,
    and three-state safety decision fusion.
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        self.face_detector = FaceDetector()
        self.detector = XceptionDetector(weights_path=weights_path, device=self.device)

    def analyze(self, file_path: str, max_sample_frames: int = 8) -> Dict[str, Any]:
        """
        Analyzes a video file through multi-frame sampling, spatial face evaluation via Xception,
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

        # Evaluate each frame with Xception
        frame_fake_probs: List[float] = []
        usable_faces = 0
        quality_issues: List[str] = []

        for frame in frames:
            q_res = self.quality_gate.evaluate(frame, require_face=False)
            if not q_res.is_suitable:
                quality_issues.extend(q_res.issues)

            faces = self.face_detector.detect_faces(frame)
            if faces:
                usable_faces += 1
                crop = self.face_detector.crop_face(frame, faces[0])
            else:
                crop = frame

            pred = self.detector.predict_frame(crop)
            frame_fake_probs.append(pred["fake_probability"])

        # Temporal aggregation
        mean_fake_prob = float(np.mean(frame_fake_probs))
        max_fake_prob = float(np.max(frame_fake_probs))
        min_fake_prob = float(np.min(frame_fake_probs))
        temporal_variance = float(np.std(frame_fake_probs))

        # Temporal conflict detection
        temporal_conflict = bool(
            (max_fake_prob > 0.70 and min_fake_prob < 0.25) or (temporal_variance > 0.22)
        )

        # Forensics on midpoint frame
        key_frame = frames[len(frames) // 2]
        forensic_res = self.forensics.evaluate(key_frame)

        # Three-state Safety Decision
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
                "experimental_backbone": "deepfakebench_xception",
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
            "evidence": decision.evidence,
            "reason": decision.reason,
            "summary": decision.user_facing_statement,
            "user_facing_statement": decision.user_facing_statement,
            "temporal_metrics": {
                "mean_frame_fake_prob": round(mean_fake_prob, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "min_frame_fake_prob": round(min_fake_prob, 4),
                "temporal_variance": round(temporal_variance, 4),
                "temporal_conflict": temporal_conflict,
                "frame_scores": [round(p, 4) for p in frame_fake_probs]
            },
            "forensics": forensic_res,
            "processing_time": processing_time,
        }

    def _empty_failure_response(self, filename: str, size: int, fmt: str, reason: str, dt: float) -> Dict[str, Any]:
        return {
            "metadata": {"filename": filename, "file_size_bytes": size, "format": fmt},
            "prediction": DecisionState.UNCERTAIN,
            "verdict": DecisionState.UNCERTAIN,
            "confidence": 0.0,
            "fake_probability": 0.50,
            "reason": reason,
            "processing_time": dt,
        }
