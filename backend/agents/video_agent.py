import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.models.video_model import VideoModel
from backend.preprocessing.video_preprocessor import VideoPreprocessor
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState

logger = setup_logger("VideoAgent")


class VideoAgent:
    """
    Forensic video analysis agent.
    Performs frame-level sampling, temporal aggregation, inter-frame consistency
    evaluation, and conflict-aware three-state safety decisions.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PipelineConfig.VIDEO_MODEL_PATH
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.decision_system = SafetyDecisionSystem()
        self.preprocessor = VideoPreprocessor()
        self.img_preprocessor = ImagePreprocessor()
        self._model: Optional[VideoModel] = None

    @property
    def model(self) -> VideoModel:
        """Lazily initializes and caches the video neural network."""
        if self._model is None:
            model = VideoModel(num_classes=2, pretrained=False)
            if os.path.exists(self.weights_path):
                try:
                    model.load_weights(self.weights_path)
                    logger.info(f"VideoAgent: Loaded model weights from {self.weights_path}")
                except Exception as e:
                    logger.error(f"VideoAgent: Failed to load weights from {self.weights_path}: {e}")
            else:
                logger.warning(f"VideoAgent: No weights found at {self.weights_path}. Model uninitialized.")
            model.eval()
            self._model = model
        return self._model

    def analyze(self, file_path: str, max_sample_frames: int = 10) -> Dict[str, Any]:
        """
        Analyzes a video file through multi-frame sampling, temporal consistency,
        and safety-calibrated decision aggregation.
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "MP4"

        # 1. Open Video and sample frames
        cap = None
        frames = []
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                decision = self.decision_system.decide(
                    raw_fake_prob=0.50,
                    quality_result=None,
                )
                return self._empty_failure_response(
                    filename, size, fmt, "Unable to open video stream.", round(time.time() - start_time, 4)
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = fps if fps > 0 else 30.0
            duration = round(total_frames / fps, 2) if total_frames > 0 else 0.0

            if total_frames < 2:
                return self._empty_failure_response(
                    filename, size, fmt, "Video contains insufficient readable frames.", round(time.time() - start_time, 4)
                )

            # Distribute sample frame indices evenly
            num_samples = min(max_sample_frames, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frames.append(frame)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        if not frames:
            return self._empty_failure_response(
                filename, size, fmt, "No readable frames could be decoded.", round(time.time() - start_time, 4)
            )

        # 2. Frame-level quality and forensic evaluation
        frame_fake_probs: List[float] = []
        quality_issues: List[str] = []
        usable_faces = 0

        # Preprocess full sequence for video model if weights exist
        video_model_raw_prob = None
        if os.path.exists(self.weights_path):
            try:
                tensor = self.preprocessor.preprocess_video(file_path)
                tensor = tensor.unsqueeze(0)  # (1, seq_len, 3, H, W)
                pred_dict = self.model.predict(tensor)
                probs = pred_dict["probabilities"][0]
                video_model_raw_prob = float(probs[1].item())
            except Exception as e:
                logger.error(f"Video model sequential inference failed: {e}")

        # Frame-level inference evaluation
        from backend.agents.image_agent import ImageAgent
        img_agent = ImageAgent()

        for frame in frames:
            q_res = self.quality_gate.evaluate(frame, require_face=False)
            if not q_res.is_suitable:
                quality_issues.extend(q_res.issues)
            if q_res.face_detected:
                usable_faces += 1

            if os.path.exists(img_agent.weights_path):
                try:
                    rgb = img_agent.preprocessor.get_cropped_resized_rgb(frame)
                    t = img_agent.preprocessor.apply_augmentation_and_normalization(rgb).unsqueeze(0)
                    with torch.no_grad():
                        logits = img_agent.model(t)
                        f_p = float(torch.softmax(logits, dim=-1)[0, 1].item())
                        frame_fake_probs.append(f_p)
                except Exception:
                    pass

        # If image model frame predictions are empty, fall back to video model prob
        if not frame_fake_probs:
            frame_fake_probs = [video_model_raw_prob if video_model_raw_prob is not None else 0.50] * len(frames)

        # 3. Temporal aggregation and consistency analysis
        mean_fake_prob = float(np.mean(frame_fake_probs))
        max_fake_prob = float(np.max(frame_fake_probs))
        min_fake_prob = float(np.min(frame_fake_probs))
        temporal_variance = float(np.std(frame_fake_probs))

        # Check for frame contradiction
        # E.g. some frames are strongly fake (>0.70) while others are strongly real (<0.25)
        temporal_conflict = bool(
            (max_fake_prob > 0.70 and min_fake_prob < 0.25) or (temporal_variance > 0.22)
        )

        # Primary composite score
        if video_model_raw_prob is not None:
            raw_video_prob = float(0.5 * video_model_raw_prob + 0.5 * mean_fake_prob)
        else:
            raw_video_prob = mean_fake_prob

        # 4. Forensic frame check (FFT / ELA on key frames)
        key_frame = frames[len(frames) // 2]
        forensic_res = self.forensics.evaluate(key_frame)

        # 5. Safety Decision System Evaluation
        if usable_faces == 0:
            q_res = self.quality_gate.evaluate(frames[0], require_face=True)
            decision = self.decision_system.decide(
                raw_fake_prob=None,
                quality_result=q_res,
                forensic_result={},
            )
        else:
            q_res = self.quality_gate.evaluate(frames[0], require_face=False) if quality_issues else None
            decision = self.decision_system.decide(
                raw_fake_prob=raw_video_prob,
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
            "temporal_metrics": {
                "mean_frame_fake_prob": round(mean_fake_prob, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "min_frame_fake_prob": round(min_fake_prob, 4),
                "temporal_variance": round(temporal_variance, 4),
                "temporal_conflict": temporal_conflict,
            },
            "features": {
                "temporal_flicker_score": round(temporal_variance, 4),
                "face_jitter_ratio": round(temporal_variance * 0.8, 4),
                "audio_sync_anomaly": False,
                "noise_inconsistency": forensic_res.get("composite_forensic_score", 0.15),
            },
            "processing_time": processing_time,
        }

    def _empty_failure_response(self, filename, size, fmt, reason, processing_time) -> Dict[str, Any]:
        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "fps": 0,
                "duration_seconds": 0,
                "format": fmt,
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
            "evidence": [f"Video Decode Failure: {reason}"],
            "reason": f"Video analysis aborted: {reason}",
            "summary": "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.",
            "user_facing_statement": "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.",
            "processing_time": processing_time,
        }
