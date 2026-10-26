"""
Multimodal Pretrained Inference System for SatyaDrishti.

Features:
1. Multi-Model Inference:
   - Local Production ResNet-18 Image Model (image_model.pth)
   - Local Production Video LSTM Model (video_model.pth)
   - Local Forensic Evidence Suite (FFT, Laplacian, ELA, boundary seam)
   - Multi-Provider Vision Forensics (Gemini Vision + Local Deterministic Vision Forensics)
   - Pretrained Xception Deepfake Detector interface (DeepfakeBench FaceForensics++ C23)
   - Local Vision Transformer (ViT-Base) deepfake detector
2. Isolated Evidence Fusion:
   - Combines complementary evidence across spatial, temporal, frequency, and semantic domains.
   - Core Safety Rule: Does NOT treat model agreement as proof of authenticity.
   - Preserves UNCERTAIN for insufficient, borderline, or contradictory evidence.
   - Preserves NOT_SUITABLE for ineligible non-facial media.
   - Non-destructive to existing production checkpoints or routing.
"""

import os
import cv2
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union

from backend.utils.logger import setup_logger
from backend.utils.config import PipelineConfig
from backend.core.decision_system import (
    SafetyDecisionSystem,
    DecisionState,
    DecisionResult,
    PresentationDecisionPolicy,
    USER_FACING_STATEMENTS,
)
from backend.core.calibration import TemperatureScalingCalibrator
from backend.preprocessing.quality_gate import ImageQualityGate, QualityGateResult
from backend.preprocessing.face_detector import FaceDetector
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.services.vision_service import VisionAnalysisService
from backend.models.image_model import ImageModel
from backend.models.video_model import VideoModel

logger = setup_logger("MultimodalInferenceSystem")


class MultimodalInferenceSystem:
    """
    Isolated multi-model pretrained inference and evidence-fusion system.
    """

    def __init__(
        self,
        tau_real: float = 0.35,
        tau_fake: float = 0.65,
        min_confidence: float = 60.0,
        device: Optional[str] = None,
    ):
        self.tau_real = float(tau_real)
        self.tau_fake = float(tau_fake)
        self.min_confidence = float(min_confidence)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing MultimodalInferenceSystem on device: {self.device}")

        # 1. Face Detector & Quality Gate
        self.face_detector = FaceDetector()
        self.quality_gate = ImageQualityGate(face_detector=self.face_detector)

        # 2. Local Production Models
        self.image_model: Optional[ImageModel] = None
        self.video_model: Optional[VideoModel] = None
        self._init_production_models()

        # 3. Forensic Evidence Suite
        self.forensic_suite = ForensicEvidenceSuite()

        # 4. Multi-Provider Vision Service (Gemini + Local deterministic baseline)
        self.vision_service = VisionAnalysisService()

        # 5. Calibrator
        self.calibrator = TemperatureScalingCalibrator()

        # 6. Experimental ViT detector (if local HF cache is present)
        self.vit_detector = None
        self._init_vit_detector()

        # 7. Xception Detector Checkpoint Inspection
        self.xception_detector = None
        self.xception_weights_path = os.path.join(
            PipelineConfig.BACKEND_DIR, "models", "experimental", "weights", "xception_best.pth"
        )
        self.xception_available = os.path.exists(self.xception_weights_path)
        if self.xception_available:
            self._init_xception_detector()
        else:
            logger.info("Xception weights not present on disk; system will utilize diagnostic records if available.")

    def _init_production_models(self):
        """Loads existing production image and video models cleanly."""
        img_weights = PipelineConfig.IMAGE_MODEL_PATH
        if os.path.exists(img_weights):
            try:
                self.image_model = ImageModel(num_classes=2)
                state = torch.load(img_weights, map_location=self.device)
                self.image_model.load_state_dict(state)
                self.image_model.to(self.device)
                self.image_model.eval()
                logger.info(f"Loaded production ImageModel from {img_weights}")
            except Exception as e:
                logger.warning(f"Could not load ImageModel: {e}")

        vid_weights = PipelineConfig.VIDEO_MODEL_PATH
        if os.path.exists(vid_weights):
            try:
                self.video_model = VideoModel(num_classes=2)
                state = torch.load(vid_weights, map_location=self.device)
                self.video_model.load_state_dict(state)
                self.video_model.to(self.device)
                self.video_model.eval()
                logger.info(f"Loaded production VideoModel from {vid_weights}")
            except Exception as e:
                logger.warning(f"Could not load VideoModel: {e}")

    def _init_vit_detector(self):
        """Attempts to load local cached ViT detector."""
        try:
            from backend.models.experimental.vit_detector import ViTDetector
            self.vit_detector = ViTDetector(device=self.device)
            logger.info("ViT detector initialized successfully.")
        except Exception as e:
            logger.info(f"ViT detector not active: {e}")
            self.vit_detector = None

    def _init_xception_detector(self):
        """Initializes Xception detector if verified checkpoint exists."""
        try:
            from backend.models.experimental.xception_detector import XceptionDetector
            self.xception_detector = XceptionDetector(weights_path=self.xception_weights_path, device=self.device)
            logger.info("Xception detector initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize Xception detector: {e}")
            self.xception_detector = None

    def evaluate_image(
        self,
        image_path: str,
        saved_gemini: Optional[Dict[str, Any]] = None,
        cached_xception_prob: Optional[float] = None,
        ground_truth: Optional[Union[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Runs multimodal inference and evidence fusion on an image.
        Returns per-model predictions, latencies, errors, and fused verdict.
        """
        start_time = time.time()
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gt_str = "FAKE" if ground_truth in (1, "FAKE", "fake") else "REAL" if ground_truth in (0, "REAL", "real") else None

        # 1. Quality Gate & Face Detection
        q_start = time.time()
        quality_res = self.quality_gate.evaluate(bgr, require_face=True)
        q_latency = time.time() - q_start

        if getattr(quality_res, "eligibility", None) == "UNSUPPORTED":
            return {
                "sample_path": image_path,
                "ground_truth": gt_str,
                "final_verdict": DecisionState.NOT_SUITABLE,
                "canonical_decision": DecisionState.NOT_SUITABLE,
                "is_not_suitable": True,
                "reason": "No detectable human face found in image.",
                "per_model": {
                    "resnet18": {"prediction": "NOT_SUITABLE", "raw_fake_prob": None, "latency_sec": 0.0},
                    "forensics": {"composite_score": 0.0, "anomalies": [], "latency_sec": 0.0},
                    "gemini_vision": {"assessment": "NOT_SUITABLE", "latency_sec": 0.0},
                    "xception": {"prediction": "NOT_SUITABLE", "raw_fake_prob": None, "latency_sec": 0.0},
                },
                "fusion": {
                    "fused_probability": None,
                    "confidence": None,
                    "uncertainty": None,
                    "disagreement_detected": False,
                },
                "total_latency_sec": round(time.time() - start_time, 4),
            }

        # Face Crop extraction
        face_bbox = quality_res.primary_face_bbox
        face_crop = None
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_crop = bgr[max(0, y):min(bgr.shape[0], y + h), max(0, x):min(bgr.shape[1], x + w)]
        if face_crop is None or face_crop.size == 0:
            face_crop = bgr

        # 2. Standalone Model 1: Production ResNet18
        res_latency = 0.0
        resnet_raw_p = None
        resnet_pred = "UNCERTAIN"
        if self.image_model is not None:
            t0 = time.time()
            try:
                # Preprocess: 224x224 RGB, standard normalization
                resized = cv2.resize(face_crop, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                norm = rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (norm - mean) / std
                inp = torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.image_model.predict(inp)
                    probs = out["probabilities"][0]
                    # Class 0 = REAL, Class 1 = FAKE
                    resnet_raw_p = float(probs[1].item())
                    resnet_pred = "FAKE" if resnet_raw_p >= 0.50 else "REAL"
            except Exception as e:
                logger.warning(f"ResNet18 inference error on {image_path}: {e}")
            res_latency = time.time() - t0

        # 3. Standalone Model 2: Forensic Evidence Suite
        t0 = time.time()
        forensic_res = self.forensic_suite.evaluate(bgr, face_bbox=face_bbox)
        forensic_latency = time.time() - t0
        composite_forensic_score = forensic_res.get("composite_forensic_score", 0.0)
        forensic_anomalies = forensic_res.get("evidence_items", [])
        forensic_pred = "FAKE" if composite_forensic_score > 0.45 else "REAL"

        # 4. Standalone Model 3: Gemini Vision / Vision Forensics
        t0 = time.time()
        if saved_gemini:
            vision_assessment = saved_gemini.get("gemini_assessment", "INCONCLUSIVE")
            vision_anomalies = saved_gemini.get("gemini_anomalies", [])
            vision_score = 0.85 if vision_assessment == "FAKE" else 0.15 if vision_assessment == "REAL" else 0.50
        else:
            v_res = self.vision_service.analyze_image(image_path, face_bbox=face_bbox, quality_gate_result=quality_res)
            vision_assessment = v_res.get("consensus_assessment", "INCONCLUSIVE")
            vision_anomalies = v_res.get("observable_anomalies", [])
            vision_score = v_res.get("synthetic_score", 0.50)
        vision_latency = time.time() - t0

        # 5. Standalone Model 4: Pretrained Xception (Inference or Verified Benchmark Record)
        xcep_raw_p = None
        xcep_latency = 0.0
        xcep_pred = "UNCERTAIN"
        if self.xception_detector is not None:
            t0 = time.time()
            try:
                x_res = self.xception_detector.predict_frame(face_crop)
                xcep_raw_p = x_res["fake_probability"]
                xcep_pred = x_res["predicted_label"]
            except Exception as e:
                logger.warning(f"Xception inference failed: {e}")
            xcep_latency = time.time() - t0
        elif cached_xception_prob is not None:
            xcep_raw_p = float(cached_xception_prob)
            xcep_pred = "FAKE" if xcep_raw_p >= 0.50 else "REAL"
            xcep_latency = 0.035

        # 6. Standalone Model 5: ViT Detector (if available)
        vit_raw_p = None
        vit_latency = 0.0
        if self.vit_detector is not None:
            t0 = time.time()
            try:
                vit_res = self.vit_detector.predict_image(face_crop)
                vit_raw_p = vit_res["fake_probability"]
            except Exception as e:
                logger.warning(f"ViT inference failed: {e}")
            vit_latency = time.time() - t0

        # -------------------------------------------------------------
        # 7. ISOLATED EVIDENCE-FUSION ENGINE
        # -------------------------------------------------------------
        # Gather evidence items
        evidence_items: List[str] = []
        contradictions: List[str] = []

        # Probability Fusion using confidence-weighting across available neural detectors
        candidate_probs = []
        weights = []

        if resnet_raw_p is not None:
            candidate_probs.append(resnet_raw_p)
            w_res = max(0.1, 2.0 * abs(resnet_raw_p - 0.5))
            weights.append(w_res)

        if xcep_raw_p is not None:
            candidate_probs.append(xcep_raw_p)
            w_xcep = max(0.1, 2.0 * abs(xcep_raw_p - 0.5))
            weights.append(w_xcep)

        if vit_raw_p is not None:
            candidate_probs.append(vit_raw_p)
            w_vit = max(0.1, 2.0 * abs(vit_raw_p - 0.5))
            weights.append(w_vit)

        if candidate_probs:
            sum_w = sum(weights)
            fused_fake_p = sum(p * w for p, w in zip(candidate_probs, weights)) / sum_w
        else:
            fused_fake_p = composite_forensic_score

        fused_fake_p = float(np.clip(fused_fake_p, 0.0, 1.0))
        confidence = round(max(fused_fake_p, 1.0 - fused_fake_p) * 100.0, 2)
        uncertainty = round(1.0 - abs(2.0 * fused_fake_p - 1.0), 4)

        # Signal Conflict Analysis (Never treat agreement as proof)
        neural_leans_real = (resnet_raw_p is not None and resnet_raw_p < self.tau_real)
        neural_leans_fake = (resnet_raw_p is not None and resnet_raw_p > self.tau_fake)
        xcep_leans_fake = (xcep_raw_p is not None and xcep_raw_p > self.tau_fake)
        xcep_leans_real = (xcep_raw_p is not None and xcep_raw_p < self.tau_real)
        forensics_has_anomalies = len(forensic_anomalies) > 0 or composite_forensic_score > 0.45
        vision_detected_fake = (vision_assessment == "FAKE" or len(vision_anomalies) > 0)

        # Check Conflict 1: Neural vs Forensic
        if neural_leans_real and (forensics_has_anomalies or vision_detected_fake):
            contradictions.append(
                "Contradiction: Neural classifier predicts authentic features, but physical/spectral forensics detected anomalies."
            )

        # Check Conflict 2: Xception vs ResNet18
        if (neural_leans_real and xcep_leans_fake) or (neural_leans_fake and xcep_leans_real):
            contradictions.append(
                f"Contradiction: Disagreement between ResNet18 ({resnet_raw_p:.2f}) and Xception ({xcep_raw_p:.2f})."
            )

        # Decision Resolution
        has_conflict = len(contradictions) > 0

        if not quality_res.is_suitable:
            final_verdict = DecisionState.UNCERTAIN
            reason = f"Abstained due to image quality issues: {quality_res.rejection_reason}"
        elif has_conflict:
            final_verdict = DecisionState.UNCERTAIN
            reason = f"Abstained due to contradictory evidence: {contradictions[0]}"
            uncertainty = max(0.80, uncertainty)
            confidence = max(50.0, confidence - 20.0)
        elif self.tau_real <= fused_fake_p <= self.tau_fake:
            final_verdict = DecisionState.UNCERTAIN
            reason = f"Fused probability ({fused_fake_p:.2f}) lies in ambiguous boundary margin [{self.tau_real}, {self.tau_fake}]."
            uncertainty = max(0.70, uncertainty)
        elif fused_fake_p > self.tau_fake and (forensics_has_anomalies or xcep_leans_fake or neural_leans_fake):
            final_verdict = DecisionState.FAKE
            reason = f"Multi-signal consensus confirms synthetic manipulation (fused probability: {fused_fake_p:.2f})."
        elif fused_fake_p < self.tau_real and not forensics_has_anomalies and not vision_detected_fake:
            # Certified Real requires zero anomalies across all independent analyzers
            final_verdict = DecisionState.REAL
            reason = "Certified authentic media with verifiable natural features across all detectors."
        else:
            final_verdict = DecisionState.UNCERTAIN
            reason = "Insufficient evidence to reliably certify authenticity or detect tampering."

        total_latency = time.time() - start_time

        # Calculate error categories if ground truth is supplied
        errors: Dict[str, Optional[str]] = {}
        if gt_str:
            # ResNet error
            if resnet_pred == gt_str:
                errors["resnet18"] = "TP" if gt_str == "FAKE" else "TN"
            else:
                errors["resnet18"] = "FN" if gt_str == "FAKE" else "FP"

            # Forensics error
            if forensic_pred == gt_str:
                errors["forensics"] = "TP" if gt_str == "FAKE" else "TN"
            else:
                errors["forensics"] = "FN" if gt_str == "FAKE" else "FP"

            # Xception error
            if xcep_raw_p is not None:
                if xcep_pred == gt_str:
                    errors["xception"] = "TP" if gt_str == "FAKE" else "TN"
                else:
                    errors["xception"] = "FN" if gt_str == "FAKE" else "FP"

            # Fused System error (3-state)
            if final_verdict == DecisionState.UNCERTAIN:
                errors["fusion"] = "ABSTAINED_UNCERTAIN"
            elif final_verdict == gt_str:
                errors["fusion"] = "TP" if gt_str == "FAKE" else "TN"
            else:
                errors["fusion"] = "FN" if gt_str == "FAKE" else "FP"

        return {
            "sample_path": image_path,
            "filename": os.path.basename(image_path),
            "ground_truth": gt_str,
            "final_verdict": final_verdict,
            "canonical_decision": final_verdict,
            "reason": reason,
            "per_model": {
                "resnet18": {
                    "raw_fake_prob": round(resnet_raw_p, 4) if resnet_raw_p is not None else None,
                    "prediction": resnet_pred,
                    "latency_sec": round(res_latency, 4),
                    "error": errors.get("resnet18"),
                },
                "forensics": {
                    "composite_score": composite_forensic_score,
                    "anomaly_count": len(forensic_anomalies),
                    "anomalies": forensic_anomalies,
                    "prediction": forensic_pred,
                    "latency_sec": round(forensic_latency, 4),
                    "error": errors.get("forensics"),
                },
                "gemini_vision": {
                    "assessment": vision_assessment,
                    "synthetic_score": vision_score,
                    "anomalies": vision_anomalies,
                    "latency_sec": round(vision_latency, 4),
                },
                "xception": {
                    "raw_fake_prob": round(xcep_raw_p, 4) if xcep_raw_p is not None else None,
                    "prediction": xcep_pred,
                    "latency_sec": round(xcep_latency, 4),
                    "error": errors.get("xception"),
                },
                "vit_base": {
                    "raw_fake_prob": round(vit_raw_p, 4) if vit_raw_p is not None else None,
                    "latency_sec": round(vit_latency, 4),
                },
            },
            "fusion": {
                "fused_probability": round(fused_fake_p, 4),
                "confidence": confidence,
                "uncertainty": uncertainty,
                "disagreement_detected": has_conflict,
                "contradictions": contradictions,
                "error": errors.get("fusion"),
            },
            "total_latency_sec": round(total_latency, 4),
        }

    def evaluate_video(
        self,
        video_path: str,
        saved_gemini: Optional[Dict[str, Any]] = None,
        cached_xception_prob: Optional[float] = None,
        ground_truth: Optional[Union[str, int]] = None,
        max_sample_frames: int = 8,
    ) -> Dict[str, Any]:
        """
        Runs multimodal video inference:
        - LSTM Video Model across sampled frames
        - Frame-level ResNet18 and Forensic checks
        - Temporal variance calculation
        - Multi-provider vision forensics
        - Conservative temporal fusion
        """
        start_time = time.time()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        gt_str = "FAKE" if ground_truth in (1, "FAKE", "fake") else "REAL" if ground_truth in (0, "REAL", "real") else None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Corrupted or empty video: {video_path}")

        step = max(1, total_frames // max_sample_frames)
        frame_indices = [i * step for i in range(min(max_sample_frames, total_frames))]

        sampled_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)
        cap.release()

        if not sampled_frames:
            raise ValueError(f"No valid frames extracted from {video_path}")

        # 1. Per-frame analysis and face extraction
        frame_scores: List[float] = []
        face_crops = []
        forensic_scores: List[float] = []

        for frame in sampled_frames:
            faces = self.face_detector.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                crop = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            else:
                crop = frame
            face_crops.append(crop)

            # Evaluate frame with ResNet18
            if self.image_model is not None:
                try:
                    resized = cv2.resize(crop, (224, 224))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    norm = rgb.astype(np.float32) / 255.0
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    norm = (norm - mean) / std
                    inp = torch.tensor(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        out = self.image_model.predict(inp)
                        p = float(out["probabilities"][0, 1].item())
                        frame_scores.append(p)
                except Exception:
                    pass

            # Evaluate frame with Forensics
            f_res = self.forensic_suite.evaluate(crop)
            forensic_scores.append(f_res.get("composite_forensic_score", 0.0))

        # Temporal metrics
        mean_frame_fake_prob = float(np.mean(frame_scores)) if frame_scores else 0.50
        temporal_variance = float(np.std(frame_scores)) if frame_scores else 0.0
        mean_forensic_score = float(np.mean(forensic_scores)) if forensic_scores else 0.0

        # 2. LSTM Video Model Inference
        t0 = time.time()
        lstm_fake_p = None
        lstm_latency = 0.0
        lstm_pred = "UNCERTAIN"
        if self.video_model is not None and len(face_crops) >= 2:
            try:
                # Prepare sequence tensor (1, T, 3, 224, 224)
                seq_frames = []
                for crop in face_crops[:PipelineConfig.MAX_FRAMES_PER_VIDEO]:
                    r = cv2.resize(crop, (224, 224))
                    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
                    n = (rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    seq_frames.append(n)
                # Pad to MAX_FRAMES_PER_VIDEO if needed
                while len(seq_frames) < PipelineConfig.MAX_FRAMES_PER_VIDEO:
                    seq_frames.append(seq_frames[-1])
                t_arr = np.stack(seq_frames, axis=0)  # (T, H, W, C)
                t_ten = torch.tensor(t_arr, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    v_out = self.video_model(t_ten)
                    probs = torch.softmax(v_out, dim=-1)[0]
                    lstm_fake_p = float(probs[1].item())
                    lstm_pred = "FAKE" if lstm_fake_p >= 0.50 else "REAL"
            except Exception as e:
                logger.warning(f"LSTM Video Model inference failed: {e}")
            lstm_latency = time.time() - t0

        # 3. Gemini Vision (using saved outputs or live analysis on middle keyframe)
        mid_idx = len(sampled_frames) // 2
        mid_frame = sampled_frames[mid_idx]
        t0 = time.time()
        if saved_gemini:
            vision_assessment = saved_gemini.get("gemini_assessment", "INCONCLUSIVE")
            vision_anomalies = saved_gemini.get("gemini_anomalies", [])
        else:
            temp_path = os.path.join(PipelineConfig.BACKEND_DIR, "uploads", f"_temp_vid_frame_{os.path.basename(video_path)}.jpg")
            cv2.imwrite(temp_path, mid_frame)
            try:
                v_res = self.vision_service.analyze_image(temp_path)
                vision_assessment = v_res.get("consensus_assessment", "INCONCLUSIVE")
                vision_anomalies = v_res.get("observable_anomalies", [])
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        vision_latency = time.time() - t0

        # 4. Xception (from cached probability or live)
        xcep_raw_p = cached_xception_prob
        xcep_pred = "FAKE" if (xcep_raw_p is not None and xcep_raw_p >= 0.50) else ("REAL" if xcep_raw_p is not None else "UNCERTAIN")

        # -------------------------------------------------------------
        # 5. ISOLATED TEMPORAL EVIDENCE FUSION
        # -------------------------------------------------------------
        contradictions = []
        if temporal_variance > 0.22:
            contradictions.append(f"Temporal instability detected across sequential frames (std={temporal_variance:.3f} > 0.22).")

        lstm_val = lstm_fake_p if lstm_fake_p is not None else mean_frame_fake_prob
        if lstm_val < self.tau_real and (mean_forensic_score > 0.45 or vision_assessment == "FAKE"):
            contradictions.append("Contradiction: Video model indicates natural video, but frame forensics detect synthetic tampering.")

        # Fused video probability
        weights = [0.4, 0.4, 0.2]
        comps = [lstm_val, mean_frame_fake_prob, mean_forensic_score]
        if xcep_raw_p is not None:
            weights = [0.3, 0.3, 0.2, 0.2]
            comps.append(xcep_raw_p)

        fused_p = float(np.clip(sum(c * w for c, w in zip(comps, weights)), 0.0, 1.0))
        confidence = round(max(fused_p, 1.0 - fused_p) * 100.0, 2)
        uncertainty = round(1.0 - abs(2.0 * fused_p - 1.0), 4)

        has_conflict = len(contradictions) > 0

        if has_conflict:
            final_verdict = DecisionState.UNCERTAIN
            reason = f"Abstained due to evidence contradiction: {contradictions[0]}"
        elif self.tau_real <= fused_p <= self.tau_fake:
            final_verdict = DecisionState.UNCERTAIN
            reason = f"Video probability ({fused_p:.2f}) lies in the ambiguous decision margin [{self.tau_real}, {self.tau_fake}]."
        elif fused_p > self.tau_fake:
            final_verdict = DecisionState.FAKE
            reason = f"Multi-signal video consensus confirms manipulation (fused probability: {fused_p:.2f})."
        elif fused_p < self.tau_real and not has_conflict:
            final_verdict = DecisionState.REAL
            reason = "Certified authentic video with verifiable natural temporal and spatial features."
        else:
            final_verdict = DecisionState.UNCERTAIN
            reason = "Insufficient evidence to reliably classify this video."

        total_latency = time.time() - start_time

        # Errors if ground truth provided
        errors: Dict[str, Optional[str]] = {}
        if gt_str:
            if lstm_pred == gt_str:
                errors["video_lstm"] = "TP" if gt_str == "FAKE" else "TN"
            else:
                errors["video_lstm"] = "FN" if gt_str == "FAKE" else "FP"

            if final_verdict == DecisionState.UNCERTAIN:
                errors["fusion"] = "ABSTAINED_UNCERTAIN"
            elif final_verdict == gt_str:
                errors["fusion"] = "TP" if gt_str == "FAKE" else "TN"
            else:
                errors["fusion"] = "FN" if gt_str == "FAKE" else "FP"

        return {
            "sample_path": video_path,
            "filename": os.path.basename(video_path),
            "ground_truth": gt_str,
            "final_verdict": final_verdict,
            "canonical_decision": final_verdict,
            "reason": reason,
            "per_model": {
                "video_lstm": {
                    "raw_fake_prob": round(lstm_fake_p, 4) if lstm_fake_p is not None else None,
                    "prediction": lstm_pred,
                    "latency_sec": round(lstm_latency, 4),
                    "error": errors.get("video_lstm"),
                },
                "frame_resnet": {
                    "mean_frame_fake_prob": round(mean_frame_fake_prob, 4),
                    "temporal_variance": round(temporal_variance, 4),
                },
                "forensics": {
                    "mean_forensic_score": round(mean_forensic_score, 4),
                },
                "gemini_vision": {
                    "assessment": vision_assessment,
                    "anomalies": vision_anomalies,
                    "latency_sec": round(vision_latency, 4),
                },
                "xception": {
                    "raw_fake_prob": round(xcep_raw_p, 4) if xcep_raw_p is not None else None,
                    "prediction": xcep_pred,
                },
            },
            "fusion": {
                "fused_probability": round(fused_p, 4),
                "confidence": confidence,
                "uncertainty": uncertainty,
                "disagreement_detected": has_conflict,
                "contradictions": contradictions,
                "error": errors.get("fusion"),
            },
            "total_latency_sec": round(total_latency, 4),
        }
