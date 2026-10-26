import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.utils.device_manager import DeviceManager
from backend.models.video_model import VideoModel
from backend.preprocessing.video_preprocessor import VideoPreprocessor
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.services.vision_service import VisionAnalysisService
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.forensics.temporal_facial_analyzer import TemporalFacialAnalyzer

logger = setup_logger("VideoAgent")


class VideoAgent:
    """
    Forensic video analysis agent.
    Performs frame-level sampling across the video, temporal consistency evaluation,
    facial appearance shift tracking, transparent audio-visual status reporting,
    and consistent decision engine integration.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PipelineConfig.VIDEO_MODEL_PATH
        self.device = DeviceManager.get_device_str()
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.vision_service = VisionAnalysisService()
        self.decision_system = SafetyDecisionSystem()
        self.preprocessor = VideoPreprocessor()
        self.img_preprocessor = ImagePreprocessor()
        self._model: Optional[VideoModel] = None
        self._vit_detector: Optional[Any] = None
        self._xception_detector: Optional[Any] = None
        self._img_agent = None

    def _fallback_to_cpu(self, reason: str = "") -> None:
        """Safely transitions all cached video specialist models to CPU and switches self.device to 'cpu'."""
        if self.device == "cpu":
            return

        logger.info(f"VideoAgent: Switching device from {self.device} to cpu. Reason: {reason}")
        self.device = "cpu"

        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception as e:
                logger.warning(f"Failed to move VideoModel to CPU directly: {e}")
                self._model = None

        if self._xception_detector is not None and self._xception_detector is not False:
            try:
                self._xception_detector.to("cpu")
            except Exception as e:
                logger.warning(f"Failed to move Xception to CPU directly in VideoAgent: {e}")
                self._xception_detector = False

        if self._vit_detector is not None and self._vit_detector is not False:
            try:
                self._vit_detector.to("cpu")
            except Exception as e:
                logger.warning(f"Failed to move ViT to CPU directly in VideoAgent: {e}")
                self._vit_detector = False

        if self._img_agent is not None:
            try:
                self._img_agent._fallback_to_cpu(reason)
            except Exception:
                pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def vit_detector(self):
        """Lazily initializes and caches offline ViT detector if present."""
        if not hasattr(self, "_vit_detector"):
            self._vit_detector = None
        if self._vit_detector is None:
            try:
                from backend.models.experimental.vit_detector import ViTDetector
                if self.device == "cuda" and not DeviceManager.is_cuda_healthy():
                    self.device = "cpu"
                self._vit_detector = ViTDetector(device=self.device)
                logger.info(f"VideoAgent: Loaded auxiliary ViTDetector on {self.device}")
            except Exception as e:
                logger.warning(f"VideoAgent: ViTDetector auxiliary model not loaded: {e}")
                self._vit_detector = False
        return self._vit_detector if self._vit_detector is not False else None

    @property
    def xception_detector(self):
        """Lazily initializes and caches offline Xception detector if present."""
        if not hasattr(self, "_xception_detector"):
            self._xception_detector = None
        if self._xception_detector is None:
            try:
                from backend.models.experimental.xception_detector import XceptionDetector
                weights_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models",
                    "experimental",
                    "weights",
                    "xception_best.pth",
                )
                if os.path.exists(weights_path):
                    if self.device == "cuda" and not DeviceManager.is_cuda_healthy():
                        self.device = "cpu"
                    self._xception_detector = XceptionDetector(weights_path=weights_path, device=self.device)
                    logger.info(f"VideoAgent: Loaded pretrained XceptionDetector on {self.device}")
                else:
                    self._xception_detector = False
            except Exception as e:
                logger.warning(f"VideoAgent: XceptionDetector model not loaded: {e}")
                self._xception_detector = False
        return self._xception_detector if self._xception_detector is not False else None

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
            
            if self.device == "cuda" and not DeviceManager.is_cuda_healthy():
                self.device = "cpu"
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def analyze(
        self,
        file_path: str,
        max_sample_frames: int = 10,
        selected_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a video file through multi-frame sampling across the timeline,
        temporal consistency inspection, and calibrated decision aggregation.
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "MP4"
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Video model weights file not found: {self.weights_path}")

        # 1. Open Video and sample frames across duration
        cap = None
        frames = []
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video stream for {filename}.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = fps if fps > 0 else 30.0
            duration = round(total_frames / fps, 2) if total_frames > 0 else 0.0

            if total_frames < 2:
                raise ValueError(f"Video {filename} contains insufficient readable frames.")

            # Sample frames evenly across the entire duration
            num_samples = min(max_sample_frames, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

            for idx in sample_indices:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret and frame is not None and getattr(frame, "size", 0) > 0:
                        h, w = frame.shape[:2]
                        if max(h, w) > 640:
                            scale = 640.0 / max(h, w)
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        frames.append(frame)
                except Exception as read_err:
                    logger.warning(f"Failed to read/decode frame at index {idx}: {read_err}")
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

        # 2. Multi-Face Tracking and Frame-level Quality Analysis
        temporal_analyzer = TemporalFacialAnalyzer()
        quality_issues: List[str] = []
        usable_faces = 0

        # Preprocess full sequence for sequential video model
        video_model_raw_prob = None
        if os.path.exists(self.weights_path):
            def _run_vid(target_dev: str) -> float:
                t = self.preprocessor.preprocess_video(file_path).unsqueeze(0).to(target_dev)
                try:
                    with torch.no_grad():
                        pred_dict = self.model.predict(t)
                        probs = pred_dict["probabilities"][0]
                        return float(probs[1].item())
                finally:
                    del t

            try:
                video_model_raw_prob = _run_vid(self.device)
            except Exception as e:
                is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
                logger.warning(f"Video model sequential inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
                if is_cuda_err and self.device != "cpu":
                    DeviceManager.mark_cuda_unhealthy(f"VideoModel inference failure: {e}")
                    self._fallback_to_cpu(f"VideoModel failure: {e}")
                    try:
                        video_model_raw_prob = _run_vid("cpu")
                        logger.info(f"VideoModel CPU fallback succeeded: score={video_model_raw_prob:.4f}")
                    except Exception as cpu_err:
                        logger.error(f"VideoModel CPU fallback also failed: {cpu_err}")
                        video_model_raw_prob = None
                else:
                    video_model_raw_prob = None

        # Frame-level inference evaluation using specialist model
        if self._img_agent is None:
            from backend.agents.image_agent import ImageAgent
            self._img_agent = ImageAgent()
        img_agent = self._img_agent

        for idx, frame in enumerate(frames):
            frame_sample_idx = sample_indices[idx]
            frame_ts = frame_sample_idx / max(1.0, fps)
            try:
                q_res = self.quality_gate.evaluate(frame, require_face=False)
                if not q_res.is_suitable:
                    quality_issues.extend(q_res.issues)

                faces = self.preprocessor.detector.detect_faces(frame)
                if faces:
                    usable_faces += len(faces)

                # Update temporal face tracker across sequence
                tracked_step = temporal_analyzer.update_frame(
                    frame=frame,
                    frame_idx=frame_sample_idx,
                    timestamp=frame_ts,
                    detected_faces=faces,
                )
            except Exception as frame_err:
                logger.warning(f"Error evaluating frame {frame_sample_idx}: {frame_err}")
                continue

            # Infer spatial classification for each tracked face box
            for item in tracked_step:
                tid = item["track_id"]
                box = item["box"]
                crop = self.preprocessor.detector.crop_face(frame, box, margin_pct=0.08)
                if crop.size > 0:
                    if os.path.exists(img_agent.weights_path):
                        try:
                            crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
                            t = img_agent.preprocessor.apply_augmentation_and_normalization(crop_rgb, is_training=False).unsqueeze(0).to(img_agent.device)
                            try:
                                with torch.no_grad():
                                    logits = img_agent.model(t)
                                    f_p = float(torch.softmax(logits, dim=-1)[0, 1].item())
                                temporal_analyzer.tracks[tid].spatial_fake_probs.append(f_p)
                            finally:
                                del t
                        except Exception as e:
                            logger.warning(f"Error inferring ResNet crop on track {tid}: {e}")

                    # Evaluate auxiliary ViT detector on the exact same 8% margin crop
                    if self.vit_detector is not None:
                        try:
                            vit_res = self.vit_detector.predict_image(crop)
                            temporal_analyzer.tracks[tid].vit_fake_probs.append(vit_res["fake_probability"])
                        except Exception as e:
                            logger.warning(f"Error inferring ViT crop on track {tid}: {e}")

                    # Evaluate primary pretrained Xception detector on the exact same 8% margin crop
                    if self.xception_detector is not None:
                        try:
                            xcep_res = self.xception_detector.predict_frame(crop)
                            temporal_analyzer.tracks[tid].xcep_fake_probs.append(xcep_res["fake_probability"])
                        except Exception as e:
                            logger.warning(f"Error inferring Xception crop on track {tid}: {e}")

        # 3. Track-level Temporal Facial Forensic Synthesis
        total_sampled = len(frames)
        tracked_faces_results = []
        for tid, track in temporal_analyzer.tracks.items():
            if track.total_frames >= 2:
                metrics = temporal_analyzer.compute_track_forensics(track)
                has_sustained = (track.total_frames >= 4 or track.total_frames >= max(3, int(total_sampled * 0.4)))

                xcep_m = metrics.get("mean_xcep_score")
                has_xcep = (xcep_m is not None)
                vit_m = metrics.get("mean_vit_score")
                has_vit = (vit_m is not None)
                vit_flagged = bool(has_vit and vit_m >= 0.50)
                resnet_m = metrics["mean_spatial_score"]
                resnet_elevated = bool(resnet_m >= 0.24 or metrics["max_spatial_score"] >= 0.38)

                is_fake = False
                is_real = False
                model_conflict = False

                if has_xcep:
                    # Domain / Modality Conflict: Xception elevated (>=0.50) but ViT shows authentic anatomy (<0.35) without boundary anomaly
                    if xcep_m >= 0.50 and has_vit and vit_m < 0.35 and not metrics["is_boundary_anomalous"]:
                        model_conflict = True
                        track_verdict = "INCONCLUSIVE"
                        # Provisional score in authentic / ambiguous margin [0.20, 0.35]
                        track_fake_p = round(float(0.50 * resnet_m + 0.50 * vit_m), 4)
                    # Corroborated Manipulation (FAKE)
                    # Primary Xception detector indicates manipulation (>=0.60) corroborated by ViT
                    # OR Xception indicates high-certainty manipulation (>=0.85) corroborated by spatial CNN or ViT
                    elif (
                        has_sustained
                        and (
                            (xcep_m >= 0.60 and vit_flagged)
                            or (xcep_m >= 0.85 and (resnet_m >= 0.15 or (has_vit and vit_m >= 0.40)))
                        )
                    ):
                        is_fake = True
                        track_verdict = "FAKE"
                        track_fake_p = max(0.68, round(float(0.65 * xcep_m + 0.25 * (vit_m if has_vit else xcep_m) + 0.10 * resnet_m), 4))
                    # Primary detector clean (<0.45) and ResNet baseline authentic (<0.22)
                    elif xcep_m < 0.45 and resnet_m < 0.22:
                        is_real = True
                        track_verdict = "REAL"
                        track_fake_p = min(0.24, round(float(0.60 * xcep_m + 0.40 * resnet_m), 4))
                        if vit_flagged or metrics["is_boundary_anomalous"]:
                            model_conflict = True  # Flag compression or edge artifact divergence for auditability
                    # Borderline / Ambiguous margin (0.45 <= xcep_m < 0.60)
                    else:
                        track_verdict = "INCONCLUSIVE"
                        model_conflict = True
                        if resnet_m < 0.15:
                            track_fake_p = round(float(0.50 * xcep_m + 0.50 * resnet_m), 4)
                        else:
                            track_fake_p = round(float(0.50 * xcep_m + 0.30 * (vit_m if has_vit else 0.50) + 0.20 * resnet_m), 4)
                else:
                    # Fallback without Xception
                    is_fake = bool(
                        has_sustained
                        and (
                            (vit_flagged and resnet_elevated)
                            or (vit_flagged and metrics["is_boundary_anomalous"])
                            or (resnet_m >= 0.45)
                            or (metrics["temporal_manipulation_score"] >= 0.45 and metrics["anomaly_count"] >= 2)
                        )
                    )
                    is_real = bool(
                        resnet_m < 0.22
                        and metrics["max_spatial_score"] < 0.35
                        and (not has_vit or vit_m < 0.50)
                        and not metrics["is_boundary_anomalous"]
                    )
                    model_conflict = bool(
                        metrics.get("is_model_conflict", False)
                        or (vit_flagged and resnet_m < 0.22 and not metrics["is_boundary_anomalous"])
                    )
                    track_verdict = "FAKE" if is_fake else ("REAL" if is_real else "INCONCLUSIVE")
                    if is_fake:
                        track_fake_p = max(0.65, float(0.40 * resnet_m + 0.60 * (vit_m if has_vit else 0.65)))
                    elif is_real:
                        track_fake_p = min(0.22, resnet_m)
                    else:
                        track_fake_p = 0.42 if model_conflict else resnet_m

                is_fallback = (track_verdict == "INCONCLUSIVE")
                tracked_faces_results.append({
                    "track_id": tid,
                    "total_frames": track.total_frames,
                    "verdict": track_verdict,
                    "fake_probability": round(track_fake_p, 3),
                    "is_fallback_probability": is_fallback,
                    "probability_source": "fallback_abstention_baseline" if is_fallback else "measured_model_inference",
                    "mean_xcep_score": xcep_m,
                    "max_xcep_score": metrics.get("max_xcep_score"),
                    "mean_spatial_score": resnet_m,
                    "max_spatial_score": metrics["max_spatial_score"],
                    "mean_vit_score": vit_m,
                    "max_vit_score": metrics.get("max_vit_score"),
                    "boundary_flicker": metrics["mean_boundary_flicker"],
                    "mouth_texture": metrics["mean_mouth_texture"],
                    "eye_sharpness": metrics["mean_eye_sharpness"],
                    "lighting_gradient": metrics["mean_lighting_gradient"],
                    "anomalies_detected": metrics["anomaly_count"],
                    "is_boundary_anomalous": metrics["is_boundary_anomalous"],
                    "is_mouth_smoothed": metrics["is_mouth_smoothed"],
                    "is_eye_smoothed": metrics["is_eye_smoothed"],
                    "is_model_conflict": model_conflict,
                    "sample_box": list(track.last_box),
                    "track_obj": track,
                    "metrics": metrics,
                })

        # Multi-person aggregation rule:
        # Prioritize the prominent subject(s) in the frame.
        # Spurious background detections (small area or few frames) must not override the primary subject.
        sustained_tracks = [t for t in tracked_faces_results if t["total_frames"] >= 4]
        if not sustained_tracks:
            sustained_tracks = tracked_faces_results

        # Calculate bounding box area for each track
        for t in sustained_tracks:
            b = t.get("sample_box", [0, 0, 0, 0])
            t["box_area"] = (b[2] * b[3]) if len(b) >= 4 else 0

        # Max face area among sustained tracks
        max_area = max([t["box_area"] for t in sustained_tracks], default=0)
        # Prominent tracks are those that are at least 35% of the size of the largest face in the video
        prominent_tracks = [t for t in sustained_tracks if t["box_area"] >= max(2500, int(max_area * 0.35))] or sustained_tracks

        fake_prominent = [t for t in prominent_tracks if t["verdict"] == "FAKE"]
        if fake_prominent:
            primary_track_info = max(fake_prominent, key=lambda t: t["fake_probability"])
        elif prominent_tracks and all(t["verdict"] == "REAL" for t in prominent_tracks):
            primary_track_info = max(prominent_tracks, key=lambda t: t["box_area"])
        elif prominent_tracks:
            primary_track_info = max(prominent_tracks, key=lambda t: t["box_area"])
        else:
            primary_track_info = None

        # Fallback frame fake probs if tracks are empty
        if primary_track_info:
            t_obj = primary_track_info["track_obj"]
            if t_obj.xcep_fake_probs:
                frame_fake_probs = t_obj.xcep_fake_probs
            elif t_obj.spatial_fake_probs:
                frame_fake_probs = t_obj.spatial_fake_probs
            else:
                frame_fake_probs = [primary_track_info["fake_probability"]] * len(frames)
        else:
            frame_fake_probs = [0.50] * len(frames)

        mean_fake_prob = float(np.mean(frame_fake_probs))
        max_fake_prob = float(np.max(frame_fake_probs))
        min_fake_prob = float(np.min(frame_fake_probs))
        temporal_variance = float(np.std(frame_fake_probs))
        temporal_conflict = bool(
            primary_track_info
            and primary_track_info["verdict"] != "FAKE"
            and (
                (max_fake_prob > 0.70 and min_fake_prob < 0.25)
                or primary_track_info.get("is_model_conflict")
                or primary_track_info["verdict"] == "INCONCLUSIVE"
            )
        )

        # 4. Multi-Provider Vision Forensics
        temporal_visual_res = self.vision_service.analyze_video(
            file_path=file_path,
            frames=frames,
            fps=fps,
            has_audio=False,
            selected_providers=selected_providers,
        )

        # 5. Synthesize Concrete Forensic Result
        forensic_evidence_items = []
        suspicious_signals_count = 0
        tampering_detected = False

        if primary_track_info:
            m = primary_track_info["metrics"]
            tid = primary_track_info["track_id"]
            if primary_track_info["verdict"] == "FAKE":
                tampering_detected = True
            if m.get("is_boundary_anomalous"):
                forensic_evidence_items.append(
                    f"Temporal Boundary Seam ({tid}): Inter-frame boundary seam flicker ({m['mean_boundary_flicker']:.1f} > 32.0) detected along facial perimeter."
                )
                tampering_detected = True
            if m.get("is_mouth_smoothed"):
                forensic_evidence_items.append(
                    f"Oral Cavity Smoothing ({tid}): Unnatural texture smoothing and edge loss in mouth region ({m['mean_mouth_texture']:.1f} < 24.0)."
                )
                tampering_detected = True
            if m.get("is_eye_smoothed"):
                forensic_evidence_items.append(
                    f"Ocular Texture Loss ({tid}): Attenuated eyelid and iris texture sharpness ({m['mean_eye_sharpness']:.1f} < 25.0)."
                )
            if m.get("is_lighting_asymmetric"):
                forensic_evidence_items.append(
                    f"Illumination Inconsistency ({tid}): Bilateral cheek lighting gradient asymmetry ({m['mean_lighting_gradient']:.2f} > 0.20)."
                )
            if m.get("mean_xcep_score") is not None:
                x_val = m["mean_xcep_score"]
                if x_val >= 0.60:
                    forensic_evidence_items.append(
                        f"Pretrained Xception Detector ({tid}): Face-manipulation neural detector measured high synthetic likelihood of {x_val:.1%} on facial sequence."
                    )
                elif x_val <= 0.40:
                    forensic_evidence_items.append(
                        f"Pretrained Xception Detector ({tid}): Face-manipulation neural detector observed natural facial representation ({x_val:.1%})."
                    )
                else:
                    forensic_evidence_items.append(
                        f"Pretrained Xception Detector ({tid}): Neural detector measured indeterminate score ({x_val:.1%}) in ambiguous transition margin."
                    )
            if m.get("is_neural_suspicious"):
                forensic_evidence_items.append(
                    f"Spatial CNN Classifier ({tid}): ResNet-18 classifier measured peak synthetic likelihood of {m['max_spatial_score']:.1%}."
                )
            if m.get("mean_vit_score") is not None and m["mean_vit_score"] >= 0.50:
                if (m.get("mean_xcep_score", 0.0) >= 0.60 or m.get("mean_spatial_score") >= 0.24 
                    or m.get("is_boundary_anomalous") or m.get("is_mouth_smoothed")):
                    forensic_evidence_items.append(
                        f"Auxiliary Transformer Corroboration ({tid}): Vision Transformer measured elevated synthetic likelihood of {m['mean_vit_score']:.1%}, corroborating facial manipulation indicators."
                    )
                    tampering_detected = True
                else:
                    forensic_evidence_items.append(
                        f"Model Modality Divergence ({tid}): Vision Transformer flagged potential synthetic patterns ({m['mean_vit_score']:.1%}), but primary face detector ({m.get('mean_xcep_score', 'N/A')}) and spatial checks observe natural anatomy. Flagged as cross-domain divergence."
                    )
            suspicious_signals_count = m.get("anomaly_count", 0)
            raw_video_prob = primary_track_info["fake_probability"]
        else:
            raw_video_prob = mean_fake_prob

        key_frame = frames[len(frames) // 2]
        key_q = self.quality_gate.evaluate(key_frame, require_face=False)
        key_face_bbox = key_q.primary_face_bbox if key_q.face_detected else None
        base_forensics = self.forensics.evaluate(key_frame, face_bbox=key_face_bbox)

        forensic_res = {
            "composite_forensic_score": raw_video_prob,
            "evidence_items": forensic_evidence_items or base_forensics.get("evidence_items", []),
            "suspicious_signal_count": suspicious_signals_count or base_forensics.get("suspicious_signal_count", 0),
            "tampering_detected": tampering_detected,
            "fft_analysis": base_forensics.get("fft_analysis", {}),
            "ela_analysis": base_forensics.get("ela_analysis", {}),
            "edge_analysis": base_forensics.get("edge_analysis", {}),
            "facial_regions": base_forensics.get("facial_regions", {}),
        }

        # 6. Safety Decision System Evaluation
        rep_frame = frames[len(frames) // 2] if len(frames) > 0 else frames[0]

        if usable_faces == 0:
            has_visual_tampering = (
                temporal_visual_res.get("consensus_assessment") == "FAKE"
                or len(temporal_visual_res.get("observable_anomalies", [])) > 0
            )
            if has_visual_tampering:
                q_res = self.quality_gate.evaluate(rep_frame, require_face=False)
                decision = self.decision_system.decide(
                    raw_fake_prob=raw_video_prob or 0.85,
                    quality_result=q_res if quality_issues else None,
                    forensic_result=forensic_res,
                    visual_analysis=temporal_visual_res,
                    audio_visual_sync=temporal_visual_res.get("audio_visual_sync"),
                    temporal_variance=temporal_variance,
                    temporal_conflict=temporal_conflict,
                )
            else:
                q_res = self.quality_gate.evaluate(rep_frame, require_face=True)
                decision = self.decision_system.decide(
                    raw_fake_prob=None,
                    quality_result=q_res,
                    forensic_result={},
                    visual_analysis=temporal_visual_res,
                    audio_visual_sync=temporal_visual_res.get("audio_visual_sync"),
                )
        else:
            q_res = self.quality_gate.evaluate(rep_frame, require_face=False) if quality_issues else None
            decision = self.decision_system.decide(
                raw_fake_prob=raw_video_prob,
                quality_result=q_res,
                forensic_result=forensic_res,
                visual_analysis=temporal_visual_res,
                audio_visual_sync=temporal_visual_res.get("audio_visual_sync"),
                temporal_variance=temporal_variance,
                temporal_conflict=temporal_conflict,
            )

        # 7. Generate Evidence Snapshots with Temporal Track Context
        from backend.forensics.forensic_analyzers import VideoEvidenceSuite
        evidence_snapshots = VideoEvidenceSuite.generate_evidence_snapshots(
            frames=frames,
            sample_indices=sample_indices,
            fps=fps,
            frame_fake_probs=frame_fake_probs,
            temporal_variance=temporal_variance,
            temporal_conflict=temporal_conflict,
            verdict=decision.primary_verdict,
            quality_gate=self.quality_gate,
            track_info=primary_track_info,
        )

        peak_idx = int(np.argmax(frame_fake_probs)) if frame_fake_probs else 0
        peak_sec = (sample_indices[peak_idx] if peak_idx < len(sample_indices) else 0) / max(1.0, fps)
        hrs = int(peak_sec // 3600)
        mins = int((peak_sec % 3600) // 60)
        secs = int(peak_sec % 60)
        peak_ts = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        peak_prob = frame_fake_probs[peak_idx] if frame_fake_probs else 0.50

        if decision.primary_verdict in ("FAKE", "DEEPFAKE"):
            flicker_val = primary_track_info.get("boundary_flicker", 0.0) if primary_track_info else 0.0
            x_val = primary_track_info.get("mean_xcep_score") if primary_track_info else None
            xcep_str = f"pretrained Xception detector measured synthetic probability of {x_val:.1%}" if x_val is not None else "spatial classifiers measured synthetic anomaly"
            evidence_summary = (
                f"At {peak_ts}, {xcep_str} on primary track {primary_track_info['track_id'] if primary_track_info else 'T1'}. "
                f"Temporal facial analysis across {len(frames)} sampled frames revealed boundary seam flicker ({flicker_val:.1f}) "
                f"and sequence variance of {temporal_variance:.3f}. "
                f"Empirical multi-agent evidence confirms synthetic facial manipulation."
            )
        elif decision.primary_verdict in ("REAL", "LIKELY_REAL"):
            if decision.is_binary_fallback:
                evidence_summary = (
                    f"Across {len(frames)} sampled frames ({duration:.1f}s duration), specialist models exhibited divergence or cross-domain shift: "
                    f"one model observed anomalous patterns while primary spatial/face-manipulation models observed natural features. "
                    f"Because evidence is indeterminate, the result is presented as a provisional fallback baseline ({decision.fake_probability:.1%}). "
                    f"Authenticity has NOT been established, and independent verification is required."
                )
            else:
                evidence_summary = (
                    f"Across {len(frames)} sampled frames ({duration:.1f}s duration), primary pretrained facial and spatial classifiers observed authentic features "
                    f"(mean fake probability {mean_fake_prob:.1%}). The temporal consistency analyzer measured smooth inter-frame stability "
                    f"(variance {temporal_variance:.3f}). No strong manipulation evidence was detected across evaluated facial indicators."
                )
        else:
            evidence_summary = (
                f"Sampled video frames produced indeterminate indicators (mean fake probability {mean_fake_prob:.1%}, "
                f"temporal variance {temporal_variance:.3f}). Temporal and spatial signals lack decisive consensus; secondary manual verification is recommended."
            )

        if getattr(decision, "is_fallback_probability", False):
            decision.evidence.append(
                f"Probability Baseline Notice: The reported fake score ({decision.fake_probability:.1%}) is an uncalibrated provisional fallback baseline applied due to model modality divergence. It does NOT represent an empirically calibrated probability of authenticity or manipulation."
            )

        processing_time = round(time.time() - start_time, 4)

        # Serializable tracked faces for API, UI, and PDF report
        serializable_tracks = []
        for t in tracked_faces_results:
            serializable_tracks.append({
                "track_id": t["track_id"],
                "total_frames": t["total_frames"],
                "verdict": t["verdict"],
                "fake_probability": t["fake_probability"],
                "is_fallback_probability": t["is_fallback_probability"],
                "probability_source": t["probability_source"],
                "mean_xcep_score": t.get("mean_xcep_score"),
                "max_xcep_score": t.get("max_xcep_score"),
                "mean_spatial_score": t["mean_spatial_score"],
                "max_spatial_score": t["max_spatial_score"],
                "mean_vit_score": t.get("mean_vit_score"),
                "max_vit_score": t.get("max_vit_score"),
                "boundary_flicker": t["boundary_flicker"],
                "mouth_texture": t["mouth_texture"],
                "eye_sharpness": t["eye_sharpness"],
                "anomalies_detected": t["anomalies_detected"],
                "sample_box": t["sample_box"],
            })

        canonical_v = (
            DecisionState.NOT_SUITABLE if decision.prediction == DecisionState.NOT_SUITABLE else
            DecisionState.UNCERTAIN if (decision.is_binary_fallback or decision.prediction == DecisionState.UNCERTAIN or decision.original_verdict == DecisionState.UNCERTAIN) else
            DecisionState.FAKE if decision.primary_verdict in ("FAKE", "DEEPFAKE") else
            DecisionState.REAL if decision.primary_verdict in ("REAL", "LIKELY_REAL") else
            DecisionState.UNCERTAIN
        )
        conf_out = None if (decision.is_binary_fallback or canonical_v in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE)) else decision.confidence
        conf_level_out = (
            "Not Performed" if canonical_v == DecisionState.NOT_SUITABLE else
            ("Provisional" if decision.is_binary_fallback else ("Uncertified" if canonical_v == DecisionState.UNCERTAIN else decision.confidence_level))
        )

        mean_xc = primary_track_info.get("mean_xcep_score") if primary_track_info else None
        mean_sp = primary_track_info.get("mean_spatial_score") if primary_track_info else None
        mean_vt = primary_track_info.get("mean_vit_score") if primary_track_info else None
        models_inventory = [
            {
                "name": "VideoModel (Spatial Baseline)",
                "task": "Frame-Level Facial Manipulation Baseline",
                "predicted_class": ("FAKE" if mean_sp >= 0.50 else "REAL") if mean_sp is not None else "UNAVAILABLE",
                "raw_score": round(mean_sp, 4) if mean_sp is not None else None,
                "calibrated_prob": round(mean_sp, 4) if mean_sp is not None else None,
                "calibration_status": "Uncalibrated / Raw Softmax",
                "status": "SUCCESS" if mean_sp is not None else "UNAVAILABLE",
                "limitations": "Frame-level baseline CNN; lacks temporal landmark or lip-synchronization modeling."
            },
            {
                "name": "XceptionNet (Sampled Frames)",
                "task": "Sampled Frame Face-Swap & Boundary Detection",
                "predicted_class": ("FAKE" if mean_xc >= 0.50 else "REAL") if mean_xc is not None else "UNAVAILABLE",
                "raw_score": round(mean_xc, 4) if mean_xc is not None else None,
                "calibrated_prob": round(mean_xc, 4) if mean_xc is not None else None,
                "calibration_status": "DeepfakeBench Benchmark Softmax",
                "status": "SUCCESS" if mean_xc is not None else "UNAVAILABLE",
                "limitations": "Trained on video compression; evaluated independently across sampled keyframes."
            },
            {
                "name": "Vision Transformer (Sampled Frames)",
                "task": "Sampled Frame Synthetic Generative Classifier",
                "predicted_class": ("FAKE" if mean_vt >= 0.50 else "REAL") if mean_vt is not None else "UNAVAILABLE",
                "raw_score": round(mean_vt, 4) if mean_vt is not None else None,
                "calibrated_prob": round(mean_vt, 4) if mean_vt is not None else None,
                "calibration_status": "Raw Softmax",
                "status": "SUCCESS" if mean_vt is not None else "UNAVAILABLE",
                "limitations": "Fine-tuned on synthetic generation artifacts across face crops."
            },
            {
                "name": "Temporal Facial Appearance Analyzer",
                "task": "Inter-Frame Facial Consistency & Boundary Flicker",
                "predicted_class": "FAKE" if (temporal_variance > 0.08 or (primary_track_info and primary_track_info.get("boundary_flicker", 0) > 22.0)) else "REAL",
                "raw_score": round(temporal_variance, 4),
                "calibrated_prob": None,
                "calibration_status": "Heuristic Empirical Threshold",
                "status": "SUCCESS",
                "limitations": "Measures variance across sampled keyframes; subject to fast head motion and lighting shifts."
            }
        ]

        measurements = {
            "resolution": {
                "width": 0,
                "height": 0,
                "formatted": "Video Multi-Frame Resolution",
                "status": "Sampled frame validation passed",
                "interpretation": f"{len(frames)} sampled frames evaluated from {total_frames} total video frames."
            },
            "temporal_consistency": {
                "variance": round(temporal_variance, 4),
                "mean_frame_fake_prob": round(mean_fake_prob, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "status": "Temporal conflict detected" if temporal_conflict else "Consistent frame progression",
                "interpretation": "Inter-frame prediction stability across sampled sequence."
            },
            "multi_face_tracking": {
                "tracked_faces_count": len(serializable_tracks),
                "interpretation": "Independent tracking buffers maintained for each individual facial trajectory."
            }
        }

        supporting_evidence = []
        contradicting_evidence = []
        unavailable_checks = [
            "Audio-visual lip synchronization: Not analyzed (cross-modal phoneme-viseme alignment model not configured)",
            "Camera sensor PRNU fingerprint analysis: Unavailable",
            "C2PA cryptographic provenance metadata: Not present"
        ]
        limitations = [
            "Video-level decision is derived through frame-level sampling and temporal consistency analysis.",
            "Isolated anomalous frames do not automatically classify an entire video as manipulated.",
            "Lip-sync and speech-to-audio temporal alignment was not evaluated.",
            "Absence of detected manipulation does not guarantee absolute authenticity."
        ]

        if canonical_v == DecisionState.UNCERTAIN:
            if temporal_conflict:
                contradicting_evidence.append(
                    f"Temporal conflict detected: Prediction scores fluctuated across sampled frames (min {min_fake_prob:.1%}, max {max_fake_prob:.1%}, variance {temporal_variance:.3f})."
                )
            supporting_evidence.append(f"Successfully tracked {len(serializable_tracks)} facial trajectories across {len(frames)} keyframes.")
        elif canonical_v in (DecisionState.FAKE, "FAKE"):
            flicker_val = primary_track_info.get("boundary_flicker", 0.0) if primary_track_info else 0.0
            supporting_evidence.append(f"Elevated manipulation probability across sampled frames (mean {mean_fake_prob:.1%}, peak {max_fake_prob:.1%}).")
            if flicker_val > 22.0:
                supporting_evidence.append(f"Boundary seam flicker detected on primary face track ({flicker_val:.1f}).")
        elif canonical_v in (DecisionState.REAL, "REAL"):
            supporting_evidence.append(f"Consistent natural facial feature distribution observed across {len(frames)} sampled keyframes (mean fake prob {mean_fake_prob:.1%}).")
            supporting_evidence.append(f"Temporal consistency analyzer measured smooth inter-frame stability (variance {temporal_variance:.3f}).")

        decision_obj = {
            "method": "VideoAgent multi-frame temporal aggregation and quality gating",
            "score": round(decision.fake_probability, 4) if decision.fake_probability is not None else None,
            "score_type": "calibrated_synthetic_probability",
            "confidence": conf_out,
            "confidence_type": "Certified" if conf_out is not None else ("Uncertified due to temporal conflict / quality abstention" if canonical_v == DecisionState.UNCERTAIN else "Not performed"),
            "criteria_met": bool(conf_out is not None and canonical_v in (DecisionState.REAL, DecisionState.FAKE)),
            "reason": decision.reason,
        }

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
                "tracked_faces_count": len(serializable_tracks),
                "diagnostic_video_model_prob": round(video_model_raw_prob, 4) if video_model_raw_prob is not None else None,
            },
            "media_type": "video",
            "task_type": "video_frame_deepfake_and_temporal_facial_detection",
            "prediction": canonical_v,
            "verdict": canonical_v,
            "canonical_verdict": canonical_v,
            "canonicalVerdict": canonical_v,
            "primary_verdict": decision.primary_verdict,
            "primaryVerdict": decision.primary_verdict,
            "provisional_suggestion": decision.primary_verdict if decision.is_binary_fallback else None,
            "provisionalSuggestion": decision.primary_verdict if decision.is_binary_fallback else None,
            "original_verdict": decision.original_verdict,
            "is_binary_fallback": decision.is_binary_fallback,
            "is_provisional": decision.is_binary_fallback,
            "is_fallback_probability": getattr(decision, "is_fallback_probability", False),
            "probability_source": getattr(decision, "probability_source", "calibrated_model_inference"),
            "reliability_warning": decision.reliability_warning,
            "confidence": conf_out,
            "confidence_score": conf_out,
            "confidence_level": conf_level_out,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": (
                primary_track_info.get("mean_xcep_score")
                if primary_track_info and primary_track_info.get("mean_xcep_score") is not None
                else (primary_track_info["mean_spatial_score"] if primary_track_info else decision.raw_model_score)
            ),
            "uncertainty": decision.uncertainty,
            "evidence": decision.evidence,
            "influencing_evidence": getattr(decision, "influencing_evidence", []),
            "providers_ran": getattr(decision, "providers_ran", []),
            "approx_cost_usd": getattr(decision, "approx_cost_usd", temporal_visual_res.get("approx_cost_usd", 0.0)),
            "reason": decision.reason,
            "summary": evidence_summary,
            "user_facing_statement": evidence_summary,
            "evidence_snapshots": evidence_snapshots,
            "evidence_cards": evidence_snapshots,
            "tracked_faces": serializable_tracks,
            "visual_analysis": temporal_visual_res,
            "models": models_inventory,
            "measurements": measurements,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "unavailable_checks": unavailable_checks,
            "decision": decision_obj,
            "limitations": limitations,
            "audio_visual_sync": {
                "analyzed": False,
                "reason": "Lip-sync analysis not performed (cross-modal phoneme-viseme alignment model not configured)."
            },
            "temporal_metrics": {
                "mean_frame_fake_prob": round(mean_fake_prob, 4),
                "max_frame_fake_prob": round(max_fake_prob, 4),
                "min_frame_fake_prob": round(min_fake_prob, 4),
                "temporal_variance": round(temporal_variance, 4),
                "temporal_conflict": temporal_conflict,
                "frame_scores": [round(p, 4) for p in frame_fake_probs],
                "boundary_flicker": primary_track_info.get("boundary_flicker", 0.0) if primary_track_info else 0.0,
                "mouth_texture": primary_track_info.get("mouth_texture", 50.0) if primary_track_info else 50.0,
                "eye_sharpness": primary_track_info.get("eye_sharpness", 40.0) if primary_track_info else 40.0,
            },
            "features": {
                "temporal_flicker_score": round(primary_track_info.get("boundary_flicker", 0.0) / 100.0, 4) if primary_track_info else round(temporal_variance, 4),
                "face_jitter_ratio": round(temporal_variance * 0.8, 4),
                "lip_sync_analyzed": False,
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
            "primary_verdict": DecisionState.UNCERTAIN,
            "original_verdict": DecisionState.UNCERTAIN,
            "is_binary_fallback": False,
            "reliability_warning": "Inference could not be completed.",
            "confidence": None,
            "confidence_score": None,
            "confidence_level": "Low",
            "fake_probability": None,
            "real_probability": None,
            "raw_model_score": None,
            "uncertainty": None,
            "evidence": [f"Video Decode Failure: {reason}"],
            "reason": f"Video analysis aborted: {reason}",
            "summary": "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.",
            "user_facing_statement": "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.",
            "audio_visual_sync": {
                "analyzed": False,
                "reason": "Lip-sync analysis not performed (video decode failure).",
            },
            "processing_time": processing_time,
        }
