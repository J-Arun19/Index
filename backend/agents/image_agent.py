import os
import cv2
import time
import json
from datetime import datetime
import torch
from typing import Optional, Dict, Any, List, Tuple

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.utils.device_manager import DeviceManager
from backend.models.image_model import ImageModel
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.services.vision_service import VisionAnalysisService
from backend.core.decision_system import SafetyDecisionSystem, DecisionState

logger = setup_logger("ImageAgent")


class ImageAgent:
    """
    Forensic image analysis agent.
    Combines input quality validation, genuine frequency/compression forensics,
    vision-capable visual reasoning, calibrated neural network inference, and
    a consistent decision engine.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or PipelineConfig.IMAGE_MODEL_PATH
        self.device = DeviceManager.get_device_str()
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.vision_service = VisionAnalysisService()
        self.decision_system = SafetyDecisionSystem()
        self.preprocessor = ImagePreprocessor()
        self._model: Optional[ImageModel] = None
        self._vit_detector: Optional[Any] = None
        self._xception_detector: Optional[Any] = None

    def _fallback_to_cpu(self, reason: str = "") -> None:
        """Safely transitions all cached specialist models to CPU and switches self.device to 'cpu'."""
        if self.device == "cpu" and (self._model is None or not next(self._model.parameters(), torch.empty(0)).is_cuda):
            return

        logger.info(f"ImageAgent: Switching device from {self.device} to cpu. Reason: {reason}")
        self.device = "cpu"

        # 1. ResNet baseline model
        if self._model is not None:
            try:
                self._model.to("cpu")
                logger.info("ImageAgent: Moved ResNet-18 model to CPU.")
            except Exception as e:
                logger.warning(f"Failed to move ResNet model to CPU directly ({e}), reloading from checkpoint on CPU...")
                try:
                    model = ImageModel(num_classes=2, pretrained=False)
                    if os.path.exists(self.weights_path):
                        model.load_weights(self.weights_path)
                    model.to("cpu")
                    model.eval()
                    self._model = model
                except Exception as re_err:
                    logger.error(f"Failed to reload ResNet on CPU: {re_err}")
                    self._model = None

        # 2. Xception detector
        if self._xception_detector is not None and self._xception_detector is not False:
            try:
                self._xception_detector.to("cpu")
                logger.info("ImageAgent: Moved XceptionDetector to CPU.")
            except Exception as e:
                logger.warning(f"Failed to move Xception to CPU directly ({e}), reloading on CPU...")
                try:
                    from backend.models.experimental.xception_detector import XceptionDetector
                    self._xception_detector = XceptionDetector(weights_path=self._xception_detector.weights_path, device="cpu")
                except Exception as re_err:
                    logger.error(f"Failed to reload Xception on CPU: {re_err}")
                    self._xception_detector = False

        # 3. ViT detector
        if self._vit_detector is not None and self._vit_detector is not False:
            try:
                self._vit_detector.to("cpu")
                logger.info("ImageAgent: Moved ViTDetector to CPU.")
            except Exception as e:
                logger.warning(f"Failed to move ViT to CPU directly ({e}), reloading on CPU...")
                try:
                    from backend.models.experimental.vit_detector import ViTDetector
                    self._vit_detector = ViTDetector(device="cpu")
                except Exception as re_err:
                    logger.error(f"Failed to reload ViT on CPU: {re_err}")
                    self._vit_detector = False

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
                logger.info(f"ImageAgent: Loaded auxiliary ViTDetector on {self.device}")
            except Exception as e:
                logger.warning(f"ImageAgent: ViTDetector auxiliary model not loaded: {e}")
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
                    logger.info(f"ImageAgent: Loaded pretrained XceptionDetector on {self.device}")
                else:
                    self._xception_detector = False
            except Exception as e:
                logger.warning(f"ImageAgent: XceptionDetector model not loaded: {e}")
                self._xception_detector = False
        return self._xception_detector if self._xception_detector is not False else None

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
            
            if self.device == "cuda" and not DeviceManager.is_cuda_healthy():
                self.device = "cpu"
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def analyze(
        self,
        file_path: str,
        require_face: bool = True,
        selected_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete forensic verification on an image file.
        Combines specialist neural detector with visual reasoning and frequency forensics.
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
                visual_res={},
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=processing_time,
                img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
            )

        # 2. Genuine Forensic Evidence
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox) if img_bgr is not None else {}

        # 3. Vision-Capable Visual Reasoning (Observes face regions, symmetry, boundaries, compression)
        visual_res = self.vision_service.analyze_image(
            file_path,
            face_bbox=face_bbox,
            quality_gate_result=quality_res,
            selected_providers=selected_providers,
        )

        # If quality gate failed (LOW QUALITY: blurry or face too small), abstain via decision system
        if not quality_res.is_suitable:
            decision = self.decision_system.decide(
                raw_fake_prob=None,
                quality_result=quality_res,
                forensic_result=forensic_res,
                visual_analysis=visual_res,
            )
            processing_time = round(time.time() - start_time, 4)
            evidence_cards = []
            if img_bgr is not None and face_bbox is not None:
                try:
                    evidence_cards = self.forensics.generate_evidence_cards(
                        forensic_res, img_bgr, face_bbox=face_bbox, verdict=decision.verdict
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate evidence cards: {e}")
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res=forensic_res,
                visual_res=visual_res,
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=processing_time,
                img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
                face_bbox=face_bbox,
                evidence_cards=evidence_cards,
            )

        # 4. Specialist Multi-Model Inference (Xception primary + ViT auxiliary + ResNet baseline)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")

        # Ensure device status is healthy before beginning inference
        if self.device == "cuda" and not DeviceManager.is_cuda_healthy():
            self._fallback_to_cpu("CUDA context probe unhealthy prior to inference start")

        # Prepare face crop (with padding) or full image
        face_crop_bgr = None
        if face_bbox is not None and img_bgr is not None:
            x, y, bw, bh = face_bbox
            pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(img_bgr.shape[1], x + bw + pad_x)
            y1 = min(img_bgr.shape[0], y + bh + pad_y)
            face_crop_bgr = img_bgr[y0:y1, x0:x1]
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            face_crop_bgr = img_bgr

        # 4a. ResNet-18 baseline inference with safe CPU fallback
        resnet_fake_p = None
        def _run_resnet(target_dev: str) -> float:
            t = self.preprocessor.preprocess_image(file_path).unsqueeze(0).to(target_dev)
            try:
                with torch.no_grad():
                    pred_dict = self.model.predict(t)
                    probs = pred_dict["probabilities"][0]
                    return float(probs[1].item())
            finally:
                del t

        try:
            resnet_fake_p = _run_resnet(self.device)
        except Exception as e:
            is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
            logger.warning(f"ImageAgent: ResNet-18 inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
            if is_cuda_err and self.device != "cpu":
                DeviceManager.mark_cuda_unhealthy(f"ResNet-18 failure: {e}")
                self._fallback_to_cpu(f"ResNet-18 CUDA failure: {e}")
                try:
                    resnet_fake_p = _run_resnet("cpu")
                    logger.info(f"ImageAgent: ResNet-18 CPU fallback succeeded: score={resnet_fake_p:.4f}")
                except Exception as cpu_err:
                    logger.error(f"ImageAgent: ResNet-18 CPU fallback also failed: {cpu_err}")
                    resnet_fake_p = None
            else:
                logger.error(f"ImageAgent: ResNet-18 non-recoverable error: {e}")
                resnet_fake_p = None

        # 4b. Pretrained DeepfakeBench Xception inference with safe CPU fallback
        xcep_fake_p = None
        if self.xception_detector is not None and face_crop_bgr is not None and face_crop_bgr.size > 0:
            crop_256 = cv2.resize(face_crop_bgr, (256, 256))
            crop_rgb = cv2.cvtColor(crop_256, cv2.COLOR_BGR2RGB)

            def _run_xcep(target_dev: str) -> float:
                t = (torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
                t = t.unsqueeze(0).to(target_dev)
                try:
                    with torch.no_grad():
                        xcep_logits = self.xception_detector(t)
                        xcep_probs = torch.softmax(xcep_logits, dim=-1)[0]
                        return float(xcep_probs[1].item())
                finally:
                    del t

            try:
                xcep_fake_p = _run_xcep(self.device)
            except Exception as e:
                is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
                logger.warning(f"ImageAgent: Xception inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
                if is_cuda_err and self.device != "cpu":
                    DeviceManager.mark_cuda_unhealthy(f"Xception failure: {e}")
                    self._fallback_to_cpu(f"Xception CUDA failure: {e}")
                    try:
                        xcep_fake_p = _run_xcep("cpu")
                        logger.info(f"ImageAgent: Xception CPU fallback succeeded: score={xcep_fake_p:.4f}")
                    except Exception as cpu_err:
                        logger.error(f"ImageAgent: Xception CPU fallback also failed: {cpu_err}")
                        xcep_fake_p = None
                else:
                    logger.error(f"ImageAgent: Xception non-recoverable error: {e}")
                    xcep_fake_p = None

        # 4c. Pretrained Image ViT inference with safe CPU fallback
        vit_fake_p = None
        if self.vit_detector is not None and face_crop_bgr is not None and face_crop_bgr.size > 0:
            from PIL import Image as PILImage
            crop_pil = PILImage.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))

            def _run_vit(target_dev: str) -> float:
                vit_inputs = self.vit_detector.processor(images=crop_pil, return_tensors="pt")
                vit_pixels = vit_inputs["pixel_values"].to(target_dev)
                try:
                    with torch.no_grad():
                        vit_logits = self.vit_detector(vit_pixels)
                        vit_probs = torch.softmax(vit_logits, dim=-1)[0]
                        return float(vit_probs[1].item())
                finally:
                    del vit_inputs, vit_pixels

            try:
                vit_fake_p = _run_vit(self.device)
            except Exception as e:
                is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
                logger.warning(f"ImageAgent: ViT inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
                if is_cuda_err and self.device != "cpu":
                    DeviceManager.mark_cuda_unhealthy(f"ViT failure: {e}")
                    self._fallback_to_cpu(f"ViT CUDA failure: {e}")
                    try:
                        vit_fake_p = _run_vit("cpu")
                        logger.info(f"ImageAgent: ViT CPU fallback succeeded: score={vit_fake_p:.4f}")
                    except Exception as cpu_err:
                        logger.error(f"ImageAgent: ViT CPU fallback also failed: {cpu_err}")
                        vit_fake_p = None
                else:
                    logger.error(f"ImageAgent: ViT non-recoverable error: {e}")
                    vit_fake_p = None

        # 4d. Model Evidence Fusion & Partial-Failure Handling
        has_resnet = resnet_fake_p is not None
        has_xcep = xcep_fake_p is not None
        has_vit = vit_fake_p is not None

        if not (has_resnet or has_xcep or has_vit):
            error_msg = f"Model inference failed for {filename}: All specialist neural models failed inference (GPU and CPU retry exhausted)."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        r_val = resnet_fake_p if has_resnet else 0.50
        is_conflict = False

        if has_xcep and has_vit:
            diff = abs(xcep_fake_p - vit_fake_p)
            both_fake = (xcep_fake_p >= 0.50) and (vit_fake_p >= 0.50)
            both_real = (xcep_fake_p < 0.50) and (vit_fake_p < 0.50)

            if both_fake:
                # Corroborated manipulation detection: both specialists detect synthetic/manipulation signals
                raw_fake_prob = round(float(0.60 * max(xcep_fake_p, vit_fake_p) + 0.40 * min(xcep_fake_p, vit_fake_p)), 4)
                is_conflict = False
            elif both_real:
                # Both models detect authentic features: authentic consensus
                raw_fake_prob = round(float(0.60 * min(xcep_fake_p, vit_fake_p) + 0.40 * max(xcep_fake_p, vit_fake_p)), 4)
                is_conflict = False
            elif xcep_fake_p >= 0.90:
                # Decisive Face-Swap / Blending Boundary Specialist (FaceForensics / deepfake splicing)
                raw_fake_prob = round(float(0.85 * xcep_fake_p + 0.15 * vit_fake_p), 4)
                is_conflict = False
            elif vit_fake_p >= 0.90 and (xcep_fake_p >= 0.50 or r_val >= 0.15):
                # Corroborated Synthetic Portrait Specialist: ViT detects generative synthesis textures
                raw_fake_prob = round(float(0.75 * vit_fake_p + 0.15 * xcep_fake_p + 0.10 * r_val), 4)
                is_conflict = False
            elif vit_fake_p < 0.15 and r_val < 0.38 and xcep_fake_p < 0.75:
                # Still-image portrait domain resolution: ViT and ResNet corroborate genuine portrait;
                # Xception's moderate elevation is a known out-of-domain compression artifact on high-res stills.
                raw_fake_prob = round(float(0.50 * vit_fake_p + 0.30 * r_val + 0.20 * xcep_fake_p), 4)
                is_conflict = False
            elif xcep_fake_p < 0.25 and r_val < 0.35 and vit_fake_p >= 0.50:
                # Video-frame domain resolution: Xception and ResNet corroborate authentic video frame;
                # ViT's elevation is a known sensitivity to video block noise.
                raw_fake_prob = round(float(0.50 * xcep_fake_p + 0.30 * r_val + 0.20 * vit_fake_p), 4)
                is_conflict = False
            elif diff <= 0.25:
                # Boundary proximity: models near threshold; not an irreconcilable conflict
                raw_fake_prob = round(float(0.50 * xcep_fake_p + 0.50 * vit_fake_p), 4)
                is_conflict = False
            else:
                # Material unresolvable disagreement between specialist detectors
                raw_fake_prob = round(float(0.50 * xcep_fake_p + 0.50 * vit_fake_p), 4)
                is_conflict = True
        elif has_xcep and has_resnet:
            # Xception + ResNet available (ViT missing/failed)
            diff = abs(xcep_fake_p - resnet_fake_p)
            if xcep_fake_p >= 0.90 and resnet_fake_p >= 0.20:
                raw_fake_prob = round(float(0.80 * xcep_fake_p + 0.20 * resnet_fake_p), 4)
                is_conflict = False
            elif (xcep_fake_p >= 0.50 and resnet_fake_p >= 0.50) or (xcep_fake_p < 0.50 and resnet_fake_p < 0.50):
                raw_fake_prob = round(float(0.70 * xcep_fake_p + 0.30 * resnet_fake_p), 4)
                is_conflict = False
            else:
                raw_fake_prob = round(float(0.60 * xcep_fake_p + 0.40 * resnet_fake_p), 4)
                is_conflict = (diff > 0.35)
        elif has_vit and has_resnet:
            # ViT + ResNet available (Xception missing/failed)
            diff = abs(vit_fake_p - resnet_fake_p)
            if vit_fake_p >= 0.90 and resnet_fake_p >= 0.20:
                raw_fake_prob = round(float(0.80 * vit_fake_p + 0.20 * resnet_fake_p), 4)
                is_conflict = False
            elif (vit_fake_p >= 0.50 and resnet_fake_p >= 0.50) or (vit_fake_p < 0.50 and resnet_fake_p < 0.50):
                raw_fake_prob = round(float(0.70 * vit_fake_p + 0.30 * resnet_fake_p), 4)
                is_conflict = False
            else:
                raw_fake_prob = round(float(0.60 * vit_fake_p + 0.40 * resnet_fake_p), 4)
                is_conflict = (diff > 0.35)
        elif has_xcep:
            raw_fake_prob = round(float(xcep_fake_p), 4)
            is_conflict = False
        elif has_vit:
            raw_fake_prob = round(float(vit_fake_p), 4)
            is_conflict = False
        else:
            raw_fake_prob = round(float(resnet_fake_p), 4)
            is_conflict = False

        # 5. Combined Multi-Signal Decision Fusion
        forensic_dict = dict(forensic_res)
        if is_conflict:
            forensic_dict["classifier_conflict"] = True
            forensic_dict["tampering_detected"] = (raw_fake_prob >= 0.60)

        decision = self.decision_system.decide(
            raw_fake_prob=raw_fake_prob,
            quality_result=quality_res,
            forensic_result=forensic_dict,
            visual_analysis=visual_res,
        )

        processing_time = round(time.time() - start_time, 4)
        evidence_cards = []
        if img_bgr is not None and face_bbox is not None:
            try:
                evidence_cards = self.forensics.generate_evidence_cards(
                    forensic_res, img_bgr, face_bbox=face_bbox, verdict=decision.verdict
                )
            except Exception as e:
                logger.warning(f"Failed to generate evidence cards: {e}")

        return self._format_response(
            decision=decision,
            quality_res=quality_res,
            forensic_res=forensic_res,
            visual_res=visual_res,
            filename=filename,
            size=size,
            fmt=fmt,
            processing_time=processing_time,
            img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0),
            face_bbox=face_bbox,
            evidence_cards=evidence_cards,
            raw_xcep_p=xcep_fake_p,
            raw_vit_p=vit_fake_p,
            raw_resnet_p=resnet_fake_p,
        )

    def _format_response(
        self,
        decision,
        quality_res,
        forensic_res: Dict[str, Any],
        visual_res: Dict[str, Any],
        filename: str,
        size: int,
        fmt: str,
        processing_time: float,
        img_shape: tuple,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        evidence_cards: Optional[List[Dict[str, Any]]] = None,
        raw_xcep_p: Optional[float] = None,
        raw_vit_p: Optional[float] = None,
        raw_resnet_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Formats the structured forensic response."""
        is_not_suitable = decision.prediction == DecisionState.NOT_SUITABLE
        fft_data = forensic_res.get("fft_analysis", {})
        ela_data = forensic_res.get("ela_analysis", {})
        edge_data = forensic_res.get("edge_analysis", {})

        # Construct evidence-driven factual summary
        evidence_summary = None
        if not is_not_suitable:
            f_p = decision.fake_probability if decision.fake_probability is not None else 0.50
            xc_str = f"Xception: {raw_xcep_p:.1%}" if raw_xcep_p is not None else "N/A"
            vt_str = f"ViT: {raw_vit_p:.1%}" if raw_vit_p is not None else "N/A"

            if decision.is_binary_fallback or decision.prediction == DecisionState.UNCERTAIN or decision.reliability_warning:
                evidence_summary = (
                    f"Facial region analysis detected divergent or ambiguous signals between neural classifiers ({xc_str}, {vt_str}). "
                    f"Classified as UNCERTAIN (uncertified). Provisional directional suggestion: {decision.primary_verdict} (calibrated probability: {f_p:.1%}). "
                    "Notice: Indicators lack consensus; authenticity has NOT been established, and secondary manual verification is required."
                )
            elif decision.primary_verdict in ("FAKE", "DEEPFAKE"):
                anomaly_cards = [c for c in (evidence_cards or []) if c.get("verdict_support") == "SUPPORTS_FAKE"]
                if anomaly_cards:
                    top_obs = anomaly_cards[0].get("observed_finding")
                    evidence_summary = (
                        f"Facial region analysis identified local forensic anomalies: {top_obs} "
                        f"Specialist classifiers measured a calibrated synthetic probability of {f_p:.1%}, "
                        f"corroborated by neural feature inspection ({xc_str}, {vt_str})."
                    )
                else:
                    evidence_summary = (
                        f"Specialist classifiers measured a calibrated synthetic probability of {f_p:.1%}, "
                        f"corroborated by neural feature inspection ({xc_str}, {vt_str}). "
                        "Local frequency and boundary measurements remained within nominal baselines."
                    )
            elif decision.primary_verdict in ("REAL", "LIKELY_REAL"):
                evidence_summary = (
                    f"Facial region analysis detected no significant manipulation artifacts across neural classifiers "
                    f"({1.0 - f_p:.1%} authentic probability, {xc_str}, {vt_str}). "
                    "Notice: Absence of detected manipulation does not guarantee absolute authenticity."
                )
            else:
                evidence_summary = (
                    f"Facial region analysis returned indeterminate forensic signals (calibrated probability {f_p:.1%}). "
                    "Indicators lack decisive consensus; secondary manual verification is recommended."
                )

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

        # Structured Model Inventory
        models_inventory = [
            {
                "name": "ResNet-18 (Baseline)",
                "task": "General Facial Deepfake Classification (Baseline)",
                "predicted_class": ("FAKE" if raw_resnet_p >= 0.50 else "REAL") if raw_resnet_p is not None else "UNAVAILABLE",
                "raw_score": round(raw_resnet_p, 4) if raw_resnet_p is not None else None,
                "calibrated_prob": None,
                "calibration_status": "Not calibrated; raw softmax",
                "status": "SUCCESS" if raw_resnet_p is not None else "FAILED",
                "limitations": "Standard CNN baseline classifier; does not model fine generative diffusion or spectral blending artifacts."
            },
            {
                "name": "XceptionNet (DeepfakeBench)",
                "task": "Face-Swap & Video Blending Boundary Detection",
                "predicted_class": ("FAKE" if raw_xcep_p >= 0.50 else "REAL") if raw_xcep_p is not None else "UNAVAILABLE",
                "raw_score": round(raw_xcep_p, 4) if raw_xcep_p is not None else None,
                "calibrated_prob": None,
                "calibration_status": "Not calibrated; raw softmax",
                "status": "SUCCESS" if raw_xcep_p is not None else "UNAVAILABLE",
                "limitations": "Trained predominantly on video compression artifacts and face-swap boundary blending; sensitive to JPEG compression artifacts."
            },
            {
                "name": "Vision Transformer (ViT-B/16)",
                "task": "Synthetic & AI-Generated Portrait Classifier",
                "predicted_class": ("FAKE" if raw_vit_p >= 0.50 else "REAL") if raw_vit_p is not None else "UNAVAILABLE",
                "raw_score": round(raw_vit_p, 4) if raw_vit_p is not None else None,
                "calibrated_prob": None,
                "calibration_status": "Not calibrated; raw softmax",
                "status": "SUCCESS" if raw_vit_p is not None else "UNAVAILABLE",
                "limitations": "Trained on synthetic portrait generation artifacts; sensitive to crop alignment and out-of-distribution lighting."
            }
        ]

        # Structured Measurements
        res_w = img_shape[1] if len(img_shape) > 1 else (quality_res.resolution[0] if quality_res.resolution else 0)
        res_h = img_shape[0] if len(img_shape) > 0 else (quality_res.resolution[1] if quality_res.resolution else 0)
        measurements = {
            "resolution": {
                "width": res_w,
                "height": res_h,
                "formatted": f"{res_w} × {res_h} px",
                "status": "Pass: Native resolution meets minimum analysis threshold (100×100 px)" if (res_w >= 100 and res_h >= 100) else "Caution: Low input resolution",
                "interpretation": "Spatial input dimensions. Verified against minimum analysis threshold (100×100 px)."
            },
            "blur": {
                "value": round(quality_res.blur_score, 2),
                "method": "Laplacian variance",
                "status": "Pass: Crisp high-frequency focus" if quality_res.blur_score >= 40.0 else "Caution: Low optical sharpness or motion blur",
                "interpretation": "Focus & high-frequency sharpness measurement. A score above 40 indicates adequate focus; this is a quality gate requirement, NOT evidence of authenticity."
            },
            "ela": {
                "value": round(ela_data.get("ela_score", 0.10), 3),
                "method": "Error Level Analysis",
                "status": "Elevated local compression variance" if ela_data.get("ela_score", 0.10) > 0.35 else "Uniform compression signature",
                "interpretation": "Exploratory JPEG recompression error disparity measurement. Subject to resaving artifacts; does NOT constitute definitive proof of tampering."
            },
            "face_detection": {
                "detected": quality_res.face_detected,
                "face_count": quality_res.face_count,
                "bbox": list(face_bbox) if face_bbox else None,
                "interpretation": "Face detection is a preprocessing step to extract the facial region for model inference. Finding a face is NOT evidence of authenticity or manipulation."
            }
        }

        # Supporting, Contradicting, Unavailable, Limitations
        supporting_evidence = []
        contradicting_evidence = []
        unavailable_checks = [
            "Camera sensor PRNU fingerprint analysis: Unavailable",
            "Audio-visual lip synchronization: Unavailable (Static image)",
            "C2PA cryptographic provenance metadata: Not present"
        ]
        limitations = [
            "Absence of detected manipulation does not guarantee absolute authenticity.",
            "Face-swap models trained on video compression (Xception) can produce false alarms on high-frequency web JPEG compression.",
            "Error Level Analysis and Laplacian sharpness are exploratory quality diagnostics, not definitive forensic proof."
        ]

        if canonical_v == DecisionState.UNCERTAIN:
            if raw_xcep_p is not None and raw_vit_p is not None:
                if (raw_xcep_p >= 0.50) != (raw_vit_p >= 0.50):
                    contradicting_evidence.append(
                        f"Specialist classifier disagreement: Xception (face-swap detector) predicted {('FAKE (' + str(round(raw_xcep_p*100, 1)) + '%)') if raw_xcep_p >= 0.5 else ('REAL (' + str(round((1-raw_xcep_p)*100, 1)) + '%)')} whereas ViT (generative synthetic detector) predicted {('FAKE (' + str(round(raw_vit_p*100, 1)) + '%)') if raw_vit_p >= 0.5 else ('REAL (' + str(round((1-raw_vit_p)*100, 1)) + '%)')}."
                    )
            supporting_evidence.append("Quality gate verification completed; facial region extracted.")
        elif canonical_v in (DecisionState.FAKE, "FAKE"):
            if raw_vit_p is not None and raw_vit_p >= 0.50:
                supporting_evidence.append(f"Vision Transformer measured synthetic generative artifacts ({raw_vit_p*100:.1f}% fake probability).")
            if raw_xcep_p is not None and raw_xcep_p >= 0.50:
                supporting_evidence.append(f"Xception detector measured boundary blending anomalies ({raw_xcep_p*100:.1f}% fake probability).")
            if raw_resnet_p is not None and raw_resnet_p < 0.50:
                contradicting_evidence.append(f"ResNet baseline predicted lower probability ({raw_resnet_p*100:.1f}%).")
        elif canonical_v in (DecisionState.REAL, "REAL"):
            if raw_vit_p is not None and raw_vit_p < 0.50:
                supporting_evidence.append(f"Vision Transformer observed authentic generative feature distribution ({(1-raw_vit_p)*100:.1f}% authentic).")
            if raw_xcep_p is not None and raw_xcep_p < 0.50:
                supporting_evidence.append(f"Xception detector detected no boundary blending anomalies ({(1-raw_xcep_p)*100:.1f}% authentic).")
        elif canonical_v == DecisionState.NOT_SUITABLE:
            supporting_evidence.append("Media suitability verification: No human face detected.")
            unavailable_checks.append("All facial neural classifiers skipped due to absence of facial target.")

        raw_fused_score = decision.raw_model_score
        temperature_scaled_score = (
            self.decision_system.calibrator.calibrate_probability(raw_fused_score)
            if raw_fused_score is not None
            else None
        )
        score_type = (
            "temperature_scaled_score"
            if decision.fake_probability is not None
            and temperature_scaled_score is not None
            and abs(decision.fake_probability - temperature_scaled_score) < 1e-4
            else "decision_adjusted_score"
        )
        decision_obj = {
            "method": "SafetyDecisionSystem with specialist score fusion",
            "score": round(decision.fake_probability, 4) if decision.fake_probability is not None else None,
            "score_type": score_type,
            "probability_is_calibrated": False,
            "raw_fused_score": round(raw_fused_score, 4) if raw_fused_score is not None else None,
            "temperature_scaled_score": round(temperature_scaled_score, 4) if temperature_scaled_score is not None else None,
            "temperature": self.decision_system.calibrator.temperature,
            "calibration_note": "Temperature scaling is applied, but its validation fit is not bound to the active model checkpoint or fusion version.",
            "confidence": conf_out,
            "confidence_type": "Uncertified" if canonical_v == DecisionState.UNCERTAIN else ("Not performed" if canonical_v == DecisionState.NOT_SUITABLE else "Model score; not calibrated probability"),
            "criteria_met": bool(conf_out is not None and canonical_v in (DecisionState.REAL, DecisionState.FAKE)),
            "reason": decision.reason,
        }

        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "width": res_w,
                "height": res_h,
                "format": fmt,
            },
            "media_type": "image",
            "task_type": "ai_generated_and_facial_deepfake_detection",
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
            "reliability_warning": decision.reliability_warning,
            "confidence": conf_out,
            "confidence_score": conf_out,
            "confidence_level": conf_level_out,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": decision.raw_model_score,
            "uncertainty": decision.uncertainty,
            "evidence": decision.evidence,
            "influencing_evidence": getattr(decision, "influencing_evidence", []),
            "providers_ran": getattr(decision, "providers_ran", []),
            "approx_cost_usd": getattr(decision, "approx_cost_usd", 0.0),
            "reason": decision.reason,
            "summary": evidence_summary or decision.user_facing_statement,
            "user_facing_statement": evidence_summary or decision.user_facing_statement,
            "recommendation": getattr(decision, "recommendation", None),
            "analysis_status": getattr(decision, "analysis_status", "Completed"),
            "face_detected": quality_res.face_detected,
            "face_bbox": list(face_bbox) if face_bbox is not None else None,
            "evidence_cards": evidence_cards or [],
            "forensic_evidence": forensic_res,
            "media_category": getattr(decision, "media_category", "Facial Media"),
            "quality_gate": quality_res.to_dict(),
            "visual_analysis": visual_res,
            "models": models_inventory,
            "measurements": measurements,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "unavailable_checks": unavailable_checks,
            "decision": decision_obj,
            "limitations": limitations,
            "features": {
                "noise_inconsistency": 0.0 if is_not_suitable else forensic_res.get("composite_forensic_score", 0.15),
                "edge_boundary_artifacts": 0.0 if is_not_suitable else edge_data.get("edge_score", 0.10),
                "fft_spectral_anomaly": 0.0 if is_not_suitable else fft_data.get("fft_score", 0.10),
                "ela_compression_anomaly": 0.0 if is_not_suitable else ela_data.get("ela_score", 0.10),
                "metadata_anomaly": False,
                "blur_score": quality_res.blur_score,
                "raw_xcep_score": raw_xcep_p,
                "raw_vit_score": raw_vit_p,
                "raw_resnet_score": raw_resnet_p,
            },
            "processing_time": processing_time,
        }

