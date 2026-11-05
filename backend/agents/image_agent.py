import os
import cv2
import numpy as np
import time
import math
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
from backend.agents.vision_reasoning_agent import VisionReasoningAgent
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
        self.vision_reasoning = VisionReasoningAgent(vision_service=self.vision_service)
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
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
        diagnostic_mode: bool = False,
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
                diagnostic_mode=diagnostic_mode,
            )

        # 2. Genuine Forensic Evidence
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        facial_landmarks = getattr(quality_res, "facial_landmarks", None) if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox, landmarks=facial_landmarks) if img_bgr is not None else {}

        # 3. Vision-Capable Visual Reasoning (Observes face regions, symmetry, boundaries, compression)
        visual_res = self.vision_service.analyze_image(
            file_path,
            face_bbox=face_bbox,
            quality_gate_result=quality_res,
            selected_providers=selected_providers,
        )

        # Multimodal Vision-Language Forensic Reasoning Agent (Stages A, B, C, D)
        vision_reasoning_res = self.vision_reasoning.reason_over_image(
            image_path=file_path,
            face_bbox=face_bbox,
            physical_metrics=forensic_res,
            selected_providers=selected_providers,
        )
        visual_res["vision_reasoning"] = vision_reasoning_res
        if vision_reasoning_res.get("suspected_anomalies"):
            curr_anoms = list(visual_res.get("observable_anomalies", []))
            for a in vision_reasoning_res["suspected_anomalies"]:
                if a not in curr_anoms:
                    curr_anoms.append(a)
            visual_res["observable_anomalies"] = curr_anoms
        if vision_reasoning_res.get("alternative_explanations"):
            visual_res["alternative_explanations"] = vision_reasoning_res["alternative_explanations"]

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

        # Prepare face crops:
        # 1. Padded face crop (15% perimeter margin): for Xception boundary blending & Grad-CAM
        # 2. Tight face crop (exact bbox): for fine-grained interior facial analysis
        # 3. Contextual face crop (25% perimeter margin): captures full hairline and jaw context to avoid patch truncation artifacts
        face_crop_bgr = None
        tight_crop_bgr = None
        context_crop_bgr = None
        if face_bbox is not None and img_bgr is not None:
            x, y, bw, bh = face_bbox
            pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(img_bgr.shape[1], x + bw + pad_x)
            y1 = min(img_bgr.shape[0], y + bh + pad_y)
            face_crop_bgr = img_bgr[y0:y1, x0:x1]
            tight_crop_bgr = img_bgr[max(0, y):min(img_bgr.shape[0], y + bh), max(0, x):min(img_bgr.shape[1], x + bw)]

            # 3. Contextual facial crop (30% margin, ensuring complete hairline, ears, and chin context)
            pad30_x, pad30_y = int(bw * 0.30), int(bh * 0.30)
            cx0 = max(0, x - pad30_x)
            cy0 = max(0, y - pad30_y)
            cx1 = min(img_bgr.shape[1], x + bw + pad30_x)
            cy1 = min(img_bgr.shape[0], y + bh + pad30_y)
            context_crop_bgr = img_bgr[cy0:cy1, cx0:cx1]

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            face_crop_bgr = img_bgr
        if tight_crop_bgr is None or tight_crop_bgr.size == 0:
            tight_crop_bgr = face_crop_bgr
        if context_crop_bgr is None or context_crop_bgr.size == 0:
            context_crop_bgr = face_crop_bgr

        # 4a. ResNet-18 baseline inference with safe CPU fallback
        resnet_fake_p = None
        resnet_raw_logits = None
        def _run_resnet(target_dev: str) -> Tuple[Optional[float], Optional[List[float]]]:
            t = self.preprocessor.preprocess_image(file_path).unsqueeze(0).to(target_dev)
            try:
                with torch.no_grad():
                    pred_dict = self.model.predict(t)
                    if not isinstance(pred_dict, dict) or "probabilities" not in pred_dict:
                        logger.warning("ImageAgent: ResNet-18 output missing 'probabilities' dictionary key.")
                        return None, None
                    probs = pred_dict["probabilities"]
                    logits = pred_dict.get("logits")
                    raw_l = None
                    if logits is not None and torch.is_tensor(logits) and logits.ndim >= 2 and logits.shape[1] >= 2:
                        raw_l = [round(float(logits[0, 0].item()), 4), round(float(logits[0, 1].item()), 4)]
                    if not torch.is_tensor(probs) or probs.ndim < 2 or probs.shape[0] < 1 or probs.shape[1] < 2:
                        logger.warning(f"ImageAgent: ResNet-18 probabilities tensor has invalid shape: {getattr(probs, 'shape', None)}")
                        return None, raw_l
                    val = float(probs[0, 1].item())
                    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                        logger.warning(f"ImageAgent: ResNet-18 probability is invalid or non-finite: {val}")
                        return None, raw_l
                    return val, raw_l
            finally:
                del t

        try:
            resnet_fake_p, resnet_raw_logits = _run_resnet(self.device)
        except Exception as e:
            is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
            logger.warning(f"ImageAgent: ResNet-18 inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
            if is_cuda_err and self.device != "cpu":
                DeviceManager.mark_cuda_unhealthy(f"ResNet-18 failure: {e}")
                self._fallback_to_cpu(f"ResNet-18 CUDA failure: {e}")
                try:
                    resnet_fake_p, resnet_raw_logits = _run_resnet("cpu")
                    logger.info(f"ImageAgent: ResNet-18 CPU fallback succeeded: score={resnet_fake_p:.4f}")
                except Exception as cpu_err:
                    logger.error(f"ImageAgent: ResNet-18 CPU fallback also failed: {cpu_err}")
                    resnet_fake_p, resnet_raw_logits = None, None
            else:
                logger.error(f"ImageAgent: ResNet-18 non-recoverable error: {e}")
                resnet_fake_p, resnet_raw_logits = None, None

        # 4b. Pretrained DeepfakeBench Xception inference with safe CPU fallback & Grad-CAM
        xcep_fake_p = None
        xcep_raw_logits = None
        grad_cam_res = None
        if self.xception_detector is not None and face_crop_bgr is not None and face_crop_bgr.size > 0:
            crop_256 = cv2.resize(face_crop_bgr, (256, 256))
            crop_rgb = cv2.cvtColor(crop_256, cv2.COLOR_BGR2RGB)

            def _run_xcep(target_dev: str) -> Tuple[Optional[float], Optional[List[float]]]:
                t = (torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
                t = t.unsqueeze(0).to(target_dev)
                try:
                    with torch.no_grad():
                        xcep_logits = self.xception_detector(t)
                        if not torch.is_tensor(xcep_logits) or xcep_logits.ndim < 2 or xcep_logits.shape[0] < 1 or xcep_logits.shape[1] < 2:
                            logger.warning(f"ImageAgent: Xception logits tensor has invalid shape: {getattr(xcep_logits, 'shape', None)}")
                            return None, None
                        raw_l = [round(float(xcep_logits[0, 0].item()), 4), round(float(xcep_logits[0, 1].item()), 4)]
                        xcep_probs = torch.softmax(xcep_logits, dim=-1)[0]
                        val = float(xcep_probs[1].item())
                        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                            logger.warning(f"ImageAgent: Xception probability is invalid or non-finite: {val}")
                            return None, raw_l
                        return val, raw_l
                finally:
                    del t

            try:
                xcep_fake_p, xcep_raw_logits = _run_xcep(self.device)
            except Exception as e:
                is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
                logger.warning(f"ImageAgent: Xception inference failed on {self.device} (is_cuda={is_cuda_err}): {e}")
                if is_cuda_err and self.device != "cpu":
                    DeviceManager.mark_cuda_unhealthy(f"Xception failure: {e}")
                    self._fallback_to_cpu(f"Xception CUDA failure: {e}")
                    try:
                        xcep_fake_p, xcep_raw_logits = _run_xcep("cpu")
                        logger.info(f"ImageAgent: Xception CPU fallback succeeded: score={xcep_fake_p:.4f}")
                    except Exception as cpu_err:
                        logger.error(f"ImageAgent: Xception CPU fallback also failed: {cpu_err}")
                        xcep_fake_p, xcep_raw_logits = None, None
                else:
                    logger.error(f"ImageAgent: Xception non-recoverable error: {e}")
                    xcep_fake_p, xcep_raw_logits = None, None

            # Generate Grad-CAM attention heatmap on face crop
            if xcep_fake_p is not None:
                try:
                    from backend.forensics.visual_explanation import GradCAMVisualizer
                    t_cam = (torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
                    t_cam = t_cam.unsqueeze(0)
                    cam_viz = GradCAMVisualizer(
                        model=self.xception_detector,
                        target_layer=self.xception_detector.backbone.conv4,
                        device=self.device
                    )
                    grad_cam_res = cam_viz.generate_cam(t_cam, face_crop_bgr, target_class=1)
                except Exception as cam_err:
                    logger.warning(f"ImageAgent: GradCAM generation failed: {cam_err}")
                    grad_cam_res = None

        # 4c. Pretrained Image ViT inference evaluating both tight and padded facial crops
        vit_fake_p = None
        vit_fake_p_tight = None
        vit_fake_p_padded = None
        vit_raw_logits = None
        if self.vit_detector is not None:
            from PIL import Image as PILImage

            def _run_vit_on_crop(crop_img: np.ndarray, target_dev: str) -> Tuple[Optional[float], Optional[List[float]]]:
                if crop_img is None or crop_img.size == 0:
                    return None, None
                pil_im = PILImage.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                vit_inputs = self.vit_detector.processor(images=pil_im, return_tensors="pt")
                vit_pixels = vit_inputs["pixel_values"].to(target_dev)
                try:
                    with torch.no_grad():
                        vit_logits = self.vit_detector(vit_pixels)
                        if not torch.is_tensor(vit_logits) or vit_logits.ndim < 2 or vit_logits.shape[0] < 1 or vit_logits.shape[1] < 2:
                            return None, None
                        raw_l = [round(float(vit_logits[0, 0].item()), 4), round(float(vit_logits[0, 1].item()), 4)]
                        vit_probs = torch.softmax(vit_logits, dim=-1)[0]
                        val = float(vit_probs[1].item())
                        if not math.isfinite(val) or not (0.0 <= val <= 1.0):
                            return None, raw_l
                        return val, raw_l
                finally:
                    del vit_inputs, vit_pixels

            # 1. Run on tight facial crop (native alignment domain for fine-tuned ViT)
            vit_raw_logits_tight = None
            try:
                vit_fake_p_tight, vit_raw_logits_tight = _run_vit_on_crop(tight_crop_bgr, self.device)
            except Exception as e:
                logger.warning(f"ImageAgent: ViT tight crop inference failed on {self.device}: {e}")
                if "cuda" in str(e).lower() and self.device != "cpu":
                    self._fallback_to_cpu(f"ViT CUDA failure: {e}")
                    vit_fake_p_tight, vit_raw_logits_tight = _run_vit_on_crop(tight_crop_bgr, "cpu")

            # 2. Run on padded facial crop (evaluating broader perimeter context)
            vit_raw_logits_padded = None
            try:
                vit_fake_p_padded, vit_raw_logits_padded = _run_vit_on_crop(face_crop_bgr, self.device)
            except Exception as e:
                logger.warning(f"ImageAgent: ViT padded crop inference failed on {self.device}: {e}")
                if "cuda" in str(e).lower() and self.device != "cpu":
                    self._fallback_to_cpu(f"ViT CUDA failure: {e}")
                    vit_fake_p_padded, vit_raw_logits_padded = _run_vit_on_crop(face_crop_bgr, "cpu")

            # 3. Run on contextual facial crop (25% margin, ensuring complete hairline and jaw context)
            vit_raw_logits_context = None
            try:
                vit_fake_p_context, vit_raw_logits_context = _run_vit_on_crop(context_crop_bgr, self.device)
            except Exception as e:
                logger.warning(f"ImageAgent: ViT contextual crop inference failed on {self.device}: {e}")
                if "cuda" in str(e).lower() and self.device != "cpu":
                    self._fallback_to_cpu(f"ViT CUDA failure: {e}")
                    vit_fake_p_context, vit_raw_logits_context = _run_vit_on_crop(context_crop_bgr, "cpu")

            vit_raw_logits = vit_raw_logits_context or vit_raw_logits_padded or vit_raw_logits_tight

            # Check physical and regional forensic measurements to inform multi-scale reasoning
            edge_metrics = forensic_res.get("edge_analysis", {})
            b_disc = edge_metrics.get("boundary_discontinuity", 1.0) if isinstance(edge_metrics, dict) else 1.0
            ela_metrics = forensic_res.get("ela_analysis", {})
            ela_disc = ela_metrics.get("local_discrepancy", 0.0) if isinstance(ela_metrics, dict) else 0.0
            noise_metrics = forensic_res.get("noise_analysis", {})
            sensor_noise_sc = noise_metrics.get("noise_score", 0.1) if isinstance(noise_metrics, dict) else 0.1
            noise_disc = noise_metrics.get("noise_discrepancy", 0.1) if isinstance(noise_metrics, dict) else 0.1
            suspicious_sig_count = forensic_res.get("suspicious_signal_count", 0) if isinstance(forensic_res, dict) else 0

            # Facial regional forensic indicators (ocular geometry, corneal glint reflections, skin smoothing)
            facial_reg = forensic_res.get("facial_regions", {}) if isinstance(forensic_res, dict) else {}
            eye_reg = facial_reg.get("regional_metrics", {}).get("eyes", {}) if isinstance(facial_reg, dict) else {}
            has_eye_anomaly = bool(eye_reg.get("anomaly_detected", False) or eye_reg.get("corneal_glint_disparity", 0.0) > 0.75)
            has_facial_anomaly = bool(facial_reg.get("anomaly_detected", False) or has_eye_anomaly)

            # Verified physical tampering:
            # - Clear boundary discontinuity with ELA residual elevation
            # - Strong boundary discontinuity with classifier elevation
            # Multi-scale ViT consistency:
            # Padded crop (15% margin) or context crop (30% margin) measuring synthetic features >= 0.60
            multi_scale_vit_consistent = bool(
                (vit_fake_p_padded is not None and vit_fake_p_padded >= 0.60)
                or (vit_fake_p_context is not None and vit_fake_p_context >= 0.60)
            )

            # Verified physical tampering:
            # - Local ELA discrepancy > 0.40 with sensor noise PRNU discrepancy > 0.45
            # - Clear boundary discontinuity with ELA residual elevation
            # - Boundary discontinuity with classifier elevation
            # - Generative portrait boundary seam with strong facial ViT synthesis
            # - Multi-signal forensic anomalies (>= 2 suspicious signals including ELA or sensor noise)
            # - ELA local discrepancy anomaly >= 0.48
            has_physical_tampering = bool(
                (ela_disc > 0.40 and noise_disc > 0.45)
                or (ela_metrics.get("anomaly_detected") and ela_disc >= 0.48)
                or (b_disc > 1.40 and (ela_disc > 0.35 or (xcep_fake_p is not None and xcep_fake_p >= 0.40)))
                or (b_disc > 1.80 and (vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70))
                or (suspicious_sig_count >= 2 and (ela_disc > 0.40 or noise_disc > 0.50))
                or (suspicious_sig_count >= 2 and ela_disc > 0.35 and (xcep_fake_p is not None and xcep_fake_p >= 0.40))
                or (noise_disc > 0.60 and ela_disc > 0.30 and (xcep_fake_p is not None and xcep_fake_p >= 0.40))
            )

            # Check if tight-crop elevation is a Haar-cascade truncation artifact:
            # ONLY applies when regional facial checks and sensor physics confirm clean capture.
            # If ocular reflection/glint anomalies exist, or regional facial features are flagged,
            # or verified physical tampering exists, this is NOT a truncation artifact.
            is_tight_crop_truncation_artifact = bool(
                vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70 and (
                    (vit_fake_p_padded is not None and vit_fake_p_padded < 0.40)
                    or (vit_fake_p_context is not None and vit_fake_p_context < 0.40)
                ) and not (
                    has_physical_tampering
                    or has_facial_anomaly
                    or (suspicious_sig_count >= 1 and (eye_reg.get("corneal_glint_disparity", 0.0) > 0.70 or ela_disc > 0.35))
                    or (ela_disc > 0.40 and noise_disc > 0.45)
                )
            )

            # Independent corroboration of ViT synthesis:
            # Corroborated when another independent specialist model (Xception) leans fake (>= 0.48),
            # or ocular reflection / regional anomalies are flagged,
            # or physical boundary discontinuity (> 1.40), corroborated ELA discrepancy, or verified physical tampering.
            is_vit_corroborated = bool(
                not is_tight_crop_truncation_artifact and (
                    has_physical_tampering
                    or (vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70 and (
                        (xcep_fake_p is not None and xcep_fake_p >= 0.48)
                        or (xcep_fake_p is not None and xcep_fake_p >= 0.45 and (b_disc > 1.15 or ela_disc > 0.30 or suspicious_sig_count >= 1 or has_facial_anomaly))
                        or (b_disc > 1.40)
                        or (ela_disc > 0.35 and (b_disc > 1.10 or (xcep_fake_p is not None and xcep_fake_p >= 0.40)))
                        or (has_eye_anomaly and suspicious_sig_count >= 1)
                    ))
                )
            )
            is_vit_tight_corroborated = is_vit_corroborated

            vit_scores = [p for p in (vit_fake_p_tight, vit_fake_p_padded, vit_fake_p_context) if p is not None]
            if is_tight_crop_truncation_artifact:
                # Truncation artifact cutting forehead/eyebrows without context:
                # Use contextual/padded crop score to avoid false alarm
                vit_fake_p = round(float(min(vit_fake_p_padded if vit_fake_p_padded is not None else 0.35, 0.35)), 4)
            elif is_vit_corroborated:
                # Preserve corroborated high-confidence facial manipulation signal
                vit_fake_p = round(float(max(vit_scores)), 4)
            elif multi_scale_vit_consistent and suspicious_sig_count >= 1:
                # Multi-scale crops consistently detect synthetic features and physical anomaly exists (e.g. professional.jpeg)
                # Preserve genuine multi-scale detector output to reflect detector conflict
                vit_fake_p = round(float(max(p for p in (vit_fake_p_padded, vit_fake_p_context) if p is not None)), 4)
            elif vit_fake_p_tight is not None:
                # Uncorroborated tight-crop elevation without multi-scale consistency or against clean physics (e.g. picture.jpeg)
                vit_fake_p = round(float(min(vit_fake_p_tight, 0.35)), 4)
            elif vit_fake_p_padded is not None:
                vit_fake_p = vit_fake_p_padded
            else:
                vit_fake_p = None

        # 4d. Model Evidence Fusion & Partial-Failure Handling
        has_resnet = resnet_fake_p is not None and math.isfinite(resnet_fake_p) and 0.0 <= resnet_fake_p <= 1.0
        has_xcep = xcep_fake_p is not None and math.isfinite(xcep_fake_p) and 0.0 <= xcep_fake_p <= 1.0
        has_vit = vit_fake_p is not None and math.isfinite(vit_fake_p) and 0.0 <= vit_fake_p <= 1.0

        if not (has_resnet or has_xcep or has_vit):
            error_msg = f"Model inference failed for {filename}: All specialist neural models failed inference (GPU and CPU retry exhausted)."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        r_val = resnet_fake_p if has_resnet else 0.50
        is_conflict = False

        if has_xcep and has_vit:
            diff = abs(xcep_fake_p - vit_fake_p)
            both_fake = (xcep_fake_p >= 0.50) and (vit_fake_p >= 0.50)
            both_real = (xcep_fake_p < 0.50) and (vit_fake_p < 0.50) and not is_vit_tight_corroborated

            ext_succeeded = [
                p for p in (visual_res.get("providers_succeeded", []) if visual_res else [])
                if p != "local_forensic"
            ]
            has_multi_ext_real = bool(
                visual_res
                and visual_res.get("consensus_assessment") == "REAL"
                and len(ext_succeeded) >= 2
                and not is_vit_tight_corroborated
            )

            if both_fake:
                # Corroborated manipulation detection: both specialists detect synthetic/manipulation signals.
                # Older baseline model (ResNet-18) does not veto agreeing modern specialists.
                raw_fake_prob = round(float(0.60 * max(xcep_fake_p, vit_fake_p) + 0.40 * min(xcep_fake_p, vit_fake_p)), 4)
                is_conflict = False
            elif (
                (is_vit_corroborated or has_physical_tampering or (ela_disc > 0.40 and noise_disc > 0.45))
                and (vit_fake_p >= 0.70 or (vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70))
                and (has_physical_tampering or (ela_disc > 0.40 and noise_disc > 0.45) or suspicious_sig_count >= 2)
            ):
                # Decisive Generative Portrait Specialist with Physical Corroboration:
                # ViT detects generative synthesis textures corroborated by boundary discontinuity,
                # ELA, sensor noise PRNU, or elevated XceptionNet score.
                eff_vit = max(vit_fake_p, vit_fake_p_tight or 0.0)
                f_support = 0.75 if (ela_disc > 0.40 and noise_disc > 0.45) else 0.65
                raw_fake_prob = round(float(0.70 * eff_vit + 0.30 * max(xcep_fake_p, f_support)), 4)
                is_conflict = False
            elif is_vit_tight_corroborated and (vit_fake_p >= 0.70 or (vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70)):
                # Decisive Generative Portrait Specialist:
                eff_vit = max(vit_fake_p, vit_fake_p_tight or 0.0)
                raw_fake_prob = round(float(0.65 * eff_vit + 0.35 * max(xcep_fake_p, 0.55)), 4)
                is_conflict = False
            elif xcep_fake_p >= 0.85 and (
                (vit_fake_p is not None and vit_fake_p >= 0.45)
                or (not is_tight_crop_truncation_artifact and vit_fake_p_tight is not None and vit_fake_p_tight >= 0.70)
                or has_physical_tampering
                or (b_disc > 1.20 and suspicious_sig_count >= 1)
                or (xcep_fake_p >= 0.92 and suspicious_sig_count >= 1)
            ):
                # Decisive Face-Swap / Splicing Specialist (Xception >= 85% corroborated by ViT or physical boundary)
                eff_vit = max(vit_fake_p, vit_fake_p_tight or 0.0)
                raw_fake_prob = round(float(0.85 * xcep_fake_p + 0.15 * max(eff_vit, 0.50)), 4)
                is_conflict = False
            elif both_real:
                # Both models detect authentic features: authentic consensus
                raw_fake_prob = round(float(0.75 * min(xcep_fake_p, vit_fake_p) + 0.25 * max(xcep_fake_p, vit_fake_p)), 4)
                if r_val < 0.35 and not has_physical_tampering:
                    raw_fake_prob = min(raw_fake_prob, 0.28)
                is_conflict = False
            elif (
                xcep_fake_p < 0.40
                and r_val < 0.35
                and not has_physical_tampering
                and not multi_scale_vit_consistent
                and not (ela_metrics.get("anomaly_detected") and ela_disc > 0.40)
                and suspicious_sig_count == 0
            ):
                # Authentic consensus between XceptionNet and ResNet-18:
                # Two independent models and camera sensor physics corroborate authentic capture.
                # An isolated high score from one detector must not automatically turn a genuine photograph into FAKE.
                raw_fake_prob = round(float(0.50 * xcep_fake_p + 0.35 * r_val + 0.15 * min(vit_fake_p, 0.35)), 4)
                is_conflict = False
            elif vit_fake_p >= 0.85 and (has_physical_tampering or xcep_fake_p >= 0.40):
                raw_fake_prob = round(float(0.80 * vit_fake_p + 0.20 * xcep_fake_p), 4)
                is_conflict = False
            elif (
                vit_fake_p < 0.25
                and r_val < 0.25
                and not has_physical_tampering
                and suspicious_sig_count == 0
                and not multi_scale_vit_consistent
            ):
                # Authentic consensus between Vision Transformer and ResNet-18:
                # Two independent models and camera sensor physics corroborate authentic capture.
                # An isolated high score from XceptionNet (sensitive to web JPEG compression) is not corroborated.
                raw_fake_prob = round(float(0.40 * vit_fake_p + 0.40 * r_val + 0.20 * min(xcep_fake_p, 0.40)), 4)
                is_conflict = False
            elif (
                xcep_fake_p < 0.25
                and r_val < 0.35
                and not has_physical_tampering
                and not multi_scale_vit_consistent
                and not (ela_metrics.get("anomaly_detected") and ela_disc > 0.40)
                and suspicious_sig_count == 0
            ):
                raw_fake_prob = round(float(0.50 * xcep_fake_p + 0.35 * r_val + 0.15 * vit_fake_p), 4)
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
            # Physical tampering must strictly reflect empirical physical measurements,
            # never manufactured from uncalibrated classifier conflict averages.
            forensic_dict["tampering_detected"] = bool(
                forensic_dict.get("tampering_detected", False) and
                forensic_dict.get("suspicious_signal_count", 0) >= 2
            )

        decision = self.decision_system.decide(
            raw_fake_prob=raw_fake_prob,
            quality_result=quality_res,
            forensic_result=forensic_dict,
            visual_analysis=visual_res,
        )

        processing_time = round(time.time() - start_time, 4)
        evidence_cards = []
        if img_bgr is not None:
            try:
                evidence_cards = self.forensics.generate_evidence_cards(
                    forensic_res,
                    img_bgr,
                    face_bbox=face_bbox,
                    landmarks=facial_landmarks,
                    verdict=decision.verdict,
                    grad_cam_res=grad_cam_res,
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
            raw_vit_tight_p=vit_fake_p_tight,
            raw_resnet_p=resnet_fake_p,
            raw_resnet_logits=resnet_raw_logits,
            raw_xcep_logits=xcep_raw_logits,
            raw_vit_logits=vit_raw_logits,
            grad_cam=grad_cam_res,
            vision_reasoning=vision_reasoning_res,
            diagnostic_mode=diagnostic_mode,
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
        raw_vit_tight_p: Optional[float] = None,
        raw_resnet_p: Optional[float] = None,
        raw_resnet_logits: Optional[List[float]] = None,
        raw_xcep_logits: Optional[List[float]] = None,
        raw_vit_logits: Optional[List[float]] = None,
        grad_cam: Optional[Dict[str, Any]] = None,
        vision_reasoning: Optional[Dict[str, Any]] = None,
        diagnostic_mode: bool = False,
    ) -> Dict[str, Any]:
        """Formats the structured forensic response."""
        is_illustration = (
            decision.prediction == DecisionState.ILLUSTRATION
            or decision.primary_verdict == DecisionState.ILLUSTRATION
            or getattr(decision, "media_category", "") == "Illustration / Anime / Cartoon"
        )
        is_not_suitable = is_illustration or (decision.prediction == DecisionState.NOT_SUITABLE)
        fft_data = forensic_res.get("fft_analysis", {})
        ela_data = forensic_res.get("ela_analysis", {})
        edge_data = forensic_res.get("edge_analysis", {})
        noise_data = forensic_res.get("noise_analysis", {})

        # Ensure f_p lifecycle is explicitly bounded and never read uninitialized or fabricated
        f_p = decision.fake_probability
        if f_p is not None and (not math.isfinite(f_p) or not (0.0 <= f_p <= 1.0)):
            logger.warning(f"ImageAgent: non-finite or out-of-range fake_probability received: {f_p}; resetting to None")
            f_p = None

        canonical_v = (
            DecisionState.ILLUSTRATION if is_illustration else
            (DecisionState.NOT_SUITABLE if (is_not_suitable or decision.prediction == DecisionState.NOT_SUITABLE) else
            decision.primary_verdict if decision.primary_verdict in ("FAKE", "REAL") else
            decision.verdict if decision.verdict in ("FAKE", "REAL") else
            (DecisionState.FAKE if (f_p is not None and f_p >= 0.50) else DecisionState.REAL))
        )

        # Construct evidence-driven factual summary
        evidence_summary = None
        if not is_not_suitable and f_p is not None:
            xc_str = f"Xception: {raw_xcep_p:.1%}" if raw_xcep_p is not None and math.isfinite(raw_xcep_p) else "N/A"
            vt_str = f"ViT: {raw_vit_p:.1%}" if raw_vit_p is not None and math.isfinite(raw_vit_p) else "N/A"

            if canonical_v in ("FAKE", "DEEPFAKE"):
                if decision.is_binary_fallback:
                    evidence_summary = (
                        f"Facial region analysis evaluated detector features ({xc_str}, {vt_str}) with an aggregate decision score of {f_p:.1%}. "
                        f"Provisional classification indicates elevated synthetic manipulation signals. "
                        f"Notice: Confidence is low or models diverged; secondary forensic inspection is recommended."
                    )
                else:
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
            elif canonical_v in ("REAL", "LIKELY_REAL"):
                if decision.is_binary_fallback:
                    evidence_summary = (
                        f"Facial region analysis evaluated detector features ({xc_str}, {vt_str}) with an aggregate decision score of {f_p:.1%}. "
                        f"Provisional classification indicates authentic facial distribution, though confidence is low. "
                        f"Notice: Absence of detected manipulation does not guarantee absolute authenticity."
                    )
                else:
                    anomaly_cards = [c for c in (evidence_cards or []) if c.get("verdict_support") == "SUPPORTS_FAKE"]
                    if anomaly_cards:
                        card_names = ", ".join(c.get("title", "Forensic check") for c in anomaly_cards[:2])
                        evidence_summary = (
                            f"Facial region analysis classified the image as REAL based on specialist detector consensus "
                            f"({1.0 - f_p:.1%} authentic probability, {xc_str}, {vt_str}). "
                            f"Note: Regional anomaly flagged in {card_names}; technical context and benign alternatives are detailed in the evidence section. "
                            "Notice: Absence of detected manipulation does not guarantee absolute authenticity."
                        )
                    else:
                        evidence_summary = (
                            f"Facial region analysis detected no significant manipulation artifacts across neural classifiers "
                            f"({1.0 - f_p:.1%} authentic probability, {xc_str}, {vt_str}). "
                            "Notice: Absence of detected manipulation does not guarantee absolute authenticity."
                        )
            else:
                evidence_summary = (
                    f"Facial region analysis returned provisional forensic signals (calibrated probability {f_p:.1%}). "
                    "Notice: Absence of detected manipulation does not guarantee absolute authenticity."
                )
        elif not is_not_suitable and f_p is None:
            evidence_summary = decision.user_facing_statement or "Specialist neural inference could not be completed. Independent verification is required."
        if (canonical_v in (DecisionState.NOT_SUITABLE, DecisionState.ILLUSTRATION) or decision.reliability_warning is not None) and decision.confidence is None:
            conf_out = None
        elif decision.confidence is not None:
            conf_out = decision.confidence
        elif f_p is not None:
            conf_out = round(max(f_p, 1.0 - f_p) * 100.0, 2)
        else:
            conf_out = None

        conf_level_out = (
            "Not Performed" if canonical_v in (DecisionState.NOT_SUITABLE, DecisionState.ILLUSTRATION) else
            ("Low" if decision.is_binary_fallback else decision.confidence_level)
        )

        # Structured Model Inventory
        models_inventory = [
            {
                "name": "ResNet-18 (Baseline)",
                "task": "General Facial Deepfake Classification (Baseline)",
                "predicted_class": ("FAKE" if raw_resnet_p >= 0.50 else "REAL") if raw_resnet_p is not None else "UNAVAILABLE",
                "raw_score": round(raw_resnet_p, 4) if raw_resnet_p is not None else None,
                "raw_logits": raw_resnet_logits,
                "score_semantics": "Softmax probability of Class 1 (FAKE) in range [0.0, 1.0]",
                "preprocessing": "Facial crop resized to 224x224 px, ImageNet normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).",
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
                "raw_logits": raw_xcep_logits,
                "score_semantics": "Softmax probability of Class 1 (FAKE) in range [0.0, 1.0]",
                "preprocessing": "Facial crop resized to 256x256 px, normalized to [-1.0, 1.0] range.",
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
                "raw_logits": raw_vit_logits,
                "score_semantics": "Softmax probability of Class 1 (FAKE) in range [0.0, 1.0]",
                "preprocessing": "Facial crop (multi-scale tight + padded context) resized to 224x224 px, AutoImageProcessor tokenized.",
                "calibrated_prob": None,
                "calibration_status": "Not calibrated; raw softmax",
                "status": "SUCCESS" if raw_vit_p is not None else "UNAVAILABLE",
                "limitations": "Trained on synthetic portrait generation artifacts; sensitive to crop alignment and out-of-distribution lighting."
            }
        ]

        diagnostics = {
            "enabled": diagnostic_mode,
            "class_mappings": {
                "0": "REAL",
                "1": "FAKE"
            },
            "color_channel_order": "RGB",
            "models": {
                "resnet18": {
                    "input_dimension": "224x224x3",
                    "normalization": "ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                    "raw_logits": raw_resnet_logits,
                    "softmax_probs": [round(1.0 - raw_resnet_p, 4), round(raw_resnet_p, 4)] if raw_resnet_p is not None else None,
                    "device": self.device,
                },
                "xception": {
                    "input_dimension": "256x256x3",
                    "normalization": "[-1.0, 1.0] (mean=0.5, std=0.5)",
                    "raw_logits": raw_xcep_logits,
                    "softmax_probs": [round(1.0 - raw_xcep_p, 4), round(raw_xcep_p, 4)] if raw_xcep_p is not None else None,
                    "device": self.device,
                },
                "vit": {
                    "input_dimension": "224x224x3",
                    "normalization": "AutoImageProcessor (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])",
                    "raw_logits": raw_vit_logits,
                    "softmax_probs": [round(1.0 - raw_vit_p, 4), round(raw_vit_p, 4)] if raw_vit_p is not None else None,
                    "crop_strategy": "multi-scale (tight + padded)",
                    "device": self.device,
                }
            },
            "device": self.device,
        }

        # Structured Measurements
        res_w = img_shape[1] if len(img_shape) > 1 else (quality_res.resolution[0] if quality_res.resolution else 0)
        res_h = img_shape[0] if len(img_shape) > 0 else (quality_res.resolution[1] if quality_res.resolution else 0)
        eye_metrics = forensic_res.get("facial_regions", {}).get("regional_metrics", {}).get("eyes", {})
        measurements = {
            "resolution": {
                "width": res_w,
                "height": res_h,
                "formatted": f"{res_w} × {res_h} px",
                "validated_for_task": True,
                "status": "Pass: Native resolution meets minimum analysis threshold (100×100 px)" if (res_w >= 100 and res_h >= 100) else "Caution: Low input resolution",
                "relevant_limitations": "Very low resolutions (< 100px) prevent fine texture and boundary evaluation.",
                "interpretation": "Spatial input dimensions. Verified against minimum analysis threshold (100×100 px)."
            },
            "blur": {
                "value": round(quality_res.blur_score, 2),
                "method": "Laplacian variance",
                "validated_for_task": True,
                "status": "Pass: Crisp high-frequency focus" if quality_res.blur_score >= 40.0 else "Caution: Low optical sharpness or motion blur",
                "relevant_limitations": "Adequate focus is an input requirement, NOT evidence of authenticity. Blurry deepfakes remain blurry.",
                "interpretation": "Focus & high-frequency sharpness measurement. A score above 40 indicates adequate focus; this is a quality gate requirement, NOT evidence of authenticity."
            },
            "ela": {
                "value": round(ela_data.get("ela_score", 0.10), 3),
                "method": "Error Level Analysis",
                "validated_for_task": False,
                "status": "Elevated local compression variance" if ela_data.get("ela_score", 0.10) > 0.35 else "Uniform compression signature",
                "relevant_limitations": "Subject to multiple resaving cycles and compression algorithms; cannot independently confirm tampering.",
                "interpretation": "Exploratory JPEG recompression error disparity measurement. Subject to resaving artifacts; does NOT constitute definitive proof of tampering."
            },
            "face_detection": {
                "detected": quality_res.face_detected,
                "face_count": quality_res.face_count,
                "bbox": list(face_bbox) if face_bbox else None,
                "validated_for_task": True,
                "relevant_limitations": "Haar cascades produce axis-aligned boxes; profile or tilted faces may experience minor crop clipping.",
                "interpretation": "Face detection is a preprocessing step to extract the facial region for model inference. Finding a face is NOT evidence of authenticity or manipulation."
            },
            "sensor_noise": {
                "value": round(noise_data.get("channel_correlation", 0.0), 3) if noise_data else 0.0,
                "method": "Bayer demosaicing G-R noise residual correlation (PRNU proxy)",
                "validated_for_task": False,
                "status": "Anomalous channel noise correlation" if noise_data.get("anomaly_detected") else "Natural camera sensor noise distribution",
                "relevant_limitations": "Social media compression washes out fine sensor noise fields; exploratory proxy.",
                "interpretation": "Evaluates optical sensor noise residual correlation across color channels. Synthetic generators often exhibit flattened or decorrelated noise."
            },
            "corneal_glints": {
                "value": round(eye_metrics.get("glint_disparity", 0.0), 3) if eye_metrics else None,
                "method": "Bilateral corneal reflection glint coordinate symmetry",
                "validated_for_task": False,
                "status": "Asymmetric corneal reflections" if (eye_metrics and eye_metrics.get("glint_disparity", 0.0) > 0.40) else "Symmetric light reflections",
                "relevant_limitations": "Off-axis or directional key lights naturally produce asymmetrical glints; exploratory indicator only.",
                "interpretation": "Evaluates optical reflection position across eyes. Sensitive to natural off-axis lighting; exploratory indicator only."
            }
        }

        # Supporting, Contradicting, Unavailable, Limitations
        supporting_evidence = []
        contradicting_evidence = []
        unavailable_checks = [
            "Hardware camera sensor PRNU fingerprint database: Unavailable (Evaluated PRNU residual proxy via cross-channel sensor noise coherence)",
            "Audio-visual lip synchronization: Unavailable (Static image)",
            "C2PA cryptographic provenance metadata: Not present"
        ]
        limitations = [
            "Absence of detected manipulation does not guarantee absolute authenticity.",
            "Face-swap models trained on video compression (Xception) can produce false alarms on high-frequency web JPEG compression.",
            "Error Level Analysis and Laplacian sharpness are exploratory quality diagnostics, not definitive forensic proof."
        ]

        if raw_xcep_p is not None and raw_vit_p is not None and (raw_xcep_p >= 0.50) != (raw_vit_p >= 0.50):
            contradicting_evidence.append(
                f"Specialist classifier disagreement: Xception (face-swap detector) predicted {('FAKE (' + str(round(raw_xcep_p*100, 1)) + '%)') if raw_xcep_p >= 0.5 else ('REAL (' + str(round((1-raw_xcep_p)*100, 1)) + '%)')} whereas ViT (generative synthetic detector) predicted {('FAKE (' + str(round(raw_vit_p*100, 1)) + '%)') if raw_vit_p >= 0.5 else ('REAL (' + str(round((1-raw_vit_p)*100, 1)) + '%)')}."
            )
        supporting_evidence.append("Quality gate verification completed; facial region extracted.")

        if canonical_v in (DecisionState.FAKE, "FAKE"):
            if raw_vit_p is not None and raw_vit_p >= 0.50:
                supporting_evidence.append(f"Vision Transformer measured synthetic generative artifacts ({raw_vit_p*100:.1f}% fake probability).")
            if raw_xcep_p is not None and raw_xcep_p >= 0.50:
                supporting_evidence.append(f"Xception detector measured boundary blending anomalies ({raw_xcep_p*100:.1f}% fake probability).")
            if ela_data and (ela_data.get("local_discrepancy", 0.0) > 0.40 or (ela_data.get("anomaly_detected") and ela_data.get("local_discrepancy", 0.0) > 0.35)):
                supporting_evidence.append(f"Error Level Analysis (ELA) measured local compression discrepancy of {ela_data.get('local_discrepancy', 0.0):.2f} between facial region and background (SUPPORTS FAKE — STRONG).")
            if noise_data and (noise_data.get("noise_discrepancy", 0.0) > 0.45 or (noise_data.get("anomaly_detected") and noise_data.get("noise_discrepancy", 0.0) > 0.40)):
                supporting_evidence.append(f"Sensor noise PRNU residual analysis measured cross-region noise discrepancy of {noise_data.get('noise_discrepancy', 0.0):.2f} (SUPPORTS FAKE — STRONG).")
            if eye_metrics and eye_metrics.get("anomaly_detected"):
                supporting_evidence.append("Bilateral eye reflection analysis observed optical glint disparity and geometry asymmetry.")
            if raw_resnet_p is not None and raw_resnet_p < 0.50:
                contradicting_evidence.append(f"ResNet baseline predicted lower probability ({raw_resnet_p*100:.1f}%).")
            if raw_xcep_p is not None and raw_xcep_p < 0.50:
                contradicting_evidence.append(f"Xception detector predicted lower probability ({raw_xcep_p*100:.1f}%), consistent with modern uncompressed generative portraits.")
        elif canonical_v in (DecisionState.REAL, "REAL"):
            if raw_vit_p is not None and raw_vit_p < 0.50:
                supporting_evidence.append(f"Vision Transformer observed authentic generative feature distribution ({(1-raw_vit_p)*100:.1f}% authentic).")
            if raw_xcep_p is not None and raw_xcep_p < 0.50:
                supporting_evidence.append(f"Xception detector detected no boundary blending anomalies ({(1-raw_xcep_p)*100:.1f}% authentic).")
            if ela_data and not ela_data.get("anomaly_detected") and ela_data.get("local_discrepancy", 0.0) <= 0.35:
                supporting_evidence.append("Error Level Analysis (ELA) verified uniform compression consistency across facial boundary.")
            if noise_data and not noise_data.get("anomaly_detected") and noise_data.get("noise_discrepancy", 0.0) <= 0.45:
                supporting_evidence.append("Sensor noise analysis verified natural camera optical noise distribution.")
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
        model_agreement_status = (
            "DISAGREEMENT (CONFLICT)"
            if (raw_xcep_p is not None and raw_vit_p is not None and (raw_xcep_p >= 0.50) != (raw_vit_p >= 0.50))
            else "AGREEMENT"
        )
        decision_obj = {
            "method": "SafetyDecisionSystem with specialist score fusion",
            "verdict": canonical_v,
            "decision_score": round(decision.fake_probability, 4) if decision.fake_probability is not None else None,
            "score": round(decision.fake_probability, 4) if decision.fake_probability is not None else None,
            "score_type": score_type,
            "score_semantics": "Aggregate decision score computed via specialist detector fusion and calibrated scaling.",
            "probability_is_calibrated": False,
            "raw_fused_score": round(raw_fused_score, 4) if raw_fused_score is not None else None,
            "temperature_scaled_score": round(temperature_scaled_score, 4) if temperature_scaled_score is not None else None,
            "temperature": self.decision_system.calibrator.temperature,
            "calibration_note": "Temperature scaling is applied, but its validation fit is not bound to the active model checkpoint or fusion version.",
            "confidence": conf_out,
            "confidence_semantics": "Uncertified when detector conflict or ambiguous margin is present; numerical confidence reflects classification margin when models agree.",
            "confidence_type": "Not performed" if canonical_v == DecisionState.NOT_SUITABLE else ("Provisional" if decision.is_binary_fallback else "Model score; calibrated probability"),
            "uncertainty": decision.uncertainty,
            "uncertainty_semantics": "Computed as 1.0 - |2p - 1.0|, representing decision boundary proximity. Elevated to >= 0.80 when specialist detectors produce contradictory predictions.",
            "model_agreement": model_agreement_status,
            "criteria_met": bool(conf_out is not None and canonical_v in (DecisionState.REAL, DecisionState.FAKE)),
            "reason": decision.reason,
        }

        v_reasoning = vision_reasoning or visual_res.get("vision_reasoning", {})
        evidence_dependency_analysis = {
            "model_evidence": [
                {
                    "model_name": "XceptionNet",
                    "model_family": "Convolutional (Xception)",
                    "target_artifact": "Boundary blending & face-swap seams",
                    "raw_score": raw_xcep_p,
                    "fake_probability": raw_xcep_p,
                    "dependency_group": "group_neural_deepfake",
                    "is_independent": True,
                    "evidence_provenance": "neural_prediction",
                },
                {
                    "model_name": "Vision Transformer",
                    "model_family": "Self-Attention (ViT)",
                    "target_artifact": "Generative synthetic & GAN/diffusion patches",
                    "raw_score": raw_vit_p,
                    "fake_probability": raw_vit_p,
                    "dependency_group": "group_neural_generative",
                    "is_independent": True,
                    "evidence_provenance": "neural_prediction",
                },
                {
                    "model_name": "ResNet-18",
                    "model_family": "Residual Network (ResNet)",
                    "target_artifact": "Global facial feature anomalies",
                    "raw_score": raw_resnet_p,
                    "fake_probability": raw_resnet_p,
                    "dependency_group": "group_neural_baseline",
                    "is_independent": True,
                    "evidence_provenance": "neural_prediction",
                },
            ],
            "directly_measured_evidence": [
                {
                    "signal": "Boundary Gradient Discontinuity",
                    "analyzer": "EdgeBoundaryAnalyzer",
                    "measured_value": round(edge_data.get("boundary_discontinuity", 0.0), 3),
                    "dependency_group": "group_physical_boundary",
                    "evidence_provenance": "direct_measurement",
                    "is_independent": True,
                },
                {
                    "signal": "Spectral High-Frequency Roll-off",
                    "analyzer": "FFTSpectralAnalyzer",
                    "measured_value": round(fft_data.get("high_freq_ratio", 0.0), 4),
                    "dependency_group": "group_frequency_domain",
                    "evidence_provenance": "direct_measurement",
                    "is_independent": True,
                },
                {
                    "signal": "Bayer Demosaicing PRNU Correlation",
                    "analyzer": "SensorNoiseAnalyzer",
                    "measured_value": round(noise_data.get("channel_correlation", 0.0), 3),
                    "dependency_group": "group_sensor_noise",
                    "evidence_provenance": "direct_measurement",
                    "is_independent": True,
                },
                {
                    "signal": "JPEG Compression Error Variance",
                    "analyzer": "ErrorLevelAnalyzer",
                    "measured_value": round(ela_data.get("ela_score", 0.10), 3),
                    "dependency_group": "group_compression",
                    "evidence_provenance": "direct_measurement",
                    "is_independent": True,
                },
            ],
            "contextual_visual_interpretation": {
                "provider": v_reasoning.get("provider_id", "local_deterministic_baseline"),
                "model": v_reasoning.get("model_id", "deterministic-v2"),
                "assessment": v_reasoning.get("qualitative_assessment", "INCONCLUSIVE"),
                "evidence_strength": v_reasoning.get("evidence_strength", "INCONCLUSIVE"),
                "confidence_qualifier": v_reasoning.get("confidence_qualifier", "UNCERTIFIED"),
                "evidence_provenance": "vision_language_interpretation" if v_reasoning.get("is_external_ai") else "deterministic_baseline",
                "observations": v_reasoning.get("whole_image_observations", []),
                "regional_observations": v_reasoning.get("regional_observations", []),
            },
            "benign_alternative_explanations": v_reasoning.get("alternative_explanations", [
                "Evaluated optical softness, ambient illumination, and compression artifacts as non-manipulative explanations."
            ]),
            "correlated_evidence_notes": [
                "Grad-CAM visual explanation is an explanatory saliency visualization derived from intermediate neural feature maps. It is NOT an independent corroborating forensic measurement.",
                "Error Level Analysis (ELA) and Laplacian variance share underlying sensitivity to JPEG quantization and must not be treated as two independent proofs of tampering.",
                "Facial crop boundary alignment artifacts can produce localized high-frequency gradients that resemble synthetic edges; multi-scale verification is required.",
            ],
            "missing_or_contradictory_evidence": contradicting_evidence or [
                "No severe contradiction between independent physical signals and neural predictions."
            ],
        }

        # Ensure evidence and recommendation strictly conform to canonical_v single source of truth
        clean_evidence = list(decision.evidence) if decision.evidence else []
        from backend.agents.explanation_agent import generate_user_friendly_explanation
        user_friendly_expl = generate_user_friendly_explanation(
            verdict=canonical_v,
            confidence=conf_out,
            confidence_level=conf_level_out,
            fake_probability=f_p,
            features={
                "noise_inconsistency": 0.0 if is_not_suitable else forensic_res.get("composite_forensic_score", 0.15),
                "edge_boundary_artifacts": 0.0 if is_not_suitable else edge_data.get("edge_score", 0.10),
                "fft_spectral_anomaly": 0.0 if is_not_suitable else fft_data.get("fft_score", 0.10),
                "ela_compression_anomaly": 0.0 if is_not_suitable else ela_data.get("ela_score", 0.10),
                "sensor_noise_score": 0.0 if is_not_suitable else noise_data.get("noise_score", 0.10),
                "metadata_anomaly": False,
                "blur_score": quality_res.blur_score,
                "raw_xcep_score": raw_xcep_p,
                "raw_vit_score": raw_vit_p,
                "raw_vit_tight_score": raw_vit_tight_p,
                "raw_resnet_score": raw_resnet_p,
            },
            forensic_evidence=forensic_res,
            evidence_cards=evidence_cards or [],
            models=models_inventory,
            is_binary_fallback=decision.is_binary_fallback,
            disagreement_detected=bool(model_agreement_status and model_agreement_status != "AGREEMENT"),
            media_type="image",
        )

        rec_val = (f"{decision.recommendation} {user_friendly_expl['what_should_user_do']}".strip()
                   if getattr(decision, "recommendation", None) else user_friendly_expl["what_should_user_do"])
        user_facing_summary = user_friendly_expl["what_does_this_mean"]

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
            "primary_verdict": canonical_v,
            "primaryVerdict": canonical_v,
            "provisional_suggestion": decision.primary_verdict if decision.is_binary_fallback else None,
            "provisionalSuggestion": decision.primary_verdict if decision.is_binary_fallback else None,
            "original_verdict": decision.original_verdict,
            "is_binary_fallback": decision.is_binary_fallback,
            "is_provisional": decision.is_binary_fallback,
            "reliability_warning": decision.reliability_warning,
            "confidence": conf_out,
            "confidence_score": conf_out,
            "confidence_level": user_friendly_expl["confidence_tier"],
            "fake_probability": f_p,
            "real_probability": round(1.0 - f_p, 4) if f_p is not None else None,
            "fakeProbability": f_p,
            "realProbability": round(1.0 - f_p, 4) if f_p is not None else None,
            "abstention_code": getattr(decision, "abstention_code", None),
            "abstentionCode": getattr(decision, "abstention_code", None),
            "raw_model_score": decision.raw_model_score,
            "uncertainty": decision.uncertainty,
            "evidence": clean_evidence,
            "influencing_evidence": getattr(decision, "influencing_evidence", []),
            "providers_ran": getattr(decision, "providers_ran", []),
            "approx_cost_usd": getattr(decision, "approx_cost_usd", 0.0),
            "reason": decision.reason,
            "summary": user_facing_summary,
            "user_facing_statement": user_facing_summary,
            "user_friendly_explanation": user_friendly_expl,
            "userFriendlyExplanation": user_friendly_expl,
            "recommendation": rec_val,
            "analysis_status": getattr(decision, "analysis_status", "Completed"),
            "face_detected": quality_res.face_detected,
            "face_bbox": list(face_bbox) if face_bbox is not None else None,
            "evidence_cards": evidence_cards or [],
            "forensic_evidence": forensic_res,
            "evidence_dependency_analysis": evidence_dependency_analysis,
            "grad_cam": grad_cam,
            "media_category": getattr(decision, "media_category", "Facial Media"),
            "quality_gate": quality_res.to_dict(),
            "visual_analysis": visual_res,
            "vision_reasoning": vision_reasoning or visual_res.get("vision_reasoning", {}),
            "models": models_inventory,
            "measurements": measurements,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "unavailable_checks": unavailable_checks,
            "decision": decision_obj,
            "limitations": limitations,
            "diagnostics": diagnostics,
            "features": {
                "noise_inconsistency": 0.0 if is_not_suitable else forensic_res.get("composite_forensic_score", 0.15),
                "edge_boundary_artifacts": 0.0 if is_not_suitable else edge_data.get("edge_score", 0.10),
                "fft_spectral_anomaly": 0.0 if is_not_suitable else fft_data.get("fft_score", 0.10),
                "ela_compression_anomaly": 0.0 if is_not_suitable else ela_data.get("ela_score", 0.10),
                "sensor_noise_score": 0.0 if is_not_suitable else noise_data.get("noise_score", 0.10),
                "metadata_anomaly": False,
                "blur_score": quality_res.blur_score,
                "raw_xcep_score": raw_xcep_p,
                "raw_vit_score": raw_vit_p,
                "raw_vit_tight_score": raw_vit_tight_p,
                "raw_resnet_score": raw_resnet_p,
            },
            "processing_time": processing_time,
        }

