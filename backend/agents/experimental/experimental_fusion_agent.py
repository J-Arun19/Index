import os
import cv2
import time
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from backend.utils.logger import setup_logger
from backend.preprocessing.quality_gate import ImageQualityGate, QualityGateResult
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.calibration import TemperatureScalingCalibrator
from backend.core.decision_system import SafetyDecisionSystem, DecisionState, DecisionResult, USER_FACING_STATEMENTS
from backend.agents.experimental.experimental_image_agent import ExperimentalImageAgent
from backend.agents.experimental.experimental_sbi_agent import ExperimentalSBIImageAgent
from backend.agents.experimental.experimental_video_agent import ExperimentalVideoAgent
from backend.agents.experimental.experimental_sbi_video_agent import ExperimentalSBIVideoAgent

logger = setup_logger("ExperimentalFusion")


class ForensicFusionEngine:
    """
    Principled multi-detector forensic fusion engine for SBI (EfficientNet-B4) and Xception.
    
    Principles:
    1. Calibrated Fusion: Raw scores from heterogeneous architectures and datasets cannot be
       treated as directly comparable. They are transformed via temperature scaling into a
       common calibrated probability space.
    2. Confidence-Weighted Combination: More confident models receive higher weight:
       w_m = 2 * |p_m - 0.5|.
    3. Conservative Disagreement-to-UNCERTAIN Rule:
       When one model indicates REAL (p < tau_real=0.35) and the other indicates FAKE (p > tau_fake=0.65),
       the models are in direct forensic conflict. The engine abstains to UNCERTAIN rather than
       guessing. This simultaneously suppresses Xception's domain false positives and prevents
       SBI's blind spots (facial reenactments) from being falsely certified authentic.
    4. Guarded Evaluation: Fully preserves NOT_SUITABLE for non-facial media, handles missing
       or None scores safely, and respects image texture / video temporal consistency gates.
    """

    def __init__(
        self,
        calibrator: Optional[TemperatureScalingCalibrator] = None,
        tau_real: float = 0.35,
        tau_fake: float = 0.65,
        min_confidence: float = 60.0,
        t_sbi: Optional[float] = None,
        t_xcep: Optional[float] = None,
    ):
        self.calibrator = calibrator or TemperatureScalingCalibrator()
        self.tau_real = float(tau_real)
        self.tau_fake = float(tau_fake)
        self.min_confidence = float(min_confidence)

        # Architecture-specific temperatures from validation calibration
        exp_cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models", "experimental", "fusion_calibration_config.json"
        )
        self.t_sbi = t_sbi
        self.t_xcep = t_xcep
        if (self.t_sbi is None or self.t_xcep is None) and os.path.exists(exp_cfg_path):
            try:
                import json
                with open(exp_cfg_path, "r", encoding="utf-8") as fp:
                    cfg_data = json.load(fp)
                    if self.t_sbi is None:
                        self.t_sbi = float(cfg_data.get("sbi", {}).get("temperature", self.calibrator.temperature))
                    if self.t_xcep is None:
                        self.t_xcep = float(cfg_data.get("xception", {}).get("temperature", self.calibrator.temperature))
                logger.info(f"Loaded architecture-specific calibration: T_sbi={self.t_sbi}, T_xcep={self.t_xcep}")
            except Exception as e:
                logger.warning(f"Could not load experimental calibration config: {e}")

    def calibrate(self, raw_score: Optional[float], temperature: Optional[float] = None) -> Optional[float]:
        """Calibrates a raw model fake probability using temperature scaling."""
        if raw_score is None:
            return None
        try:
            t = temperature if temperature is not None else self.calibrator.temperature
            p = float(np.clip(raw_score, 1e-6, 1.0 - 1e-6))
            logit = np.log(p / (1.0 - p))
            scaled_logit = logit / max(1e-2, t)
            cal_p = 1.0 / (1.0 + np.exp(-scaled_logit))
            return float(np.clip(cal_p, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Calibration failed for raw score {raw_score}: {e}")
            return float(raw_score)

    def get_confidence_weight(self, prob: Optional[float]) -> Tuple[float, float, float]:
        """
        Computes (confidence_percent, uncertainty_ratio, certainty_weight).
        Returns (50.0, 1.0, 0.0) if prob is None.
        """
        if prob is None:
            return 50.0, 1.0, 0.0
        p = float(np.clip(prob, 0.0, 1.0))
        conf_pct, unc = self.calibrator.get_confidence_and_uncertainty(p)
        # Weight represents certainty distance from random guess 0.5:
        # p=0.5 -> weight=0.0; p=0.0 or 1.0 -> weight=1.0
        weight = float(max(1e-4, 2.0 * abs(p - 0.5)))
        return conf_pct, unc, weight

    def fuse_probabilities(
        self,
        p_sbi: Optional[float],
        p_xcep: Optional[float],
    ) -> Dict[str, Any]:
        """
        Fuses calibrated probabilities from SBI and Xception using confidence weighting.
        Handles None values safely.
        """
        has_sbi = p_sbi is not None
        has_xcep = p_xcep is not None

        if not has_sbi and not has_xcep:
            return {
                "fused_probability": 0.50,
                "confidence": 50.0,
                "uncertainty": 1.0,
                "sbi_weight": 0.0,
                "xcep_weight": 0.0,
                "status": "both_models_missing",
            }

        if has_sbi and not has_xcep:
            conf, unc, w = self.get_confidence_weight(p_sbi)
            return {
                "fused_probability": float(p_sbi),
                "confidence": conf,
                "uncertainty": unc,
                "sbi_weight": 1.0,
                "xcep_weight": 0.0,
                "status": "fallback_sbi_only",
            }

        if not has_sbi and has_xcep:
            conf, unc, w = self.get_confidence_weight(p_xcep)
            return {
                "fused_probability": float(p_xcep),
                "confidence": conf,
                "uncertainty": unc,
                "sbi_weight": 0.0,
                "xcep_weight": 1.0,
                "status": "fallback_xcep_only",
            }

        # Both models present
        _, _, w_sbi = self.get_confidence_weight(p_sbi)
        _, _, w_xcep = self.get_confidence_weight(p_xcep)

        total_w = w_sbi + w_xcep
        if total_w < 1e-6:
            p_fused = 0.50
        else:
            p_fused = (w_sbi * float(p_sbi) + w_xcep * float(p_xcep)) / total_w

        p_fused = float(np.clip(p_fused, 0.0, 1.0))
        conf_pct, unc = self.calibrator.get_confidence_and_uncertainty(p_fused)

        return {
            "fused_probability": p_fused,
            "confidence": conf_pct,
            "uncertainty": unc,
            "sbi_weight": round(w_sbi / total_w, 4) if total_w > 0 else 0.5,
            "xcep_weight": round(w_xcep / total_w, 4) if total_w > 0 else 0.5,
            "status": "complete_fusion",
        }

    def evaluate(
        self,
        raw_sbi: Optional[float],
        raw_xcep: Optional[float],
        quality_result: Optional[QualityGateResult] = None,
        forensic_result: Optional[Dict[str, Any]] = None,
        temporal_variance: Optional[float] = None,
        temporal_conflict: bool = False,
        ground_truth: Optional[int] = None,
    ) -> DecisionResult:
        """
        Executes complete multimodal decision fusion:
        1. Media suitability gate (NOT_SUITABLE for non-face media)
        2. Media quality gate (UNCERTAIN for degraded/blurry media)
        3. Model score calibration
        4. Temporal consistency check
        5. Conservative model disagreement check (UNCERTAIN when SBI and Xception conflict)
        6. Boundary ambiguity check (UNCERTAIN if in [tau_real, tau_fake])
        7. Low confidence check (UNCERTAIN if conf < min_confidence)
        8. Unanimous or dominant FAKE decision
        9. Authenticity certification policy check
        10. LIKELY_REAL decision
        """
        evidence_list: List[str] = []

        # 1. Media Eligibility Validation
        if quality_result is not None:
            if getattr(quality_result, "eligibility", None) == "UNSUPPORTED":
                return DecisionResult(
                    prediction=DecisionState.NOT_SUITABLE,
                    fake_probability=None,
                    real_probability=None,
                    raw_model_score=None,
                    confidence=None,
                    confidence_level="Not Performed",
                    uncertainty=None,
                    evidence=["Eligibility: Unsupported media (No detectable human face found in image)."],
                    reason="No suitable face detected",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE],
                    quality_passed=False,
                    face_valid=False,
                    forensic_signals={"quality_gate": quality_result.to_dict()},
                    analysis_status="Not performed",
                    media_category="Document / Non-face image",
                    recommendation="Upload a clear image containing a visible human face.",
                )

            if not quality_result.is_suitable:
                rejection = quality_result.rejection_reason or "Input failed media quality checks."
                evidence_list.append(f"Quality Gate: {rejection}")
                for issue in quality_result.issues:
                    evidence_list.append(f"Defect: {issue}")

                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=0.50,
                    real_probability=0.50,
                    raw_model_score=0.50,
                    confidence=50.0,
                    confidence_level="Low",
                    uncertainty=1.0,
                    evidence=evidence_list,
                    reason=f"Abstained due to image quality failure: {rejection}",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                    quality_passed=False,
                    face_valid=quality_result.face_detected,
                    forensic_signals={"quality_gate": quality_result.to_dict()},
                    analysis_status="Completed",
                    media_category="Low Quality Facial Media",
                    recommendation="Upload a higher resolution or sharper facial image.",
                )

        # 2. Probability Calibration
        cal_sbi = self.calibrate(raw_sbi, temperature=self.t_sbi)
        cal_xcep = self.calibrate(raw_xcep, temperature=self.t_xcep)

        if cal_sbi is None and cal_xcep is None:
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=0.50,
                real_probability=0.50,
                raw_model_score=0.50,
                confidence=50.0,
                confidence_level="Low",
                uncertainty=1.0,
                evidence=["Model Failure: Both SBI and Xception scores were unavailable."],
                reason="Abstained due to complete model failure (scores None).",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
            )

        # 3. Fuse Probabilities
        fuse_res = self.fuse_probabilities(cal_sbi, cal_xcep)
        p_fused = fuse_res["fused_probability"]
        conf_score = fuse_res["confidence"]
        base_uncertainty = fuse_res["uncertainty"]
        p_real = round(1.0 - p_fused, 4)

        raw_fused = p_fused
        if raw_sbi is not None and raw_xcep is not None:
            raw_fused = float((raw_sbi + raw_xcep) / 2.0)
        elif raw_sbi is not None:
            raw_fused = float(raw_sbi)
        elif raw_xcep is not None:
            raw_fused = float(raw_xcep)

        # Forensic signal aggregation
        suspicious_signals = 0
        forensic_signals = {}
        if forensic_result:
            forensic_signals = forensic_result
            evidence_list.extend(forensic_result.get("evidence_items", []))
            suspicious_signals += forensic_result.get("suspicious_signal_count", 0)

        # Evidence logging of fusion details
        sbi_str = f"SBI={cal_sbi:.3f}" if cal_sbi is not None else "SBI=None"
        xcep_str = f"Xception={cal_xcep:.3f}" if cal_xcep is not None else "Xception=None"
        evidence_list.append(
            f"Fusion Breakdown: {sbi_str} (w={fuse_res['sbi_weight']:.2f}), "
            f"{xcep_str} (w={fuse_res['xcep_weight']:.2f}) -> Calibrated Fused P={p_fused:.3f}"
        )

        conf_tier = "High" if conf_score >= 80.0 else ("Medium" if conf_score >= 60.0 else "Low")

        # 4. Temporal Consistency Check (for video)
        if temporal_conflict or (temporal_variance is not None and temporal_variance > 0.22):
            var_str = f"{temporal_variance:.3f}" if temporal_variance is not None else "N/A"
            evidence_list.append(
                f"Temporal Inconsistency: High variance across frames ({var_str} > 0.22). "
                "Frame predictions contradict each other."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.75, base_uncertainty),
                evidence=evidence_list,
                reason="Abstained due to temporal inconsistency across video frames. Sequential evidence is contradictory.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 5. CONSERVATIVE DISAGREEMENT CHECK (Safety Rule)
        # If both models produced valid calibrated scores:
        if cal_sbi is not None and cal_xcep is not None:
            sbi_leans_real = cal_sbi < self.tau_real
            sbi_leans_fake = cal_sbi > self.tau_fake
            xcep_leans_real = cal_xcep < self.tau_real
            xcep_leans_fake = cal_xcep > self.tau_fake

            is_direct_conflict = (sbi_leans_real and xcep_leans_fake) or (sbi_leans_fake and xcep_leans_real)
            if is_direct_conflict:
                sbi_stance = "FAKE" if sbi_leans_fake else "REAL"
                xcep_stance = "FAKE" if xcep_leans_fake else "REAL"
                evidence_list.append(
                    f"Model Disagreement: Direct conflict between models. SBI indicates {sbi_stance} (p={cal_sbi:.2f}), "
                    f"while Xception indicates {xcep_stance} (p={cal_xcep:.2f})."
                )
                escalated_uncertainty = max(0.80, base_uncertainty + 0.30)
                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=p_fused,
                    real_probability=p_real,
                    raw_model_score=raw_fused,
                    confidence=conf_score,
                    confidence_level="Low",
                    uncertainty=min(1.0, escalated_uncertainty),
                    evidence=evidence_list,
                    reason=(
                        f"Abstained due to direct model disagreement: SBI indicates {sbi_stance} "
                        f"while Xception indicates {xcep_stance}. Conservative safety policy enforces UNCERTAIN."
                    ),
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                    forensic_signals=forensic_signals,
                )

        # 6. Model / Forensic Disagreement Check
        model_leans_real = p_fused < self.tau_real
        model_leans_fake = p_fused > self.tau_fake

        if model_leans_real and suspicious_signals >= 1:
            evidence_list.append(
                f"Evidence Disagreement: Fused model score indicates REAL (p={p_fused:.2f}), "
                f"but forensic analyzers identified {suspicious_signals} suspicious signal(s)."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.70, base_uncertainty + 0.30),
                evidence=evidence_list,
                reason="Abstained due to contradictory evidence: Models indicate authentic features, but forensic analyzers detected tampering.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 7. Ambiguous Margin / Boundary Check
        if self.tau_real <= p_fused <= self.tau_fake:
            evidence_list.append(
                f"Boundary Ambiguity: Calibrated fused probability ({p_fused:.2f}) falls "
                f"within the indeterminate decision margin [{self.tau_real:.2f}, {self.tau_fake:.2f}]."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level=conf_tier,
                uncertainty=max(0.60, base_uncertainty),
                evidence=evidence_list,
                reason=f"Calibrated probability ({p_fused:.2f}) lies in the ambiguous decision boundary [{self.tau_real:.2f} - {self.tau_fake:.2f}]. Evidence is inconclusive.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 8. Low Confidence Abstention
        if conf_score < self.min_confidence:
            evidence_list.append(
                f"Low Confidence: Fused confidence ({conf_score}%) is below operational threshold ({self.min_confidence}%)."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.65, base_uncertainty),
                evidence=evidence_list,
                reason=f"Fused confidence ({conf_score}%) is insufficient to deliver a reliable verdict.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 9. High-Confidence FAKE Decision
        if model_leans_fake:
            evidence_list.append(
                f"High synthetic probability ({p_fused:.2f}) confirmed by fused forensic analysis."
            )
            return DecisionResult(
                prediction=DecisionState.FAKE,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level=conf_tier,
                uncertainty=base_uncertainty,
                evidence=evidence_list,
                reason=f"Strong evidence of synthetic manipulation (fused calibrated probability: {p_fused:.2f}).",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.FAKE],
                forensic_signals=forensic_signals,
            )

        # 10. Authenticity Certification Policy Check
        edge_metrics = forensic_result.get("edge_analysis", {}) if forensic_result else {}
        b_disc = edge_metrics.get("boundary_discontinuity", 0.0)
        blur_val = quality_result.blur_score if (quality_result and quality_result.blur_score is not None) else edge_metrics.get("laplacian_var", 50.0)

        if (blur_val is not None and blur_val < 38.0) or b_disc > 0.35:
            evidence_list.append(
                f"Authenticity Policy: Texture sharpness ({blur_val:.1f}) is borderline or boundary transition "
                f"discontinuity ({b_disc:.2f}) is elevated. Insufficient evidence to certify authenticity."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=p_fused,
                real_probability=p_real,
                raw_model_score=raw_fused,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.65, base_uncertainty),
                evidence=evidence_list,
                reason="Abstained: Neural scores lean real, but image texture clarity or boundary continuity is insufficient to reliably certify authenticity.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                is_hard_negative=(ground_truth == 1),
            )

        # 11. High-Confidence LIKELY_REAL Decision
        evidence_list.append("SBI and Xception models agree on authentic visual representation without conflicting forensic anomalies.")
        evidence_list.append("Notice: Absence of detected manipulation does not prove absolute authenticity.")

        return DecisionResult(
            prediction=DecisionState.LIKELY_REAL,
            fake_probability=p_fused,
            real_probability=p_real,
            raw_model_score=raw_fused,
            confidence=conf_score,
            confidence_level=conf_tier,
            uncertainty=base_uncertainty,
            evidence=evidence_list,
            reason="No significant evidence of manipulation detected. Both forensic models agree on authentic representation.",
            user_facing_statement=USER_FACING_STATEMENTS[DecisionState.LIKELY_REAL],
            forensic_signals=forensic_signals,
            is_hard_negative=(ground_truth == 1),
        )


class ExperimentalFusionImageAgent:
    """
    Experimental forensic image analysis agent combining SBI EfficientNet-B4 and DeepfakeBench Xception.
    Runs strictly in isolation from production.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quality_gate = ImageQualityGate()
        self.forensics = ForensicEvidenceSuite()
        self.calibrator = TemperatureScalingCalibrator()
        self.fusion_engine = ForensicFusionEngine(calibrator=self.calibrator)
        self.sbi_agent = ExperimentalSBIImageAgent(device=self.device)
        self.xcep_agent = ExperimentalImageAgent(backbone="xception", device=self.device)

    def analyze(self, file_path: str, require_face: bool = True) -> Dict[str, Any]:
        """
        Executes isolated fusion analysis on an image.
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
            decision = self.fusion_engine.evaluate(
                raw_sbi=None,
                raw_xcep=None,
                quality_result=quality_res,
                forensic_result={},
            )
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res={},
                sbi_details=None,
                xcep_details=None,
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=round(time.time() - start_time, 4),
            )

        # 2. Forensics
        face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
        forensic_res = self.forensics.evaluate(img_bgr, face_bbox=face_bbox)

        # Quality failure
        if not quality_res.is_suitable:
            decision = self.fusion_engine.evaluate(
                raw_sbi=0.50,
                raw_xcep=0.50,
                quality_result=quality_res,
                forensic_result=forensic_res,
            )
            return self._format_response(
                decision=decision,
                quality_res=quality_res,
                forensic_res=forensic_res,
                sbi_details=None,
                xcep_details=None,
                filename=filename,
                size=size,
                fmt=fmt,
                processing_time=round(time.time() - start_time, 4),
            )

        # 3. Model Inferences
        t0_sbi = time.time()
        try:
            res_sbi = self.sbi_agent.analyze(file_path, require_face=require_face, crop_mode="sbi_margin")
            dt_sbi = round(time.time() - t0_sbi, 4)
            raw_sbi = res_sbi.get("raw_model_score")
        except Exception as e:
            logger.error(f"SBI inference failed: {e}")
            dt_sbi = round(time.time() - t0_sbi, 4)
            res_sbi = {"error": str(e)}
            raw_sbi = None

        t0_xcep = time.time()
        try:
            res_xcep = self.xcep_agent.analyze(file_path, require_face=require_face)
            dt_xcep = round(time.time() - t0_xcep, 4)
            raw_xcep = res_xcep.get("raw_model_score")
        except Exception as e:
            logger.error(f"Xception inference failed: {e}")
            dt_xcep = round(time.time() - t0_xcep, 4)
            res_xcep = {"error": str(e)}
            raw_xcep = None

        # 4. Fusion Evaluation
        decision = self.fusion_engine.evaluate(
            raw_sbi=raw_sbi,
            raw_xcep=raw_xcep,
            quality_result=quality_res,
            forensic_result=forensic_res,
        )

        sbi_details = {
            "raw_score": raw_sbi,
            "calibrated_prob": res_sbi.get("fake_probability") if isinstance(res_sbi, dict) else None,
            "verdict": res_sbi.get("verdict") if isinstance(res_sbi, dict) else None,
            "latency_sec": dt_sbi,
        }
        xcep_details = {
            "raw_score": raw_xcep,
            "calibrated_prob": res_xcep.get("fake_probability") if isinstance(res_xcep, dict) else None,
            "verdict": res_xcep.get("verdict") if isinstance(res_xcep, dict) else None,
            "latency_sec": dt_xcep,
        }

        return self._format_response(
            decision=decision,
            quality_res=quality_res,
            forensic_res=forensic_res,
            sbi_details=sbi_details,
            xcep_details=xcep_details,
            filename=filename,
            size=size,
            fmt=fmt,
            processing_time=round(time.time() - start_time, 4),
        )

    def _format_response(
        self,
        decision: DecisionResult,
        quality_res: Any,
        forensic_res: Dict[str, Any],
        sbi_details: Optional[Dict[str, Any]],
        xcep_details: Optional[Dict[str, Any]],
        filename: str,
        size: int,
        fmt: str,
        processing_time: float,
    ) -> Dict[str, Any]:
        # Determine forced binary prediction (excluding NOT_SUITABLE)
        if decision.prediction == DecisionState.NOT_SUITABLE:
            forced_binary = "NOT_SUITABLE"
        else:
            p_fake = decision.fake_probability if decision.fake_probability is not None else 0.50
            forced_binary = "FAKE" if p_fake >= 0.50 else "REAL"

        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "format": fmt,
                "experimental_fusion": "sbi_efficientnet_b4_plus_deepfakebench_xception",
            },
            "prediction": decision.prediction,
            "verdict": decision.verdict,
            "forced_binary": forced_binary,
            "confidence": decision.confidence,
            "confidence_score": decision.confidence,
            "confidence_level": decision.confidence_level,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": decision.raw_model_score,
            "uncertainty": decision.uncertainty,
            "evidence": decision.evidence,
            "reason": decision.reason,
            "user_facing_statement": decision.user_facing_statement,
            "quality_gate": quality_res.to_dict() if hasattr(quality_res, "to_dict") else {},
            "forensic_signals": forensic_res,
            "sbi_details": sbi_details,
            "xception_details": xcep_details,
            "processing_time": processing_time,
            "is_hard_negative": decision.is_hard_negative,
        }


class ExperimentalFusionVideoAgent:
    """
    Experimental forensic video analysis agent combining SBI Video Agent and Xception Video Agent.
    Runs strictly in isolation from production.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.calibrator = TemperatureScalingCalibrator()
        self.fusion_engine = ForensicFusionEngine(calibrator=self.calibrator)
        self.sbi_video_agent = ExperimentalSBIVideoAgent(device=self.device)
        self.xcep_video_agent = ExperimentalVideoAgent(device=self.device)

    def analyze(self, file_path: str, max_sample_frames: int = 8) -> Dict[str, Any]:
        """
        Executes isolated fusion analysis on a video file.
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        _, ext = os.path.splitext(filename)
        fmt = ext[1:].upper() if ext else "MP4"

        # 1. Run SBI Video Agent
        t0_sbi = time.time()
        try:
            res_sbi = self.sbi_video_agent.analyze(file_path, max_sample_frames=max_sample_frames)
            dt_sbi = round(time.time() - t0_sbi, 4)
            raw_sbi = res_sbi.get("raw_model_score")
            sbi_temp_var = res_sbi.get("temporal_metrics", {}).get("temporal_variance")
            sbi_temp_conflict = res_sbi.get("temporal_metrics", {}).get("temporal_conflict", False)
        except Exception as e:
            logger.error(f"SBI video inference failed: {e}")
            dt_sbi = round(time.time() - t0_sbi, 4)
            res_sbi = {"error": str(e)}
            raw_sbi, sbi_temp_var, sbi_temp_conflict = None, None, False

        # 2. Run Xception Video Agent
        t0_xcep = time.time()
        try:
            res_xcep = self.xcep_video_agent.analyze(file_path, max_sample_frames=max_sample_frames)
            dt_xcep = round(time.time() - t0_xcep, 4)
            raw_xcep = res_xcep.get("raw_model_score")
            xcep_temp_var = res_xcep.get("temporal_metrics", {}).get("temporal_variance")
            xcep_temp_conflict = res_xcep.get("temporal_metrics", {}).get("temporal_conflict", False)
        except Exception as e:
            logger.error(f"Xception video inference failed: {e}")
            dt_xcep = round(time.time() - t0_xcep, 4)
            res_xcep = {"error": str(e)}
            raw_xcep, xcep_temp_var, xcep_temp_conflict = None, None, False

        # Check for non-face / unreadable media
        if res_sbi.get("verdict") == DecisionState.NOT_SUITABLE and res_xcep.get("verdict") == DecisionState.NOT_SUITABLE:
            decision = DecisionResult(
                prediction=DecisionState.NOT_SUITABLE,
                reason="No detectable human faces in sampled video frames.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE],
            )
            forced_binary = "NOT_SUITABLE"
        else:
            # Aggregate temporal variance and conflict
            temp_var = max(filter(lambda x: x is not None, [sbi_temp_var, xcep_temp_var]), default=None)
            temp_conflict = bool(sbi_temp_conflict or xcep_temp_conflict)

            decision = self.fusion_engine.evaluate(
                raw_sbi=raw_sbi,
                raw_xcep=raw_xcep,
                temporal_variance=temp_var,
                temporal_conflict=temp_conflict,
            )
            p_fake = decision.fake_probability if decision.fake_probability is not None else 0.50
            forced_binary = "FAKE" if p_fake >= 0.50 else "REAL"

        total_time = round(time.time() - start_time, 4)

        return {
            "metadata": {
                "file_size_bytes": size,
                "filename": filename,
                "format": fmt,
                "sampled_frames": max_sample_frames,
                "experimental_fusion": "sbi_efficientnet_b4_plus_deepfakebench_xception_video",
            },
            "prediction": decision.prediction,
            "verdict": decision.verdict,
            "forced_binary": forced_binary,
            "confidence": decision.confidence,
            "confidence_score": decision.confidence,
            "confidence_level": decision.confidence_level,
            "fake_probability": decision.fake_probability,
            "real_probability": decision.real_probability,
            "raw_model_score": decision.raw_model_score,
            "uncertainty": decision.uncertainty,
            "evidence": decision.evidence,
            "reason": decision.reason,
            "user_facing_statement": decision.user_facing_statement,
            "sbi_details": {
                "raw_score": raw_sbi,
                "calibrated_prob": res_sbi.get("fake_probability"),
                "verdict": res_sbi.get("verdict"),
                "temporal_variance": sbi_temp_var,
                "latency_sec": dt_sbi,
            },
            "xception_details": {
                "raw_score": raw_xcep,
                "calibrated_prob": res_xcep.get("fake_probability"),
                "verdict": res_xcep.get("verdict"),
                "temporal_variance": xcep_temp_var,
                "latency_sec": dt_xcep,
            },
            "processing_time": total_time,
        }
