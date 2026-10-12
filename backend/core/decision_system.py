import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.core.calibration import TemperatureScalingCalibrator
from backend.preprocessing.quality_gate import QualityGateResult
from backend.utils.logger import setup_logger

logger = setup_logger("DecisionSystem")


class DecisionState:
    FAKE = "FAKE"
    LIKELY_REAL = "LIKELY_REAL"
    UNCERTAIN = "UNCERTAIN"
    NOT_SUITABLE = "NOT_SUITABLE"


# Standardized, safety-compliant user-facing messages
USER_FACING_STATEMENTS = {
    DecisionState.FAKE: "Deepfake detected / high likelihood of manipulation.",
    DecisionState.LIKELY_REAL: "No strong evidence of manipulation was detected. This does not prove authenticity.",
    DecisionState.UNCERTAIN: "Insufficient evidence to determine authenticity reliably. Additional verification is recommended.",
    DecisionState.NOT_SUITABLE: "No detectable human face was found in this image. SatyaDrishti's current image deepfake detector is designed for facial-media authenticity analysis. Upload a clear image containing a visible human face.",
}


class DecisionResult:
    """Structured, immutable output of the safety decision system."""

    def __init__(
        self,
        prediction: str,
        fake_probability: Optional[float] = None,
        real_probability: Optional[float] = None,
        raw_model_score: Optional[float] = None,
        confidence: Optional[float] = None,
        confidence_level: str = "Low",
        uncertainty: Optional[float] = None,
        evidence: Optional[List[str]] = None,
        reason: str = "",
        user_facing_statement: str = "",
        quality_passed: bool = True,
        face_valid: bool = True,
        forensic_signals: Optional[Dict[str, Any]] = None,
        is_hard_negative: bool = False,
        analysis_status: str = "Completed",
        media_category: str = "Facial Media",
        recommendation: Optional[str] = None,
    ):
        self.prediction = prediction
        self.verdict = prediction  # UI compatibility alias
        self.fake_probability = round(float(fake_probability), 4) if fake_probability is not None else None
        self.real_probability = round(float(real_probability), 4) if real_probability is not None else None
        self.raw_model_score = round(float(raw_model_score), 4) if raw_model_score is not None else None
        self.confidence = round(float(confidence), 2) if confidence is not None else None
        self.confidence_level = confidence_level
        self.uncertainty = round(float(uncertainty), 4) if uncertainty is not None else None
        self.evidence = evidence or []
        self.reason = reason
        self.user_facing_statement = user_facing_statement
        self.quality_passed = quality_passed
        self.face_valid = face_valid
        self.forensic_signals = forensic_signals or {}
        self.is_hard_negative = is_hard_negative
        self.analysis_status = analysis_status
        self.media_category = media_category
        self.recommendation = recommendation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "verdict": self.verdict,
            "fake_probability": self.fake_probability,
            "real_probability": self.real_probability,
            "raw_model_score": self.raw_model_score,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "uncertainty": self.uncertainty,
            "evidence": self.evidence,
            "reason": self.reason,
            "user_facing_statement": self.user_facing_statement,
            "quality_passed": self.quality_passed,
            "face_valid": self.face_valid,
            "analysis_status": self.analysis_status,
            "media_category": self.media_category,
            "recommendation": self.recommendation,
            "forensic_signals": self.forensic_signals,
            "is_hard_negative": self.is_hard_negative,
        }


class SafetyDecisionSystem:
    """
    Forensic three-state decision engine for SatyaDrishti.
    Enforces abstention when evidence is weak, ambiguous, low-quality, or contradictory.
    Strictly separates raw classifier probabilities from final user-facing decisions.
    """

    def __init__(self, calibrator: Optional[TemperatureScalingCalibrator] = None):
        self.calibrator = calibrator or TemperatureScalingCalibrator()

    def decide(
        self,
        raw_fake_prob: Optional[float] = None,
        quality_result: Optional[QualityGateResult] = None,
        forensic_result: Optional[Dict[str, Any]] = None,
        metadata_result: Optional[Dict[str, Any]] = None,
        temporal_variance: Optional[float] = None,
        temporal_conflict: bool = False,
        ground_truth: Optional[int] = None,
    ) -> DecisionResult:
        """
        Executes the three-state decision logic over multimodal and forensic inputs.
        """
        evidence_list: List[str] = []

        # 1. Media Eligibility Validation (Separate Suitability from Authenticity)
        if quality_result is not None:
            # Check for UNSUPPORTED media (no human face detected when required)
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

            # Check for LOW QUALITY media (face detected, but blurry or too small)
            if not quality_result.is_suitable:
                rejection = quality_result.rejection_reason or "Input failed media quality checks."
                evidence_list.append(f"Quality Gate: {rejection}")
                for issue in quality_result.issues:
                    evidence_list.append(f"Defect: {issue}")

                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=raw_fake_prob if raw_fake_prob is not None else 0.50,
                    real_probability=(1.0 - raw_fake_prob) if raw_fake_prob is not None else 0.50,
                    raw_model_score=raw_fake_prob if raw_fake_prob is not None else 0.50,
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

        # Ensure raw_fake_prob has valid float for downstream authenticity analysis
        if raw_fake_prob is None:
            raw_fake_prob = 0.50

        # 2. Probability Calibration
        calibrated_fake_p = self.calibrator.calibrate_probability(raw_fake_prob)
        calibrated_real_p = 1.0 - calibrated_fake_p
        conf_score, base_uncertainty = self.calibrator.get_confidence_and_uncertainty(calibrated_fake_p)

        # 3. Gather genuine forensic evidence
        suspicious_signals = 0
        forensic_signals = {}

        if forensic_result:
            forensic_signals = forensic_result
            items = forensic_result.get("evidence_items", [])
            evidence_list.extend(items)
            suspicious_signals += forensic_result.get("suspicious_signal_count", 0)

        # Metadata analysis evidence
        if metadata_result:
            if metadata_result.get("anomalies_detected"):
                tag = metadata_result.get("software_tag") or "Editing signature detected"
                evidence_list.append(f"Metadata: Editing signature identified ({tag}).")
                suspicious_signals += 1

        # Determine confidence level tier
        if conf_score >= 80.0:
            conf_tier = "High"
        elif conf_score >= 60.0:
            conf_tier = "Medium"
        else:
            conf_tier = "Low"

        # 4. Check for Video Temporal Inconsistency
        if temporal_conflict or (temporal_variance is not None and temporal_variance > 0.22):
            evidence_list.append(
                f"Temporal Inconsistency: High variance across frames ({temporal_variance:.3f} > 0.22). "
                "Frame predictions contradict each other."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.75, base_uncertainty),
                evidence=evidence_list,
                reason="Abstained due to temporal inconsistency across video frames. Sequential evidence is contradictory.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 5. Model / Forensic Disagreement Check (CRITICAL FALSE-NEGATIVE PROTECTION)
        # If neural network predicts REAL (calibrated_fake_p < tau_real), but forensic analyzers detected tampering
        tau_real = self.calibrator.tau_real  # default 0.35
        tau_fake = self.calibrator.tau_fake  # default 0.65

        model_leans_real = calibrated_fake_p < tau_real
        model_leans_fake = calibrated_fake_p > tau_fake

        if model_leans_real and suspicious_signals >= 1:
            evidence_list.append(
                "Evidence Disagreement: Primary classifier indicated low probability of manipulation, "
                f"but independent forensic analyzers identified {suspicious_signals} suspicious signal(s)."
            )
            escalated_uncertainty = max(0.70, base_uncertainty + 0.30)
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=min(1.0, escalated_uncertainty),
                evidence=evidence_list,
                reason=(
                    "Abstained due to contradictory evidence: Classifier indicates authentic visual features, "
                    "but independent forensic analysis detected tampering/splicing artifacts."
                ),
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 6. Ambiguous Margin / Boundary Check
        # Sample falls inside the uncalibrated/calibrated boundary zone [tau_real, tau_fake]
        if tau_real <= calibrated_fake_p <= tau_fake:
            evidence_list.append(
                f"Boundary Ambiguity: Calibrated probability ({calibrated_fake_p:.2f}) falls "
                f"within the indeterminate decision margin [{tau_real:.2f}, {tau_fake:.2f}]."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level=conf_tier,
                uncertainty=max(0.60, base_uncertainty),
                evidence=evidence_list,
                reason=(
                    f"Calibrated probability ({calibrated_fake_p:.2f}) lies in the ambiguous decision boundary "
                    f"[{tau_real:.2f} - {tau_fake:.2f}]. Evidence is inconclusive."
                ),
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 7. Low Confidence Abstention
        if conf_score < self.calibrator.min_confidence:
            evidence_list.append(
                f"Low Confidence: Calibrated confidence ({conf_score}%) is below the operational threshold ({self.calibrator.min_confidence}%)."
            )
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.65, base_uncertainty),
                evidence=evidence_list,
                reason=f"Classifier confidence ({conf_score}%) is insufficient to deliver a reliable forensic verdict.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
            )

        # 8. High-Confidence FAKE Decision
        if model_leans_fake:
            evidence_list.append(
                f"High synthetic probability ({calibrated_fake_p:.2f}) confirmed by neural representation analysis."
            )
            return DecisionResult(
                prediction=DecisionState.FAKE,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level=conf_tier,
                uncertainty=base_uncertainty,
                evidence=evidence_list,
                reason=f"Strong evidence of synthetic manipulation (calibrated probability: {calibrated_fake_p:.2f}).",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.FAKE],
                forensic_signals=forensic_signals,
            )

        # 8b. Authenticity Certification Policy Check
        # To certify LIKELY_REAL, evidence must be sufficient to certify authenticity.
        # If the image exhibits borderline sharpness (blur_score < 38.0) or elevated boundary
        # discontinuity (> 0.35), micro-manipulation artifacts may be obscured by smoothing/compression.
        # Under safety policy, weak negative evidence under low texture must abstain to UNCERTAIN.
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
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="Low",
                uncertainty=max(0.65, base_uncertainty),
                evidence=evidence_list,
                reason=(
                    "Abstained: Neural score is low, but image texture clarity or boundary continuity is insufficient "
                    "to reliably certify authenticity without risking a false-negative."
                ),
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                is_hard_negative=(ground_truth == 1),
            )

        # 9. High-Confidence LIKELY_REAL Decision
        # Reaching here guarantees:
        # - calibrated_fake_p < tau_real
        # - conf_score >= min_confidence
        # - zero conflicting forensic signals
        # - image quality, texture, and face validity passed
        evidence_list.append("Classifier and forensic spectral/compression checks show no significant tampering.")
        evidence_list.append("Notice: Absence of detected manipulation does not prove absolute authenticity.")

        # Track if this sample was ground truth fake (a hard negative!)
        is_hard_negative = (ground_truth == 1)

        return DecisionResult(
            prediction=DecisionState.LIKELY_REAL,
            fake_probability=calibrated_fake_p,
            real_probability=calibrated_real_p,
            raw_model_score=raw_fake_prob,
            confidence=conf_score,
            confidence_level=conf_tier,
            uncertainty=base_uncertainty,
            evidence=evidence_list,
            reason="No significant evidence of manipulation detected. Forensic patterns consistent with natural imagery.",
            user_facing_statement=USER_FACING_STATEMENTS[DecisionState.LIKELY_REAL],
            forensic_signals=forensic_signals,
            is_hard_negative=is_hard_negative,
        )
