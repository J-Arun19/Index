import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.core.calibration import TemperatureScalingCalibrator
from backend.preprocessing.quality_gate import QualityGateResult
from backend.utils.logger import setup_logger

logger = setup_logger("DecisionSystem")


class DecisionState:
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"
    NOT_SUITABLE = "NOT_SUITABLE"

    # Backward compatibility mappings
    LIKELY_REAL = "LIKELY_REAL"
    DEEPFAKE = "DEEPFAKE"


# Standardized, safety-compliant user-facing messages
USER_FACING_STATEMENTS = {
    DecisionState.REAL: "No strong evidence of manipulation was detected across classifier and forensic checks. Notice: Absence of detected manipulation does not guarantee absolute authenticity.",
    DecisionState.LIKELY_REAL: "No strong evidence of manipulation was detected across classifier and forensic checks. This does not prove authenticity.",
    DecisionState.FAKE: "Deepfake detected / high likelihood of manipulation based on forensic anomaly patterns.",
    DecisionState.UNCERTAIN: "Inconclusive / Insufficient evidence: The detector has not established authenticity or manipulation. Independent verification is required.",
    DecisionState.NOT_SUITABLE: "No detectable human face was found in this media. SatyaDrishti's current deepfake detector is designed for facial-media authenticity analysis. Upload clear media containing a visible human face.",
    # Legacy alias compatibility
    "NO MANIPULATION DETECTED": "No strong evidence of manipulation was detected across classifier and forensic checks. Note: This does not prove authenticity.",
    "LIKELY_REAL": "No strong evidence of manipulation was detected across classifier and forensic checks. Note: This does not prove authenticity.",
    "INCONCLUSIVE": "Inconclusive / Insufficient evidence: The detector has not established authenticity or manipulation. Independent verification is required.",
    "SUSPICIOUS - provisional": "Inconclusive evidence: The detector has not established authenticity or manipulation. Independent verification is required.",
}


class PresentationDecisionPolicy:
    """
    Presentation policy ensuring all client surfaces, APIs, explanations,
    history, and PDF reports receive a consistent decision object.
    For supported media, primary verdicts are strictly binary: REAL or FAKE.
    Unsupported media remains NOT_SUITABLE.
    Uncertain internal results map to the higher-probability class with an explicit provisional warning.
    """
    AMBIGUITY_WARNING = (
        "Inconclusive result: Model evidence lies in the ambiguous decision margin. "
        "Presented as a provisional binary prediction; the detector has NOT established authenticity, and manual verification is advised."
    )
    CONFLICT_WARNING = (
        "Disagreement detected between neural classifier and forensic analyzers. "
        "Presented as a provisional binary prediction; authenticity has NOT been established, and manual verification is required."
    )
    QUALITY_WARNING = (
        "Media quality is degraded or facial detail is low. "
        "Presented as a provisional binary prediction; authenticity has NOT been established."
    )
    INFERENCE_FAILURE_WARNING = (
        "Analysis could not be completed due to missing model checkpoints or processing error."
    )

    @classmethod
    def normalize_verdict(
        cls,
        raw_verdict: Optional[str],
        fake_probability: Optional[float] = None,
        is_binary_fallback: bool = False,
        original_verdict: Optional[str] = None
    ) -> str:
        """
        One centralized presentation mapping used consistently by the API,
        frontend, history and PDF reports:
        - REAL / LIKELY_REAL becomes REAL.
        - FAKE / LIKELY_FAKE becomes FAKE.
        - NOT_SUITABLE remains NOT_SUITABLE.
        - UNCERTAIN, INCONCLUSIVE, or provisional binary fallbacks remain strictly UNCERTAIN.
        """
        orig = str(original_verdict or "").strip().upper()
        if is_binary_fallback or orig in ("UNCERTAIN", "INCONCLUSIVE"):
            return DecisionState.UNCERTAIN

        if not raw_verdict:
            return DecisionState.UNCERTAIN

        v = str(raw_verdict).strip().upper()
        if v in ("NOT_SUITABLE", "NOT SUITABLE", "UNSUPPORTED", "UNSUPPORTED MEDIA") or orig in ("NOT_SUITABLE", "NOT SUITABLE"):
            return DecisionState.NOT_SUITABLE
        if v in ("UNCERTAIN", "INCONCLUSIVE"):
            return DecisionState.UNCERTAIN
        if v in ("FAKE", "DEEPFAKE", "LIKELY_FAKE", "LIKELY FAKE", "SUSPICIOUS", "SUSPICIOUS - PROVISIONAL"):
            return DecisionState.FAKE
        if v in ("REAL", "LIKELY_REAL", "LIKELY REAL", "NO MANIPULATION DETECTED"):
            return DecisionState.REAL

        return DecisionState.UNCERTAIN

    @classmethod
    def get_canonical_verdict(
        cls,
        verdict: Optional[str] = None,
        primary_verdict: Optional[str] = None,
        original_verdict: Optional[str] = None,
        is_binary_fallback: bool = False,
        fake_probability: Optional[float] = None,
    ) -> str:
        """Determines the single authoritative canonical verdict."""
        return cls.normalize_verdict(
            raw_verdict=verdict or primary_verdict,
            fake_probability=fake_probability,
            is_binary_fallback=is_binary_fallback,
            original_verdict=original_verdict,
        )

    @classmethod
    def apply_presentation_policy(
        cls,
        decision_state: Optional[str] = None,
        verdict: Optional[str] = None,
        fake_probability: Optional[float] = None,
        has_conflict: bool = False,
        disagreement_detected: bool = False,
        quality_issues: bool = False,
        reason: str = "",
        boundary_anomaly: bool = False,
        temporal_conflict: bool = False,
    ) -> Tuple[str, str, bool, Optional[str]]:
        """
        Maps internal evaluation into canonical user-facing verdict: strictly REAL or FAKE for supported media.
        Unsupported media remains NOT_SUITABLE.
        Preserves raw_state in original_verdict for auditability.
        For internally UNCERTAIN results, maps to the higher-probability class with an explicit provisional warning.
        """
        raw_state = decision_state or verdict or DecisionState.UNCERTAIN

        # 1. Unsupported media remains strictly NOT_SUITABLE
        if raw_state in (DecisionState.NOT_SUITABLE, "NOT_SUITABLE", "NOT SUITABLE", "UNSUPPORTED", "UNSUPPORTED MEDIA"):
            return DecisionState.NOT_SUITABLE, raw_state, False, None

        # 2. Missing probability / inference failure when UNCERTAIN
        if (raw_state == DecisionState.UNCERTAIN or "inference failure" in reason.lower()) and fake_probability is None:
            warning = cls.QUALITY_WARNING if quality_issues else cls.INFERENCE_FAILURE_WARNING
            return DecisionState.UNCERTAIN, raw_state, False, warning

        # 3. Detect internal conflict or quality degradation
        conflict = (
            has_conflict
            or disagreement_detected
            or temporal_conflict
            or boundary_anomaly
            or ("contradictory" in reason.lower())
            or ("disagreement" in reason.lower())
            or ("evidence conflict" in reason.lower())
            or (raw_state in (DecisionState.REAL, "REAL", DecisionState.LIKELY_REAL, "LIKELY_REAL") and "tampering" in reason.lower())
        )
        is_internally_uncertain = (
            raw_state in (DecisionState.UNCERTAIN, "UNCERTAIN", "INCONCLUSIVE")
            or conflict
            or quality_issues
        )

        # 4. Internally UNCERTAIN with valid probability -> map to higher-probability class with provisional warning
        if is_internally_uncertain:
            if fake_probability is not None:
                binary_verdict = DecisionState.FAKE if fake_probability >= 0.50 else DecisionState.REAL
                warning = cls.QUALITY_WARNING if quality_issues else (cls.CONFLICT_WARNING if conflict else cls.AMBIGUITY_WARNING)
                return binary_verdict, raw_state, True, warning
            else:
                warning = cls.QUALITY_WARNING if quality_issues else cls.INFERENCE_FAILURE_WARNING
                return DecisionState.UNCERTAIN, raw_state, False, warning

        # 5. Clean Decisive FAKE / LIKELY_FAKE -> FAKE
        if raw_state in (DecisionState.FAKE, DecisionState.DEEPFAKE, "DEEPFAKE", "FAKE", "LIKELY_FAKE", "LIKELY FAKE"):
            return DecisionState.FAKE, raw_state, False, None

        # 6. Clean Decisive REAL / LIKELY_REAL -> REAL
        if raw_state in (DecisionState.REAL, "REAL", DecisionState.LIKELY_REAL, "LIKELY_REAL", "LIKELY REAL", "NO MANIPULATION DETECTED"):
            return DecisionState.REAL, raw_state, False, None

        # 7. Fallback for any other state
        if fake_probability is not None:
            binary_verdict = DecisionState.FAKE if fake_probability >= 0.50 else DecisionState.REAL
            return binary_verdict, raw_state, True, cls.AMBIGUITY_WARNING

        return DecisionState.UNCERTAIN, raw_state, True, cls.AMBIGUITY_WARNING


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
        visual_analysis: Optional[Dict[str, Any]] = None,
        audio_visual_sync: Optional[Dict[str, Any]] = None,
        is_hard_negative: bool = False,
        analysis_status: str = "Completed",
        media_category: str = "Facial Media",
        recommendation: Optional[str] = None,
        primary_verdict: Optional[str] = None,
        original_verdict: Optional[str] = None,
        is_binary_fallback: Optional[bool] = None,
        reliability_warning: Optional[str] = None,
        influencing_evidence: Optional[List[str]] = None,
        providers_ran: Optional[List[str]] = None,
        approx_cost_usd: Optional[float] = None,
        abstention_code: Optional[str] = None,
    ):
        self.prediction = prediction
        self.original_verdict = original_verdict or prediction
        self.abstention_code = abstention_code
        if self.prediction == DecisionState.NOT_SUITABLE and not self.abstention_code:
            self.abstention_code = "OUT_OF_DOMAIN"
        elif self.prediction == DecisionState.UNCERTAIN and not self.abstention_code:
            r_low = reason.lower()
            if "disagreement" in r_low or "conflict" in r_low or "contradictory" in r_low:
                self.abstention_code = "MODEL_DISAGREEMENT"
            elif "quality" in r_low or "blur" in r_low or "defect" in r_low:
                self.abstention_code = "LOW_QUALITY"
            elif "inference" in r_low or "checkpoint" in r_low or "error" in r_low or "failed" in r_low:
                self.abstention_code = "INFERENCE_FAILURE"
            else:
                self.abstention_code = "INSUFFICIENT_EVIDENCE"

        if primary_verdict is not None:
            p_verdict = PresentationDecisionPolicy.normalize_verdict(primary_verdict, fake_probability)
            orig_v = original_verdict or prediction
            fallback = bool(is_binary_fallback) if is_binary_fallback is not None else (orig_v in (DecisionState.UNCERTAIN, "UNCERTAIN") or p_verdict == DecisionState.UNCERTAIN)
            warning = reliability_warning
        else:
            is_fake_pred = prediction in (DecisionState.FAKE, DecisionState.DEEPFAKE, "DEEPFAKE", "FAKE")
            has_c = (
                ("contradictory" in reason.lower())
                or ("disagreement" in reason.lower())
                or ("evidence conflict" in reason.lower())
                or (not is_fake_pred and "tampering" in reason.lower())
                or ("conflict" in reason.lower() and not is_fake_pred)
            )
            p_verdict, orig_v, fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
                decision_state=prediction,
                fake_probability=fake_probability,
                has_conflict=has_c,
                quality_issues=not quality_passed,
                reason=reason,
            )
            if is_binary_fallback is not None:
                fallback = is_binary_fallback
            if reliability_warning is not None:
                warning = reliability_warning

        self.verdict = prediction
        self.primary_verdict = p_verdict
        if original_verdict is None and orig_v:
            self.original_verdict = orig_v

        self.fake_probability = round(float(fake_probability), 4) if fake_probability is not None else None
        self.real_probability = round(float(real_probability), 4) if real_probability is not None else (
            round(1.0 - self.fake_probability, 4) if self.fake_probability is not None else None
        )
        self.raw_model_score = round(float(raw_model_score), 4) if raw_model_score is not None else None

        # Do not invent confidence scores on UNCERTAIN or NOT_SUITABLE
        if self.prediction in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE):
            self.confidence = None
            self.confidence_level = "Not Performed" if self.prediction == DecisionState.NOT_SUITABLE else "Uncertified"
        else:
            self.confidence = round(float(confidence), 2) if confidence is not None else 85.0
            self.confidence_level = confidence_level if confidence_level in ["High", "Medium", "Low"] else (
                "High" if (self.confidence and self.confidence >= 80) else "Medium"
            )

        self.uncertainty = round(float(uncertainty), 4) if uncertainty is not None else (
            0.0 if self.prediction == DecisionState.NOT_SUITABLE else (
                0.75 if self.prediction == DecisionState.UNCERTAIN else 0.15
            )
        )
        self.evidence = evidence or []
        self.influencing_evidence = influencing_evidence or []
        self.providers_ran = providers_ran or []
        self.approx_cost_usd = approx_cost_usd or (visual_analysis.get("approx_cost_usd", 0.0) if visual_analysis else 0.0)
        self.reason = reason
        self.quality_passed = quality_passed
        self.face_valid = face_valid
        self.forensic_signals = forensic_signals or {}
        self.visual_analysis = visual_analysis or {}
        self.audio_visual_sync = audio_visual_sync or {
            "analyzed": False,
            "reason": "Lip-sync analysis not performed (no compatible AV synchronization model integrated).",
        }
        self.is_hard_negative = is_hard_negative
        self.analysis_status = analysis_status
        self.media_category = media_category
        self.recommendation = recommendation or (
            "Upload a clear media file containing a visible human face."
            if self.prediction == DecisionState.NOT_SUITABLE
            else (
                "Secondary manual verification or provenance inspection is recommended."
                if self.prediction == DecisionState.UNCERTAIN
                else "Forensic safety audit record generated."
            )
        )

        self.is_binary_fallback = fallback
        self.is_estimate = self.prediction == DecisionState.UNCERTAIN
        self.reliability_warning = warning
        self.is_fallback_probability = bool(self.is_binary_fallback and self.original_verdict == DecisionState.UNCERTAIN)
        self.probability_source = "fallback_abstention_baseline" if self.is_fallback_probability else "calibrated_model_inference"

        if not user_facing_statement:
            if self.prediction in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE):
                self.user_facing_statement = USER_FACING_STATEMENTS.get(self.prediction, "Analysis completed.")
            else:
                self.user_facing_statement = USER_FACING_STATEMENTS.get(
                    self.primary_verdict, USER_FACING_STATEMENTS.get(self.prediction, "Analysis completed.")
                )
        else:
            self.user_facing_statement = user_facing_statement

    def to_dict(self) -> Dict[str, Any]:
        canonical_verdict = (
            DecisionState.NOT_SUITABLE if self.prediction == DecisionState.NOT_SUITABLE else
            DecisionState.UNCERTAIN if (self.is_binary_fallback or self.prediction == DecisionState.UNCERTAIN or self.original_verdict == DecisionState.UNCERTAIN) else
            DecisionState.FAKE if self.primary_verdict in (DecisionState.FAKE, "FAKE", DecisionState.DEEPFAKE, "DEEPFAKE") else
            DecisionState.REAL if self.primary_verdict in (DecisionState.REAL, "REAL", DecisionState.LIKELY_REAL, "LIKELY_REAL") else
            DecisionState.UNCERTAIN
        )
        conf_out = None if (self.is_binary_fallback or canonical_verdict in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE)) else self.confidence
        conf_level_out = (
            "Not Performed" if canonical_verdict == DecisionState.NOT_SUITABLE else
            ("Provisional" if self.is_binary_fallback else ("Uncertified" if canonical_verdict == DecisionState.UNCERTAIN else self.confidence_level))
        )
        return {
            "prediction": canonical_verdict,
            "verdict": canonical_verdict,
            "canonical_verdict": canonical_verdict,
            "canonicalVerdict": canonical_verdict,
            "primary_verdict": self.primary_verdict,
            "primaryVerdict": self.primary_verdict,
            "provisional_suggestion": self.primary_verdict if self.is_binary_fallback else None,
            "provisionalSuggestion": self.primary_verdict if self.is_binary_fallback else None,
            "abstention_code": self.abstention_code,
            "abstentionCode": self.abstention_code,
            "original_verdict": self.original_verdict,
            "originalVerdict": self.original_verdict,
            "is_binary_fallback": self.is_binary_fallback,
            "isBinaryFallback": self.is_binary_fallback,
            "is_fallback_probability": self.is_fallback_probability,
            "isFallbackProbability": self.is_fallback_probability,
            "probability_source": self.probability_source,
            "probabilitySource": self.probability_source,
            "is_estimate": self.is_estimate,
            "isEstimate": self.is_estimate,
            "reliability_warning": self.reliability_warning,
            "reliabilityWarning": self.reliability_warning,
            "fake_probability": self.fake_probability,
            "real_probability": self.real_probability,
            "fakeProbability": self.fake_probability,
            "realProbability": self.real_probability,
            "raw_model_score": self.raw_model_score,
            "rawModelScore": self.raw_model_score,
            "confidence": conf_out,
            "confidence_level": conf_level_out,
            "confidenceLevel": conf_level_out,
            "uncertainty": self.uncertainty,
            "evidence": self.evidence,
            "influencing_evidence": self.influencing_evidence,
            "providers_ran": self.providers_ran,
            "approx_cost_usd": self.approx_cost_usd,
            "reason": self.reason,
            "user_facing_statement": self.user_facing_statement,
            "userFacingStatement": self.user_facing_statement,
            "explanation": self.user_facing_statement,
            "quality_passed": self.quality_passed,
            "face_valid": self.face_valid,
            "analysis_status": self.analysis_status,
            "media_category": self.media_category,
            "recommendation": self.recommendation,
            "forensic_signals": self.forensic_signals,
            "visual_analysis": self.visual_analysis,
            "audio_visual_sync": self.audio_visual_sync,
            "is_hard_negative": self.is_hard_negative,
        }


class SafetyDecisionSystem:
    """
    Consistent forensic decision engine for SatyaDrishti.
    Synthesizes:
    - Calibrated specialist neural detector probabilities
    - Multi-provider AI vision analysis & observable anomaly evidence
    - Genuine frequency/compression forensics (FFT, ELA, Edge)
    - Video temporal consistency across frame sequences
    - Quality gate suitability

    Guarantees consistent verdicts across UI, API, explanations, and PDF:
    REAL, FAKE, UNCERTAIN, or NOT_SUITABLE.
    """

    def __init__(self, calibrator: Optional[TemperatureScalingCalibrator] = None):
        self.calibrator = calibrator or TemperatureScalingCalibrator()

    def decide(
        self,
        raw_fake_prob: Optional[float] = None,
        quality_result: Optional[QualityGateResult] = None,
        forensic_result: Optional[Dict[str, Any]] = None,
        metadata_result: Optional[Dict[str, Any]] = None,
        visual_analysis: Optional[Dict[str, Any]] = None,
        audio_visual_sync: Optional[Dict[str, Any]] = None,
        temporal_variance: Optional[float] = None,
        temporal_conflict: bool = False,
        ground_truth: Optional[int] = None,
    ) -> DecisionResult:
        evidence_list: List[str] = []
        influencing_evidence: List[str] = []

        # 1. Media Suitability Validation
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
                    influencing_evidence=["No detectable human face found in image."],
                    reason="No suitable face detected.",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE],
                    quality_passed=False,
                    face_valid=False,
                    forensic_signals={"quality_gate": quality_result.to_dict()},
                    analysis_status="Not performed",
                    media_category="Non-face image",
                    recommendation="Upload a clear image containing a visible human face.",
                )

            if not quality_result.is_suitable:
                rejection = quality_result.rejection_reason or "Input failed media quality checks."
                evidence_list.append(f"Quality Gate: {rejection}")
                for issue in quality_result.issues:
                    evidence_list.append(f"Defect: {issue}")
                influencing_evidence.append(f"Quality Gate Failure: {rejection}")

                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=raw_fake_prob,
                    real_probability=(1.0 - raw_fake_prob) if raw_fake_prob is not None else None,
                    raw_model_score=raw_fake_prob,
                    confidence=None,
                    confidence_level="Low",
                    uncertainty=1.0,
                    evidence=evidence_list,
                    influencing_evidence=influencing_evidence,
                    reason=f"Abstained due to image quality failure: {rejection}",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                    quality_passed=False,
                    face_valid=quality_result.face_detected,
                    forensic_signals={"quality_gate": quality_result.to_dict()},
                    analysis_status="Completed",
                    media_category="Low Quality Facial Media",
                    recommendation="Upload a higher resolution or sharper facial image.",
                )

        # 2. Specialist Model Score & Calibration
        if raw_fake_prob is None:
            raw_fake_prob = 0.50

        calibrated_fake_p = self.calibrator.calibrate_probability(raw_fake_prob)
        calibrated_real_p = 1.0 - calibrated_fake_p
        conf_score, base_uncertainty = self.calibrator.get_confidence_and_uncertainty(calibrated_fake_p)

        # 3. Integrate Multi-Provider Visual Reasoning
        visual_anomalies: List[str] = []
        visual_synthetic_score = 0.50
        consensus_assessment = "LOCAL_ONLY"
        providers_ran: List[str] = []

        if visual_analysis:
            consensus_assessment = visual_analysis.get("consensus_assessment", "INCONCLUSIVE")
            providers_ran = visual_analysis.get(
                "providers_succeeded", visual_analysis.get("providers_attempted", [])
            )
            visual_anomalies = visual_analysis.get("observable_anomalies", [])
            visual_synthetic_score = visual_analysis.get("visual_synthetic_score", 0.50)

            if visual_anomalies:
                evidence_list.append(f"Visual Reasoning: {'; '.join(visual_anomalies[:4])}")
            if visual_analysis.get("compression_vs_synthesis_distinction"):
                evidence_list.append(f"Compression Assessment: {visual_analysis['compression_vs_synthesis_distinction']}")
            for lim in visual_analysis.get("limitations", []):
                evidence_list.append(f"Visual Limitation: {lim}")

        # 4. Integrate Forensic & Metadata Signals
        suspicious_signals = 0
        forensic_signals = {}
        if forensic_result:
            forensic_signals = forensic_result
            evidence_list.extend(forensic_result.get("evidence_items", []))
            suspicious_signals += forensic_result.get("suspicious_signal_count", 0)

        if metadata_result and metadata_result.get("anomalies_detected"):
            tag = metadata_result.get("software_tag") or "Editing signature"
            evidence_list.append(f"Metadata: Editing signature identified ({tag}).")
            suspicious_signals += 1

        # 5. Check Provider Disagreement (Multiple AI Vision Providers Disagree)
        if consensus_assessment == "DISAGREEMENT":
            evidence_list.append(
                "Provider Disagreement: Multi-provider analysis showed conflicting assessments between AI vision models."
            )
            influencing_evidence.append("Multi-provider AI vision disagreement.")
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=None,
                confidence_level="Low",
                uncertainty=0.85,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="Abstained due to provider disagreement: Independent AI vision models reached contradictory verdicts.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                reliability_warning=PresentationDecisionPolicy.CONFLICT_WARNING,
                providers_ran=providers_ran,
            )

        # 5b. Check Specialist Neural Detector Disagreement (e.g. Xception vs ViT)
        if forensic_signals.get("classifier_conflict"):
            evidence_list.append(
                "Classifier Disagreement: Specialist neural detectors produced contradictory assessments "
                "between spatial manipulation (Xception) and generative synthesis (ViT)."
            )
            influencing_evidence.append("Material disagreement between specialist neural classifiers.")
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=None,
                confidence_level="Uncertified",
                uncertainty=max(0.80, base_uncertainty),
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="Abstained due to detector disagreement: Specialist neural detectors produced conflicting predictions.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                reliability_warning=PresentationDecisionPolicy.CONFLICT_WARNING,
                providers_ran=providers_ran,
            )

        # 6. Video Temporal Consistency Check
        # Treat temporal variance as informative evidence rather than an unconditional veto
        if temporal_variance is not None and temporal_variance > 0.22:
            evidence_list.append(
                f"Temporal Note: Elevated frame appearance variance ({temporal_variance:.3f} > 0.22) observed across sequence."
            )

        if temporal_conflict:
            # Check if multi-provider keyframe inspection resolved the conflict
            if consensus_assessment == "FAKE":
                evidence_list.append(
                    "Temporal Conflict Resolved: Sequential contradiction resolved to FAKE by multi-provider keyframe inspection."
                )
                influencing_evidence.append("Multi-provider keyframe inspection confirmed manipulation across sequence.")
            elif consensus_assessment == "REAL" and (calibrated_fake_p < 0.50) and suspicious_signals == 0:
                evidence_list.append(
                    "Temporal Conflict Resolved: Sequential appearance variance resolved to REAL by authentic keyframe consensus."
                )
                influencing_evidence.append("Multi-provider keyframe inspection confirmed natural facial features.")
            else:
                evidence_list.append(
                    "Temporal Contradiction: Significant contradiction across video frames in the sequence."
                )
                influencing_evidence.append("Contradictory frame probabilities across sequence without decisive consensus.")
                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=calibrated_fake_p,
                    real_probability=calibrated_real_p,
                    raw_model_score=raw_fake_prob,
                    confidence=None,
                    confidence_level="Low",
                    uncertainty=max(0.75, base_uncertainty),
                    evidence=evidence_list,
                    influencing_evidence=influencing_evidence,
                    reason="Abstained due to temporal inconsistency across video frames. Sequential appearance is contradictory.",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                    forensic_signals=forensic_signals,
                    visual_analysis=visual_analysis,
                    audio_visual_sync=audio_visual_sync,
                    reliability_warning=PresentationDecisionPolicy.CONFLICT_WARNING,
                    providers_ran=providers_ran,
                )

        # Thresholds & Structural Metrics
        tau_real = self.calibrator.tau_real  # 0.35
        tau_fake = self.calibrator.tau_fake  # 0.65

        model_leans_real = calibrated_fake_p < tau_real
        model_leans_fake = calibrated_fake_p > tau_fake
        has_visual_tampering = (
            consensus_assessment == "FAKE"
            or visual_synthetic_score > tau_fake
            or (consensus_assessment != "REAL" and len(visual_anomalies) > 0 and visual_synthetic_score >= 0.50)
        )

        # Precompute edge and blur metrics for downstream boundary and certification rules
        edge_metrics = forensic_signals.get("edge_analysis", {}) if forensic_signals else {}
        b_disc = edge_metrics.get("boundary_discontinuity", 0.0)
        blur_val = (
            quality_result.blur_score
            if (quality_result and quality_result.blur_score is not None)
            else edge_metrics.get("laplacian_var", 50.0)
        )
        vision_certified_real = (
            consensus_assessment == "REAL"
            and visual_synthetic_score <= tau_real
            and suspicious_signals == 0
        )

        # 7. Multi-Provider Consensus FAKE (Resolves Detector False Negatives)
        # When multi-provider AI vision agrees on FAKE with concrete observable anomalies,
        # or when local forensics strongly corroborates visual tampering:
        if consensus_assessment == "FAKE" and (has_visual_tampering or suspicious_signals >= 1):
            evidence_list.append(
                "Multi-Provider Consensus: Independent AI vision models identified synthetic manipulation anomalies."
            )
            influencing_evidence.extend(visual_anomalies[:3])
            fused_fake_p = max(calibrated_fake_p, visual_synthetic_score, 0.75)
            fused_real_p = round(1.0 - fused_fake_p, 4)
            conf_val = min(95.0, max(conf_score or 75.0, 85.0))

            return DecisionResult(
                prediction=DecisionState.FAKE,
                fake_probability=round(fused_fake_p, 4),
                real_probability=fused_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_val,
                confidence_level="High",
                uncertainty=0.15,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="Strong evidence of synthetic manipulation confirmed by multi-provider AI vision analysis and observable anomalies.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.FAKE],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                providers_ran=providers_ran,
            )

        # 8. Conflict Detection (Classifier says REAL, but verified independent tampering detected)
        ela_tampering = (
            forensic_signals.get("ela_analysis", {}).get("anomaly_detected", False)
            and forensic_signals.get("ela_analysis", {}).get("local_discrepancy", 0.0) > 0.40
        )
        fft_tampering = (
            forensic_signals.get("fft_analysis", {}).get("anomaly_detected", False)
            and forensic_signals.get("fft_analysis", {}).get("fft_score", 0.0) > 0.60
        )
        forensic_confirmed_tampering = (
            (suspicious_signals >= 2 and (forensic_signals.get("composite_forensic_score", 0.0) > 0.55 or forensic_signals.get("tampering_detected", False)))
            or ela_tampering
            or fft_tampering
            or forensic_signals.get("tampering_detected", False)
        )
        if model_leans_real and (consensus_assessment == "FAKE" or forensic_confirmed_tampering):
            evidence_list.append(
                "Evidence Disagreement: Specialist classifier indicated low probability of manipulation, "
                "but independent forensic analyzers or AI vision identified potential tampering."
            )
            influencing_evidence.append("Material disagreement between neural classifier and independent tampering checks.")
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=None,
                confidence_level="Low",
                uncertainty=0.85,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason=(
                    "Abstained due to contradictory evidence: Neural classifier indicated authentic features, "
                    "but independent visual reasoning / forensics identified potential manipulation anomalies."
                ),
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                reliability_warning=PresentationDecisionPolicy.CONFLICT_WARNING,
                providers_ran=providers_ran,
            )

        # 9. Conflict Detection (Classifier says FAKE, but verified AI visual analysis confirms clean natural image)
        if model_leans_fake and consensus_assessment == "REAL" and len(visual_anomalies) == 0 and visual_synthetic_score < 0.20 and suspicious_signals == 0:
            evidence_list.append(
                "Evidence Disagreement: Specialist classifier indicated elevated score, "
                "but visual reasoning confirmed authentic natural features with no generative anomalies."
            )
            influencing_evidence.append("Disagreement between classifier elevated score and clean authentic visual features.")
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=None,
                confidence_level="Low",
                uncertainty=0.75,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="Abstained due to conflicting evidence between neural detector and visual inspection.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                reliability_warning=PresentationDecisionPolicy.CONFLICT_WARNING,
                providers_ran=providers_ran,
            )

        # 10. Boundary Zone / Indeterminate Margin [tau_real, tau_fake]
        if tau_real <= calibrated_fake_p <= tau_fake:
            # Check if multi-provider vision resolves boundary margin
            if consensus_assessment == "FAKE":
                evidence_list.append(
                    f"Boundary Resolution: Probability ({calibrated_fake_p:.2f}) in indeterminate margin "
                    "resolved to FAKE by multi-provider AI vision consensus."
                )
                influencing_evidence.append("Multi-provider AI vision consensus confirmed synthetic manipulation.")
                fused_fake_p = max(calibrated_fake_p, visual_synthetic_score)
                return DecisionResult(
                    prediction=DecisionState.FAKE,
                    fake_probability=round(fused_fake_p, 4),
                    real_probability=round(1.0 - fused_fake_p, 4),
                    raw_model_score=raw_fake_prob,
                    confidence=80.0,
                    confidence_level="High",
                    uncertainty=0.20,
                    evidence=evidence_list,
                    influencing_evidence=influencing_evidence,
                    reason="Synthetic manipulation confirmed by multi-provider AI vision analysis.",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.FAKE],
                    forensic_signals=forensic_signals,
                    visual_analysis=visual_analysis,
                    audio_visual_sync=audio_visual_sync,
                    providers_ran=providers_ran,
                )
            elif (
                consensus_assessment == "REAL"
                and len(visual_anomalies) == 0
                and suspicious_signals == 0
                and (calibrated_fake_p <= 0.48)
                and (b_disc <= 0.85)
                and (temporal_variance is None and "video" not in str(providers_ran).lower())
                and not bool(visual_analysis and visual_analysis.get("limitations"))
                and (blur_val is None or blur_val >= 22.0)
            ):
                evidence_list.append(
                    f"Boundary Resolution: Probability ({calibrated_fake_p:.2f}) in indeterminate margin "
                    "resolved to REAL by multi-provider AI vision consensus and verified boundary continuity."
                )
                influencing_evidence.append("Multi-provider AI vision consensus confirmed natural authenticity.")
                fused_fake_p = min(calibrated_fake_p, visual_synthetic_score if visual_synthetic_score is not None else 0.25)
                return DecisionResult(
                    prediction=DecisionState.REAL,
                    fake_probability=round(fused_fake_p, 4),
                    real_probability=round(1.0 - fused_fake_p, 4),
                    raw_model_score=raw_fake_prob,
                    confidence=80.0,
                    confidence_level="High",
                    uncertainty=0.20,
                    evidence=evidence_list,
                    influencing_evidence=influencing_evidence,
                    reason="Authentic natural features confirmed by multi-provider AI vision analysis.",
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.REAL],
                    forensic_signals=forensic_signals,
                    visual_analysis=visual_analysis,
                    audio_visual_sync=audio_visual_sync,
                    providers_ran=providers_ran,
                )
            else:
                evidence_list.append(
                    f"Boundary Ambiguity: Calibrated probability ({calibrated_fake_p:.2f}) falls "
                    f"within the indeterminate decision margin [{tau_real:.2f}, {tau_fake:.2f}]."
                )
                influencing_evidence.append(f"Calibrated probability ({calibrated_fake_p:.2f}) in indeterminate margin.")
                return DecisionResult(
                    prediction=DecisionState.UNCERTAIN,
                    fake_probability=calibrated_fake_p,
                    real_probability=calibrated_real_p,
                    raw_model_score=raw_fake_prob,
                    confidence=None,
                    confidence_level="Low",
                    uncertainty=max(0.60, base_uncertainty),
                    evidence=evidence_list,
                    influencing_evidence=influencing_evidence,
                    reason=(
                        f"Calibrated probability ({calibrated_fake_p:.2f}) lies in the ambiguous decision margin "
                        f"[{tau_real:.2f} - {tau_fake:.2f}]. Evidence is insufficient to determine authenticity reliably."
                    ),
                    user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                    forensic_signals=forensic_signals,
                    visual_analysis=visual_analysis,
                    audio_visual_sync=audio_visual_sync,
                    reliability_warning=PresentationDecisionPolicy.AMBIGUITY_WARNING,
                    providers_ran=providers_ran,
                )

        # 11. High-Confidence FAKE Decision
        if model_leans_fake or has_visual_tampering:
            evidence_list.append(
                f"Specialist neural detector confirms high synthetic probability ({calibrated_fake_p:.2f})."
            )
            influencing_evidence.append(f"High synthetic probability ({calibrated_fake_p:.2f}) from specialist detector.")
            return DecisionResult(
                prediction=DecisionState.FAKE,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="High" if conf_score >= 80 else "Medium",
                uncertainty=base_uncertainty,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason=f"Strong evidence of synthetic manipulation (calibrated probability: {calibrated_fake_p:.2f}).",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.FAKE],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                providers_ran=providers_ran,
            )

        # 12. Authenticity Certification Policy Check
        # Severe boundary seam or severe blur cannot be certified as authentic unless vision confirms authentic natural features.
        is_video = "video" in str(providers_ran).lower() or temporal_variance is not None
        min_blur_thresh = 10.0 if is_video else 15.0
        boundary_defect = (b_disc > 0.95 and edge_metrics.get("anomaly_detected", False)) and not vision_certified_real
        blur_defect = (blur_val is not None and blur_val < min_blur_thresh) and not vision_certified_real
        if boundary_defect or blur_defect:
            defect_type = "boundary discontinuity" if boundary_defect else f"sharpness ({blur_val:.1f} < {min_blur_thresh})"
            evidence_list.append(
                f"Authenticity Certification Policy: Texture clarity or boundary continuity "
                f"({b_disc:.2f}) is insufficient to reliably certify authenticity."
            )
            influencing_evidence.append(f"Insufficient image quality ({defect_type}).")
            return DecisionResult(
                prediction=DecisionState.UNCERTAIN,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=None,
                confidence_level="Low",
                uncertainty=max(0.65, base_uncertainty),
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason=(
                    "Abstained: Neural score is low, but image texture clarity or boundary "
                    "continuity is insufficient to reliably certify authenticity."
                ),
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.UNCERTAIN],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                reliability_warning=PresentationDecisionPolicy.AMBIGUITY_WARNING,
                providers_ran=providers_ran,
                abstention_code="LOW_QUALITY" if blur_defect else "INSUFFICIENT_EVIDENCE",
            )

        # 13. Decisive REAL or LIKELY_REAL Decision
        if vision_certified_real:
            evidence_list.append("Specialist detector and visual inspection found no significant manipulation artifacts.")
            evidence_list.append("Notice: Multi-provider AI vision certified authentic natural features.")
            influencing_evidence.append("Specialist detector and visual inspection confirmed authentic natural features.")
            return DecisionResult(
                prediction=DecisionState.REAL,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="High" if conf_score >= 80 else "Medium",
                uncertainty=base_uncertainty,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="Authentic natural features confirmed by multi-provider AI vision analysis.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.REAL],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                is_hard_negative=(ground_truth == 1),
                providers_ran=providers_ran,
                is_binary_fallback=False,
            )
        else:
            evidence_list.append("Specialist detector and forensic checks found no significant manipulation artifacts.")
            evidence_list.append("Notice: Absence of detected manipulation does not prove absolute authenticity.")
            influencing_evidence.append("Specialist detector and forensic checks found no significant manipulation artifacts.")
            return DecisionResult(
                prediction=DecisionState.LIKELY_REAL,
                fake_probability=calibrated_fake_p,
                real_probability=calibrated_real_p,
                raw_model_score=raw_fake_prob,
                confidence=conf_score,
                confidence_level="High" if conf_score >= 80 else "Medium",
                uncertainty=base_uncertainty,
                evidence=evidence_list,
                influencing_evidence=influencing_evidence,
                reason="No strong evidence of manipulation was detected across classifier and forensic checks.",
                user_facing_statement=USER_FACING_STATEMENTS[DecisionState.LIKELY_REAL],
                forensic_signals=forensic_signals,
                visual_analysis=visual_analysis,
                audio_visual_sync=audio_visual_sync,
                is_hard_negative=(ground_truth == 1),
                providers_ran=providers_ran,
                is_binary_fallback=False,
                reliability_warning=None,
            )
