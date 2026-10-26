import unittest
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from backend.core.decision_system import (
    SafetyDecisionSystem,
    DecisionState,
    DecisionResult,
    PresentationDecisionPolicy,
)
from backend.preprocessing.quality_gate import QualityGateResult
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.orchestrator import AnalysisContext, Orchestrator


class TestCriticalFalseNegativeSafety(unittest.TestCase):
    """
    Mandatory Regression Tests for SatyaDrishti False-Negative Protection:
    1. A known fake incorrectly predicted as REAL.
    2. An uncertain prediction with a low fake probability.
    3. A detector or agent failure.
    4. Missing or invalid probability.
    5. Unsupported media.
    6. Conflicting model outputs.
    7. A valid decisive REAL prediction.
    8. A valid decisive FAKE prediction.
    """

    def setUp(self):
        self.decision_system = SafetyDecisionSystem()

    # -------------------------------------------------------------------------
    # 1. A known fake with low classifier score but forensic tampering detected
    # -------------------------------------------------------------------------
    def test_01_known_fake_with_forensic_tampering_presented_as_fake(self):
        """
        Known fakes like deepy.jpeg or professional.jpeg produce low classifier fake prob (e.g. 0.10 - 0.24),
        but independent forensic analyzers detect tampering.
        CRITICAL RULE: Must be presented as FAKE with conflict warning, NEVER REAL.
        """
        raw_fake_p = 0.1025
        forensic_result = {
            "suspicious_signal_count": 2,
            "evidence_items": ["FFT: High-frequency spectral anomaly detected.", "ELA: Compression grid discrepancy."],
            "tampering_detected": True,
        }

        decision = self.decision_system.decide(
            raw_fake_prob=raw_fake_p,
            forensic_result=forensic_result,
        )

        # Internal safety engine must abstain due to contradictory evidence
        self.assertEqual(decision.prediction, DecisionState.UNCERTAIN)
        self.assertIn("contradictory", decision.reason.lower())

        # Presentation layer maps to binary provisional fallback (REAL since p=0.25 < 0.50) with visible conflict disclosure
        d = decision.to_dict()
        self.assertEqual(d["primary_verdict"], "REAL")
        self.assertEqual(d["original_verdict"], DecisionState.UNCERTAIN)
        self.assertTrue(d["is_binary_fallback"])
        self.assertEqual(d["reliability_warning"], PresentationDecisionPolicy.CONFLICT_WARNING)
        self.assertIn("Disagreement detected", d["reliability_warning"])

    # -------------------------------------------------------------------------
    # 2. An uncertain prediction with a low fake probability
    # -------------------------------------------------------------------------
    def test_02_uncertain_prediction_with_low_probability_in_boundary(self):
        """
        Sample in indeterminate boundary [0.35, 0.65] (e.g. p=0.42) with no tampering or conflict.
        Under the binary presentation requirement, internal verdict UNCERTAIN maps to higher probability
        binary class ('REAL' for p=0.42) with is_fallback=True and visible warning.
        """
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.42,
            has_conflict=False,
            disagreement_detected=False,
            quality_issues=False,
        )
        self.assertEqual(primary_v, "REAL")
        self.assertEqual(orig_v, DecisionState.UNCERTAIN)
        self.assertTrue(is_fallback)
        self.assertIn("provisional binary prediction", warning.lower())

    # -------------------------------------------------------------------------
    # 3. A detector or agent failure
    # -------------------------------------------------------------------------
    def test_03_agent_failure_not_treated_as_authentic(self):
        """
        Failure of an agent (e.g. ArtifactConsistencyAgent) must NEVER default to 'REAL'
        or be interpreted as evidence of authenticity.
        """
        # Simulate agent failure payload
        failure_payload = {
            "status": "Failed",
            "anomaly_score": None,
            "verdict": "UNKNOWN",
            "summary": "Artifact analysis failed (recovered)."
        }
        self.assertNotEqual(failure_payload.get("verdict"), "REAL")
        self.assertIsNone(failure_payload.get("anomaly_score"))

        # Video decode failure test
        v_agent = VideoAgent()
        failure_res = v_agent._empty_failure_response(
            "corrupt.mp4", 1024, "MP4", "Decoding error", 0.05
        )
        self.assertEqual(failure_res["primary_verdict"], DecisionState.UNCERTAIN)
        self.assertNotEqual(failure_res["primary_verdict"], "REAL")
        self.assertIsNone(failure_res["fake_probability"])

    # -------------------------------------------------------------------------
    # 4. Missing or invalid probability
    # -------------------------------------------------------------------------
    def test_04_missing_or_invalid_probability_does_not_fabricate(self):
        """
        If fake_probability is None (e.g. inference could not be completed or quality failed),
        do NOT fabricate 0.50 or force REAL/FAKE. Return UNCERTAIN.
        """
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=None,
            quality_issues=True,
        )
        self.assertEqual(primary_v, "UNCERTAIN")
        self.assertEqual(orig_v, DecisionState.UNCERTAIN)
        self.assertFalse(is_fallback)
        self.assertIsNotNone(warning)
        self.assertIn("quality is degraded", warning.lower())

    # -------------------------------------------------------------------------
    # 5. Unsupported media (non-face)
    # -------------------------------------------------------------------------
    def test_05_unsupported_media_remains_not_suitable(self):
        """
        Media without a detectable face (document, logo, landscape) remains strictly
        NOT_SUITABLE throughout the decision system and presentation layer.
        """
        q_result = QualityGateResult(
            is_suitable=False,
            face_detected=False,
            eligibility="UNSUPPORTED",
            rejection_reason="No detectable human face found in image."
        )

        decision = self.decision_system.decide(
            raw_fake_prob=None,
            quality_result=q_result,
        )

        self.assertEqual(decision.prediction, DecisionState.NOT_SUITABLE)
        self.assertEqual(decision.primary_verdict, DecisionState.NOT_SUITABLE)
        self.assertEqual(decision.original_verdict, DecisionState.NOT_SUITABLE)
        self.assertFalse(decision.is_binary_fallback)
        self.assertIsNone(decision.fake_probability)
        self.assertIsNone(decision.confidence)

    # -------------------------------------------------------------------------
    # 6. Conflicting model outputs
    # -------------------------------------------------------------------------
    def test_06_conflicting_model_outputs_presentation(self):
        """
        When primary model and secondary forensic/metadata agents conflict,
        disagreement is flagged and the presentation policy maps to the provisional binary class
        with visible conflict disclosure.
        """
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.25,
            disagreement_detected=True,
            reason="Contradictory evidence: Classifier indicates authentic features, but forensic analysis detected manipulation signatures."
        )
        self.assertEqual(primary_v, "REAL")
        self.assertEqual(orig_v, DecisionState.UNCERTAIN)
        self.assertTrue(is_fallback)
        self.assertIn("Disagreement detected", warning)

    # -------------------------------------------------------------------------
    # 7. Valid decisive REAL prediction
    # -------------------------------------------------------------------------
    def test_07_valid_decisive_real_prediction(self):
        """
        When no anomalies are detected, raw decision is LIKELY_REAL, which maps directly
        to REAL in the presentation layer. Explicit REAL remains REAL.
        """
        decision = self.decision_system.decide(
            raw_fake_prob=0.04,
            forensic_result={"suspicious_signal_count": 0, "evidence_items": []},
        )
        self.assertEqual(decision.prediction, DecisionState.LIKELY_REAL)
        d = decision.to_dict()
        self.assertEqual(d["primary_verdict"], DecisionState.REAL)
        self.assertEqual(d["verdict"], DecisionState.REAL)
        self.assertEqual(d["original_verdict"], DecisionState.LIKELY_REAL)
        self.assertFalse(d["is_binary_fallback"])

        # Explicit certified REAL remains REAL
        explicit_real = DecisionResult(
            prediction=DecisionState.REAL,
            fake_probability=0.04,
            confidence=95.0,
            quality_passed=True,
            face_valid=True,
        )
        d_real = explicit_real.to_dict()
        self.assertEqual(d_real["primary_verdict"], "REAL")
        self.assertEqual(d_real["verdict"], "REAL")
        self.assertEqual(d_real["original_verdict"], DecisionState.REAL)
        self.assertFalse(d_real["is_binary_fallback"])

    # -------------------------------------------------------------------------
    # 8. Valid decisive FAKE prediction
    # -------------------------------------------------------------------------
    def test_08_valid_decisive_fake_prediction(self):
        """
        A high-confidence synthetic sample (calibrated p > 0.65)
        outputs FAKE as primary verdict without fallback warning.
        """
        decision = self.decision_system.decide(
            raw_fake_prob=0.95,
            forensic_result={"suspicious_signal_count": 3, "evidence_items": ["Synthetic artifacts present."]},
        )
        self.assertEqual(decision.prediction, DecisionState.FAKE)
        d = decision.to_dict()
        self.assertEqual(d["primary_verdict"], "FAKE")
        self.assertEqual(d["original_verdict"], DecisionState.FAKE)
        self.assertFalse(d["is_binary_fallback"])
        self.assertIsNone(d["reliability_warning"])


if __name__ == "__main__":
    unittest.main()
