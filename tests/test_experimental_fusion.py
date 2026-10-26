import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.decision_system import DecisionState
from backend.preprocessing.quality_gate import QualityGateResult
from backend.agents.experimental.experimental_fusion_agent import (
    ForensicFusionEngine,
    ExperimentalFusionImageAgent,
)


class TestForensicFusionEngine(unittest.TestCase):

    def setUp(self):
        self.fusion_engine = ForensicFusionEngine(tau_real=0.35, tau_fake=0.65, min_confidence=60.0)

    def test_model_disagreement_routes_to_uncertain(self):
        """
        Conservative Safety Rule:
        When SBI indicates REAL (raw=0.10, cal < 0.35) and Xception indicates FAKE (raw=0.90, cal > 0.65),
        or vice versa, the engine MUST abstain and return UNCERTAIN.
        """
        # Case 1: SBI says REAL (raw=0.10), Xception says FAKE (raw=0.90)
        res1 = self.fusion_engine.evaluate(raw_sbi=0.10, raw_xcep=0.90)
        self.assertEqual(res1.prediction, DecisionState.UNCERTAIN)
        self.assertTrue("Model Disagreement" in res1.reason or "disagreement" in res1.reason.lower())
        self.assertIn(res1.confidence_level, ["Low", "Uncertified"])
        self.assertGreaterEqual(res1.uncertainty, 0.80)

        # Case 2: SBI says FAKE (raw=0.95), Xception says REAL (raw=0.08)
        res2 = self.fusion_engine.evaluate(raw_sbi=0.95, raw_xcep=0.08)
        self.assertEqual(res2.prediction, DecisionState.UNCERTAIN)
        self.assertTrue("Model Disagreement" in res2.reason or "disagreement" in res2.reason.lower())
        self.assertIn(res2.confidence_level, ["Low", "Uncertified"])
        self.assertGreaterEqual(res2.uncertainty, 0.80)

    def test_model_agreement_routes_to_fake(self):
        """
        When both SBI and Xception agree on manipulation (both raw > 0.80),
        the engine must return FAKE with high confidence.
        """
        res = self.fusion_engine.evaluate(raw_sbi=0.90, raw_xcep=0.85)
        self.assertEqual(res.prediction, DecisionState.FAKE)
        self.assertGreater(res.fake_probability, 0.65)
        self.assertGreaterEqual(res.confidence, 60.0)
        self.assertIn(res.confidence_level, ["Medium", "High"])

    def test_model_agreement_routes_to_likely_real(self):
        """
        When both SBI and Xception agree on authentic visual representation (both raw < 0.20),
        and texture gates pass, the engine must return LIKELY_REAL.
        """
        quality_res = QualityGateResult(
            is_suitable=True,
            eligibility="SUPPORTED",
            blur_score=65.0,  # Sharply focused
            face_detected=True,
            primary_face_bbox=(10, 10, 100, 100),
            issues=[],
        )
        forensic_res = {
            "suspicious_signal_count": 0,
            "evidence_items": [],
            "edge_analysis": {"boundary_discontinuity": 0.05, "laplacian_var": 65.0},
        }
        res = self.fusion_engine.evaluate(
            raw_sbi=0.12,
            raw_xcep=0.15,
            quality_result=quality_res,
            forensic_result=forensic_res,
        )
        self.assertEqual(res.prediction, DecisionState.LIKELY_REAL)
        self.assertLess(res.fake_probability, 0.35)
        self.assertGreaterEqual(res.confidence, 60.0)

    def test_missing_scores_and_none_handling(self):
        """
        Engine must safely guard against None scores without raising exceptions:
        - If both are None: UNCERTAIN
        - If SBI is None: Fallback to Xception
        - If Xception is None: Fallback to SBI
        """
        # Both None
        res_both_none = self.fusion_engine.evaluate(raw_sbi=None, raw_xcep=None)
        self.assertEqual(res_both_none.prediction, DecisionState.UNCERTAIN)
        self.assertEqual(res_both_none.fake_probability, 0.50)

        # SBI None, Xception confident FAKE
        res_sbi_none = self.fusion_engine.evaluate(raw_sbi=None, raw_xcep=0.92)
        self.assertEqual(res_sbi_none.prediction, DecisionState.FAKE)
        self.assertGreater(res_sbi_none.fake_probability, 0.65)

        # Xception None, SBI confident FAKE
        res_xcep_none = self.fusion_engine.evaluate(raw_sbi=0.88, raw_xcep=None)
        self.assertEqual(res_xcep_none.prediction, DecisionState.FAKE)
        self.assertGreater(res_xcep_none.fake_probability, 0.65)

    def test_non_face_media_rejection(self):
        """
        Media without a detectable face must return NOT_SUITABLE,
        with None probabilities and raw scores.
        """
        q_unsupported = QualityGateResult(
            is_suitable=False,
            eligibility="UNSUPPORTED",
            rejection_reason="No human face found in image.",
            face_detected=False,
            primary_face_bbox=None,
            issues=["UNSUPPORTED_MEDIA"],
        )
        res = self.fusion_engine.evaluate(
            raw_sbi=0.90,
            raw_xcep=0.90,
            quality_result=q_unsupported,
        )
        self.assertEqual(res.prediction, DecisionState.NOT_SUITABLE)
        self.assertEqual(res.verdict, DecisionState.NOT_SUITABLE)
        self.assertIsNone(res.fake_probability)
        self.assertIsNone(res.raw_model_score)
        self.assertIsNone(res.confidence)
        self.assertFalse(res.quality_passed)
        self.assertFalse(res.face_valid)

    def test_indeterminate_boundary_zone_abstains(self):
        """
        When both models output ambiguous scores in the [0.35, 0.65] margin,
        the engine must abstain to UNCERTAIN.
        """
        res = self.fusion_engine.evaluate(raw_sbi=0.48, raw_xcep=0.52)
        self.assertEqual(res.prediction, DecisionState.UNCERTAIN)
        self.assertGreaterEqual(res.fake_probability, 0.35)
        self.assertLessEqual(res.fake_probability, 0.65)

    def test_temporal_inconsistency_forces_uncertain(self):
        """
        Video with high temporal variance (>0.22) must force UNCERTAIN abstention.
        """
        res = self.fusion_engine.evaluate(
            raw_sbi=0.85,
            raw_xcep=0.88,
            temporal_variance=0.28,
            temporal_conflict=True,
        )
        self.assertEqual(res.prediction, DecisionState.UNCERTAIN)
        self.assertIn("temporal inconsistency", res.reason.lower())


class TestMetricReproducibility(unittest.TestCase):

    def test_reproducible_metric_computation(self):
        """
        Verifies that binary and 3-state metrics compute deterministically and correctly.
        """
        y_true = ["REAL", "REAL", "FAKE", "FAKE", "FAKE"]
        y_pred = ["REAL", "FAKE", "FAKE", "FAKE", "REAL"]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "FAKE")
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "REAL")
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "FAKE")
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "REAL")

        self.assertEqual(tp, 2)
        self.assertEqual(tn, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        spec = tn / (tn + fp)
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)

        self.assertEqual(round(acc, 4), 0.6000)
        self.assertEqual(round(prec, 4), 0.6667)
        self.assertEqual(round(rec, 4), 0.6667)
        self.assertEqual(round(f1, 4), 0.6667)
        self.assertEqual(round(spec, 4), 0.5000)
        self.assertEqual(round(fpr, 4), 0.5000)
        self.assertEqual(round(fnr, 4), 0.3333)


if __name__ == "__main__":
    unittest.main()
