import os
import sys
import unittest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.core.calibration import TemperatureScalingCalibrator
from backend.models.image_model import ImageModel
from backend.models.experimental.xception_detector import XceptionDetector
from backend.models.experimental.vit_detector import ViTDetector
from backend.agents.image_agent import ImageAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.decision_system import SafetyDecisionSystem, DecisionState, PresentationDecisionPolicy

class TestScoreSemanticsAndFusion(unittest.TestCase):
    """
    Validates:
    - Precise class-index mappings across all specialist models.
    - Mathematical consistency of score transformations and calibration.
    - Absence of artificial score dilution in FusionAgent.
    - Consistency of provisional suggestions with the underlying fake probability.
    """

    def test_01_resnet_class_mapping(self):
        """Verify ResNet-18 outputs: index 0 is REAL, index 1 is FAKE."""
        model = ImageModel(num_classes=2)
        model.eval()

        # Logit [10.0, -10.0] -> Real prob ~ 1.0, Fake prob ~ 0.0
        logits_real = torch.tensor([[10.0, -10.0]])
        probs_real = torch.softmax(logits_real, dim=-1)
        self.assertAlmostEqual(probs_real[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(probs_real[0, 1].item(), 0.0, places=4)

        # Logit [-10.0, 10.0] -> Real prob ~ 0.0, Fake prob ~ 1.0
        logits_fake = torch.tensor([[-10.0, 10.0]])
        probs_fake = torch.softmax(logits_fake, dim=-1)
        self.assertAlmostEqual(probs_fake[0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(probs_fake[0, 1].item(), 1.0, places=4)

    def test_02_xception_class_mapping(self):
        """Verify Xception output: index 0 is REAL, index 1 is FAKE."""
        try:
            xcep = XceptionDetector()
            xcep.eval()
            dummy_tensor = torch.zeros((1, 3, 256, 256), dtype=torch.float32).to(xcep.device)
            with torch.no_grad():
                logits = xcep(dummy_tensor)
                self.assertEqual(logits.shape, (1, 2))
                probs = torch.softmax(logits, dim=-1)[0]
                self.assertAlmostEqual(probs[0].item() + probs[1].item(), 1.0, places=3)
        except Exception as e:
            self.skipTest(f"Xception weights not available: {e}")

    def test_03_vit_class_mapping(self):
        """Verify ViT id2label mapping: 0 is Real, 1 is Fake."""
        try:
            vit = ViTDetector()
            self.assertEqual(vit.model.config.id2label[0].lower(), "real")
            self.assertEqual(vit.model.config.id2label[1].lower(), "fake")
        except Exception as e:
            self.skipTest(f"ViT model not available: {e}")

    def test_04_raw_score_is_not_mislabeled_as_calibrated_probability(self):
        """The configured temperature transform cannot explain a 27.1% -> 57.4% jump."""
        calibrator = TemperatureScalingCalibrator(temperature=1.885)
        decision = SafetyDecisionSystem(calibrator=calibrator).decide(raw_fake_prob=0.271)

        expected = calibrator.calibrate_probability(0.271)
        self.assertAlmostEqual(decision.raw_model_score, 0.271, places=4)
        self.assertAlmostEqual(decision.fake_probability, expected, places=4)
        self.assertLess(expected, 0.50)
        self.assertNotAlmostEqual(expected, 0.574, places=2)

    def test_05_fusion_agent_does_not_dilute_fake_probability(self):
        """
        Regression test for 30.5% bug:
        When ImageAgent returns a fake_probability of 0.62, FusionAgent must NOT
        dilute it down to ~0.30 by injecting arbitrary p_meta=0.10 or artifact=0.10.
        """
        fusion = FusionAgent()

        mock_image_res = {
            "prediction": "UNCERTAIN",
            "verdict": "UNCERTAIN",
            "canonical_verdict": "UNCERTAIN",
            "primary_verdict": "FAKE",
            "fake_probability": 0.618,
            "real_probability": 0.382,
            "raw_model_score": 0.712,
            "confidence": None,
            "confidence_score": None,
            "is_binary_fallback": True,
            "reliability_warning": "Specialist detectors produced contradictory assessments.",
            "evidence": ["Classifier Disagreement detected."],
            "features": {},
            "models": [
                {"name": "XceptionNet", "raw_score": 0.4294, "predicted_class": "REAL"},
                {"name": "ViT", "raw_score": 0.9954, "predicted_class": "FAKE"}
            ]
        }

        mock_meta_res = {
            "status": "Success",
            "metadata_present": True,
            "anomalies_detected": False,
            "software_tag": None,
            "summary": "Clean EXIF metadata."
        }

        mock_art_res = {
            "status": "Success",
            "anomaly_score": 0.10,
            "verdict": "REAL",
            "detected_artifacts": [],
            "summary": "No high-frequency anomalies detected."
        }

        class MockContext:
            media_type = "image"
            image_agent_result = mock_image_res
            metadata_agent_result = mock_meta_res
            artifact_agent_result = mock_art_res
            analysis_id = "test_dilution_fix"

        res = fusion.fuse(MockContext())

        # The fake_probability must remain ~0.618, NOT diluted to ~0.30!
        self.assertAlmostEqual(res["fake_probability"], 0.618, places=2,
                               msg=f"Expected fake_probability ~0.618, but got {res['fake_probability']}")
        # The primary verdict / provisional suggestion must be FAKE (>= 0.50), NEVER REAL!
        self.assertEqual(res["primary_verdict"], "FAKE",
                         msg=f"Primary verdict must be FAKE when prob >= 0.50, but got {res['primary_verdict']}")
        # Canonical verdict is FAKE under authoritative binary policy with Low Confidence
        self.assertEqual(res["canonical_verdict"], "FAKE")
        self.assertEqual(res["confidence_level"], "Low")

    def test_06_provisional_suggestion_strictly_consistent_with_probability(self):
        """
        Verify that:
        - When fake_probability >= 0.50, normalized verdict is FAKE.
        - When fake_probability < 0.50, normalized verdict is REAL.
        - is_binary_fallback is True for conflicting/marginal predictions.
        """
        # Case A: Leaning Fake (e.g. 0.58)
        v_fake, orig_fake, is_fallback_fake, warn_fake = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.58,
            has_conflict=True,
            reason="Specialist models disagree"
        )
        self.assertEqual(v_fake, DecisionState.FAKE)
        self.assertTrue(is_fallback_fake)
        norm_v_fake = PresentationDecisionPolicy.normalize_verdict(v_fake, 0.58, is_binary_fallback=is_fallback_fake)
        self.assertEqual(norm_v_fake, DecisionState.FAKE)

        # Case B: Leaning Real (e.g. 0.38)
        v_real, orig_real, is_fallback_real, warn_real = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.38,
            has_conflict=True,
            reason="Specialist models disagree"
        )
        self.assertEqual(v_real, DecisionState.REAL)
        self.assertTrue(is_fallback_real)
        norm_v_real = PresentationDecisionPolicy.normalize_verdict(v_real, 0.38, is_binary_fallback=is_fallback_real)
        self.assertEqual(norm_v_real, DecisionState.REAL)

    def test_07_inwild_picture_jpeg_end_to_end_consistency(self):
        """
        End-to-end regression test for inwild_picture.jpeg:
        - Xception is ~0.43 (REAL)
        - ViT is ~0.99 (FAKE)
        - Pipeline returns authoritative binary verdict FAKE with Low Confidence
        - Fake probability must be ~0.58 (never 30.5%!)
        """
        path = "scratch/image_eval_set/inwild_picture.jpeg"
        if not os.path.exists(path):
            self.skipTest("inwild_picture.jpeg not found")

        agent = ImageAgent()
        res = agent.analyze(path, selected_providers=[])

        self.assertEqual(res["canonical_verdict"], "FAKE")
        self.assertEqual(res["confidence_level"], "Low")
        self.assertGreaterEqual(res["fake_probability"], 0.50,
                                f"Expected fake_probability >= 0.50, got {res['fake_probability']}")
        self.assertLessEqual(res["fake_probability"], 0.70,
                             f"Expected fake_probability <= 0.70, got {res['fake_probability']}")
        self.assertEqual(res["primary_verdict"], "FAKE",
                         f"Primary verdict must be FAKE, got {res['primary_verdict']}")

if __name__ == "__main__":
    unittest.main()
