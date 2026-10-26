import unittest
import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from backend.core.decision_system import (
    SafetyDecisionSystem,
    DecisionState,
    DecisionResult,
    PresentationDecisionPolicy,
)
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.nlp_response import generate_nlp_response


class TestPresentationFallbackPolicy(unittest.TestCase):
    """
    Comprehensive tests for the Presentation Decision Policy:
    1. Confident REAL prediction
    2. Confident FAKE prediction
    3. UNCERTAIN internal result with valid probability -> REAL/FAKE fallback
    4. Conflicting model evidence with warning displayed
    5. NOT_SUITABLE media remaining NOT_SUITABLE
    6. Missing checkpoints and inference failures producing an explicit error (no fabricated results)
    7. Image and video responses rendering correctly
    8. Existing analysis history remaining compatible
    """

    def setUp(self):
        self.decision_system = SafetyDecisionSystem()
        self.policy = PresentationDecisionPolicy()

    # -------------------------------------------------------------------------
    # 1. LIKELY_REAL maps to REAL for presentation, preserving original_verdict
    # -------------------------------------------------------------------------
    def test_01_likely_real_maps_to_user_facing_real(self):
        """
        Internal prediction LIKELY_REAL is mapped directly to REAL in the presentation layer,
        preserving original_verdict=LIKELY_REAL for auditability.
        """
        primary_verdict, original_verdict, is_fallback, warning = (
            PresentationDecisionPolicy.apply_presentation_policy(
                decision_state=DecisionState.LIKELY_REAL,
                fake_probability=0.08,
                has_conflict=False,
                quality_issues=False,
            )
        )
        self.assertEqual(primary_verdict, DecisionState.REAL)
        self.assertEqual(original_verdict, DecisionState.LIKELY_REAL)
        self.assertFalse(is_fallback)
        self.assertIsNone(warning)

        # Verify DecisionResult integration
        result = DecisionResult(
            prediction=DecisionState.LIKELY_REAL,
            fake_probability=0.08,
            confidence=92.0,
            quality_passed=True,
            face_valid=True,
        )
        d = result.to_dict()
        self.assertEqual(d["primary_verdict"], DecisionState.REAL)
        self.assertEqual(d["verdict"], DecisionState.REAL)
        self.assertEqual(d["original_verdict"], DecisionState.LIKELY_REAL)
        self.assertFalse(d["is_binary_fallback"])
        self.assertIn("No strong evidence of manipulation", d["user_facing_statement"])

    def test_01b_explicit_certified_real_remains_real(self):
        """Explicitly certified REAL internal prediction outputs REAL as primary verdict."""
        primary_verdict, original_verdict, is_fallback, warning = (
            PresentationDecisionPolicy.apply_presentation_policy(
                decision_state=DecisionState.REAL,
                fake_probability=0.05,
                has_conflict=False,
                quality_issues=False,
            )
        )
        self.assertEqual(primary_verdict, DecisionState.REAL)
        self.assertEqual(original_verdict, DecisionState.REAL)
        self.assertFalse(is_fallback)
        self.assertIsNone(warning)

        result = DecisionResult(
            prediction=DecisionState.REAL,
            fake_probability=0.05,
            confidence=95.0,
            quality_passed=True,
            face_valid=True,
        )
        d = result.to_dict()
        self.assertEqual(d["primary_verdict"], DecisionState.REAL)
        self.assertEqual(d["verdict"], DecisionState.REAL)
        self.assertEqual(d["original_verdict"], DecisionState.REAL)
        self.assertFalse(d["is_binary_fallback"])

    # -------------------------------------------------------------------------
    # 2. Confident FAKE prediction
    # -------------------------------------------------------------------------
    def test_02_confident_fake_prediction(self):
        """A confident FAKE internal prediction outputs FAKE as primary verdict without fallback warning."""
        primary_verdict, original_verdict, is_fallback, warning = (
            PresentationDecisionPolicy.apply_presentation_policy(
                decision_state=DecisionState.FAKE,
                fake_probability=0.92,
                has_conflict=False,
                quality_issues=False,
            )
        )
        self.assertEqual(primary_verdict, "FAKE")
        self.assertEqual(original_verdict, DecisionState.FAKE)
        self.assertFalse(is_fallback)
        self.assertIsNone(warning)

        result = DecisionResult(
            prediction=DecisionState.FAKE,
            fake_probability=0.92,
            confidence=92.0,
            quality_passed=True,
            face_valid=True,
        )
        d = result.to_dict()
        self.assertEqual(d["primary_verdict"], "FAKE")
        self.assertEqual(d["original_verdict"], DecisionState.FAKE)
        self.assertFalse(d["is_binary_fallback"])
        self.assertIsNone(d["reliability_warning"])

    # -------------------------------------------------------------------------
    # 3. UNCERTAIN internal result with valid probability -> REAL / FAKE provisional fallback
    # -------------------------------------------------------------------------
    def test_03_uncertain_internal_result_with_valid_probability(self):
        """
        Ambiguous probability in the abstention margin [0.35, 0.65] is internally
        UNCERTAIN. Under the binary presentation requirement, if a valid probability exists,
        it maps to the higher-probability binary class with is_binary_fallback=True and a warning.
        """
        # Case A: p_fake >= 0.50 (e.g. 0.58) -> FAKE (Provisional)
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.58,
            has_conflict=False,
            quality_issues=False,
        )
        self.assertEqual(primary_v, "FAKE")
        self.assertEqual(orig_v, DecisionState.UNCERTAIN)
        self.assertTrue(is_fallback)
        self.assertIn("provisional binary prediction", warning.lower())

        # Case B: p_fake < 0.50 (e.g. 0.42) -> REAL (Provisional)
        primary_v2, orig_v2, is_fallback2, warning2 = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.42,
            has_conflict=False,
            quality_issues=False,
        )
        self.assertEqual(primary_v2, "REAL")
        self.assertEqual(orig_v2, DecisionState.UNCERTAIN)
        self.assertTrue(is_fallback2)
        self.assertIn("provisional binary prediction", warning2.lower())

    # -------------------------------------------------------------------------
    # 4. Conflicting model evidence with warning displayed
    # -------------------------------------------------------------------------
    def test_04_conflicting_model_evidence_displays_warning(self):
        """When multi-agent specialist signals conflict, fallback maps to higher probability class with specific conflict warning."""
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.45,
            disagreement_detected=True,
        )
        self.assertEqual(primary_v, "REAL")
        self.assertEqual(orig_v, DecisionState.UNCERTAIN)
        self.assertTrue(is_fallback)
        self.assertEqual(warning, PresentationDecisionPolicy.CONFLICT_WARNING)
        self.assertIn("Disagreement detected between neural classifier and forensic analyzers", warning)

    # -------------------------------------------------------------------------
    # 5. NOT_SUITABLE media remaining NOT_SUITABLE
    # -------------------------------------------------------------------------
    def test_05_not_suitable_media_remains_not_suitable(self):
        """Unsupported media (no face, logos, documents) remains strictly NOT_SUITABLE without fake probabilities."""
        primary_v, orig_v, is_fallback, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.NOT_SUITABLE,
            fake_probability=None,
        )
        self.assertEqual(primary_v, DecisionState.NOT_SUITABLE)
        self.assertEqual(orig_v, DecisionState.NOT_SUITABLE)
        self.assertFalse(is_fallback)
        self.assertIsNone(warning)

        # Test FusionAgent with unsupported media
        fusion = FusionAgent()
        unsupported_primary = {
            "prediction": DecisionState.NOT_SUITABLE,
            "verdict": DecisionState.NOT_SUITABLE,
            "reason": "No human face found in image.",
        }
        fused = fusion.fuse("ImageAgent", unsupported_primary)
        self.assertEqual(fused["primary_verdict"], DecisionState.NOT_SUITABLE)
        self.assertEqual(fused["verdict"], DecisionState.NOT_SUITABLE)
        self.assertEqual(fused["original_verdict"], DecisionState.NOT_SUITABLE)
        self.assertIsNone(fused["fake_probability"])
        self.assertFalse(fused["is_binary_fallback"])

    # -------------------------------------------------------------------------
    # 6. Missing checkpoints and inference failures producing explicit error
    # -------------------------------------------------------------------------
    def test_06_missing_checkpoints_produce_explicit_error(self):
        """Missing model checkpoints must raise FileNotFoundError rather than fabricating a 0.50 score."""
        v_agent = VideoAgent()
        v_agent.weights_path = "/nonexistent/checkpoint.pth"
        with patch("os.path.exists", side_effect=lambda p: False if p == "/nonexistent/checkpoint.pth" else True):
            with self.assertRaises(FileNotFoundError):
                v_agent.analyze("dummy_sample.mp4")

        img_agent = ImageAgent()
        img_agent.weights_path = "/nonexistent/model.pth"
        with patch("os.path.exists", side_effect=lambda p: False if p == "/nonexistent/model.pth" else True):
            with patch("cv2.imread", return_value=np.zeros((224, 224, 3), dtype=np.uint8)):
                mock_q = MagicMock(eligibility="VALID", is_suitable=True, face_detected=True, primary_face_bbox=(0, 0, 100, 100))
                with patch.object(img_agent.quality_gate, "evaluate", return_value=mock_q):
                    with self.assertRaises(FileNotFoundError):
                        img_agent.analyze("dummy_sample.jpg")

        # If probability is None due to inference failure, policy does not fabricate a binary class
        p_v, o_v, is_fb, warn = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=None,
        )
        self.assertEqual(p_v, "UNCERTAIN")
        self.assertFalse(is_fb)
        self.assertIn("could not be completed", warn)

    # -------------------------------------------------------------------------
    # 7. Image and video responses rendering correctly
    # -------------------------------------------------------------------------
    def test_07_image_and_video_responses_rendering_correctly(self):
        """Responses from ImageAgent and VideoAgent include all required presentation fields."""
        # Check DecisionResult schema
        res = DecisionResult(
            prediction=DecisionState.UNCERTAIN,
            fake_probability=0.62,
            confidence=62.0,
            quality_passed=True,
            face_valid=True,
        )
        d = res.to_dict()
        required_presentation_fields = [
            "primary_verdict",
            "original_verdict",
            "is_binary_fallback",
            "reliability_warning",
            "fake_probability",
            "confidence",
        ]
        for field in required_presentation_fields:
            self.assertIn(field, d, f"Field '{field}' missing from to_dict()")
        self.assertEqual(d["primary_verdict"], "FAKE")
        self.assertEqual(d["original_verdict"], DecisionState.UNCERTAIN)
        self.assertTrue(d["is_binary_fallback"])

        # Check Natural Language inquiry integration
        nlp = generate_nlp_response(d, "Is this authentic?")
        self.assertIn("primary_verdict", nlp)
        self.assertEqual(nlp["primary_verdict"], "FAKE")
        self.assertTrue(nlp["is_binary_fallback"])

    # -------------------------------------------------------------------------
    # 8. Existing analysis history remaining compatible and normalizing LIKELY_REAL
    # -------------------------------------------------------------------------
    def test_08_analysis_history_compatibility(self):
        """Historical database rows map cleanly to the updated presentation schema without schema breakage."""
        from backend.api.analysis import analysis_bp
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(analysis_bp, url_prefix="/api")

        mock_real_row = {
            "id": "test-analysis-123",
            "file_name": "portrait_sample.jpg",
            "media_type": "image",
            "prediction": "REAL",
            "confidence": 85.0,
            "timestamp": "2026-10-13 22:00:00",
        }

        with patch("backend.api.analysis.get_analysis_by_id", return_value=mock_real_row), \
             patch("backend.api.analysis.get_agent_logs", return_value=[]):
            client = app.test_client()
            resp = client.get("/api/analysis/test-analysis-123")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data["status"], "success")
            analysis = data["analysis"]
            self.assertEqual(analysis["primary_verdict"], "REAL")
            self.assertEqual(analysis["primaryVerdict"], "REAL")
            self.assertEqual(analysis["verdict"], "REAL")

        # Historical row with legacy LIKELY_REAL must be normalized to REAL
        mock_likely_row = {
            "id": "test-analysis-456",
            "file_name": "historical_sample.jpg",
            "media_type": "image",
            "prediction": "LIKELY_REAL",
            "confidence": 82.0,
            "timestamp": "2026-10-13 22:00:00",
        }

        with patch("backend.api.analysis.get_analysis_by_id", return_value=mock_likely_row), \
             patch("backend.api.analysis.get_agent_logs", return_value=[]):
            client = app.test_client()
            resp2 = client.get("/api/analysis/test-analysis-456")
            self.assertEqual(resp2.status_code, 200)
            data2 = resp2.get_json()
            self.assertEqual(data2["status"], "success")
            analysis2 = data2["analysis"]
            self.assertEqual(analysis2["primary_verdict"], "REAL")
            self.assertEqual(analysis2["primaryVerdict"], "REAL")
            self.assertEqual(analysis2["verdict"], "REAL")
            self.assertEqual(analysis2["original_verdict"], "LIKELY_REAL")


if __name__ == "__main__":
    unittest.main()
