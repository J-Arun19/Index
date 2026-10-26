import os
import cv2
import json
import torch
import unittest
import numpy as np
import tempfile
import shutil

from backend.core.decision_system import SafetyDecisionSystem, DecisionState, DecisionResult
from backend.core.calibration import TemperatureScalingCalibrator
from backend.preprocessing.quality_gate import ImageQualityGate, QualityGateResult
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.nlp_response import generate_nlp_response
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.orchestrator import Orchestrator


class TestInferenceSafety(unittest.TestCase):
    """
    Comprehensive test suite for SatyaDrishti Safety Decision Architecture.
    Validates three-state decision logic, abstention gates, critical false-negative
    protection, video temporal aggregation, and natural-language grounded responses.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.calibrator = TemperatureScalingCalibrator(temperature=1.885, tau_real=0.35, tau_fake=0.65)
        cls.decision_system = SafetyDecisionSystem(calibrator=cls.calibrator)

        # 1. Clean synthetic face image (good resolution, clear face)
        cls.clean_face_img_path = os.path.join(cls.test_dir, "clean_face.jpg")
        img = np.full((300, 300, 3), 120, dtype=np.uint8)
        # Draw realistic face-like contours
        cv2.circle(img, (150, 150), 75, (200, 180, 170), -1)
        cv2.circle(img, (125, 130), 10, (50, 50, 50), -1)
        cv2.circle(img, (175, 130), 10, (50, 50, 50), -1)
        cv2.ellipse(img, (150, 175), (25, 12), 0, 0, 180, (40, 40, 150), 3)
        cv2.imwrite(cls.clean_face_img_path, img)

        # 2. Blurry / poor-quality image (Laplacian variance < 10)
        cls.blurry_img_path = os.path.join(cls.test_dir, "blurry.jpg")
        blurry = cv2.GaussianBlur(img, (35, 35), 15.0)
        cv2.imwrite(cls.blurry_img_path, blurry)

        # 3. Tiny / low-resolution image (< 100x100)
        cls.low_res_img_path = os.path.join(cls.test_dir, "low_res.jpg")
        low_res = cv2.resize(img, (48, 48))
        cv2.imwrite(cls.low_res_img_path, low_res)

        # 4. Image without face (pure geometric pattern)
        cls.no_face_img_path = os.path.join(cls.test_dir, "no_face.jpg")
        no_face = np.zeros((300, 300, 3), dtype=np.uint8)
        for i in range(0, 300, 30):
            cv2.line(no_face, (i, 0), (i, 300), (255, 255, 255), 2)
        cv2.imwrite(cls.no_face_img_path, no_face)

        # 5. Corrupt / 0-byte file
        cls.corrupt_img_path = os.path.join(cls.test_dir, "corrupt.jpg")
        with open(cls.corrupt_img_path, "wb") as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER")

        # 6. Video clip with synthetic frames
        cls.video_path = os.path.join(cls.test_dir, "test_clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(cls.video_path, fourcc, 10.0, (256, 256))
        for _ in range(15):
            f_frame = np.full((256, 256, 3), 100, dtype=np.uint8)
            cv2.circle(f_frame, (128, 128), 60, (210, 180, 170), -1)
            out.write(f_frame)
        out.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: High-Confidence Fake -> FAKE
    # -------------------------------------------------------------------------
    def test_01_high_confidence_fake_returns_fake(self):
        """High model probability (> 0.85) with neutral/suspicious forensics must output FAKE."""
        raw_prob = 0.93
        decision = self.decision_system.decide(raw_fake_prob=raw_prob)

        self.assertEqual(decision.prediction, DecisionState.FAKE)
        self.assertEqual(decision.verdict, DecisionState.FAKE)
        self.assertGreaterEqual(decision.confidence, 60.0)
        self.assertIn("Deepfake detected", decision.user_facing_statement)
        # Must never claim "100% fake"
        self.assertNotIn("100% fake", decision.user_facing_statement)
        self.assertNotIn("definitely fake", decision.user_facing_statement)

    # -------------------------------------------------------------------------
    # Test 2: High-Confidence Real -> LIKELY_REAL
    # -------------------------------------------------------------------------
    def test_02_high_confidence_real_returns_likely_real(self):
        """Low model probability (< 0.15) with high quality and no anomalies must output LIKELY_REAL."""
        raw_prob = 0.04
        decision = self.decision_system.decide(raw_fake_prob=raw_prob)

        self.assertEqual(decision.prediction, DecisionState.LIKELY_REAL)
        self.assertEqual(decision.verdict, DecisionState.LIKELY_REAL)
        self.assertIn("No strong evidence of manipulation was detected", decision.user_facing_statement)
        # Must NEVER tell the user "This image is definitely real"
        self.assertNotIn("definitely real", decision.user_facing_statement)
        self.assertNotIn("100% real", decision.user_facing_statement)
        self.assertNotIn("proven authentic", decision.user_facing_statement)

    # -------------------------------------------------------------------------
    # Test 3: Boundary Probability -> UNCERTAIN
    # -------------------------------------------------------------------------
    def test_03_boundary_probability_abstains_to_uncertain(self):
        """Probability falling in the indeterminate margin [0.35, 0.65] must abstain to UNCERTAIN."""
        boundary_probs = [0.45, 0.50, 0.55, 0.52]
        for p in boundary_probs:
            decision = self.decision_system.decide(raw_fake_prob=p)
            self.assertEqual(
                decision.prediction,
                DecisionState.UNCERTAIN,
                f"Probability {p} failed to abstain to UNCERTAIN.",
            )
            self.assertIn("Insufficient evidence", decision.user_facing_statement)
            self.assertGreaterEqual(decision.uncertainty, 0.50)

    # -------------------------------------------------------------------------
    # Test 4: Contradictory Evidence -> UNCERTAIN (CRITICAL PROTECTION)
    # -------------------------------------------------------------------------
    def test_04_contradictory_evidence_abstains_to_uncertain(self):
        """
        Model raw score is low (leans real, e.g. 0.10), but forensic analyzer
        flags tampering artifacts. System MUST escalate to UNCERTAIN (NEVER REAL).
        """
        raw_prob = 0.10  # Model alone would say REAL!
        forensic_tampering = {
            "composite_forensic_score": 0.68,
            "evidence_items": [
                "FFT Spectrum: Anomalous high-frequency spectral distribution detected.",
                "Compression ELA: Significant localized compression inconsistency flagged.",
            ],
            "suspicious_signal_count": 2,
        }

        decision = self.decision_system.decide(
            raw_fake_prob=raw_prob,
            forensic_result=forensic_tampering,
        )

        self.assertEqual(decision.prediction, DecisionState.UNCERTAIN)
        self.assertNotEqual(decision.prediction, DecisionState.LIKELY_REAL)
        self.assertIn("contradictory evidence", decision.reason.lower())

    # -------------------------------------------------------------------------
    # Test 5: Poor-Quality Image -> UNCERTAIN
    # -------------------------------------------------------------------------
    def test_05_poor_quality_image_abstains_to_uncertain(self):
        """Excessively blurred or low-resolution image must fail quality gate and abstain to UNCERTAIN."""
        gate = ImageQualityGate()
        blurry_res = gate.evaluate(self.blurry_img_path, require_face=False)
        self.assertFalse(blurry_res.is_suitable)

        decision = self.decision_system.decide(
            raw_fake_prob=0.90,  # Even with high model score, bad quality forces UNCERTAIN
            quality_result=blurry_res,
        )
        self.assertEqual(decision.prediction, DecisionState.UNCERTAIN)
        self.assertIn("quality failure", decision.reason.lower())

    # -------------------------------------------------------------------------
    # Test 6: No Usable Face -> NOT_SUITABLE (Media Suitability Separation)
    # -------------------------------------------------------------------------
    def test_06_no_usable_face_returns_not_suitable(self):
        """When face forensic analysis is required but no face is found, must return NOT_SUITABLE."""
        gate = ImageQualityGate()
        no_face_res = gate.evaluate(self.no_face_img_path, require_face=True)

        self.assertFalse(no_face_res.is_suitable)
        self.assertFalse(no_face_res.face_detected)
        self.assertEqual(no_face_res.eligibility, "UNSUPPORTED")

        decision = self.decision_system.decide(
            raw_fake_prob=0.05,  # Model alone would declare confident REAL
            quality_result=no_face_res,
        )
        self.assertEqual(decision.prediction, DecisionState.NOT_SUITABLE)
        self.assertIsNone(decision.confidence)
        self.assertIsNone(decision.fake_probability)
        self.assertIn("face", decision.user_facing_statement.lower())

    # -------------------------------------------------------------------------
    # Test 7: Verified Fake Previously Predicted REAL -> Surfaced as False Negative
    # -------------------------------------------------------------------------
    def test_07_verified_fake_previously_predicted_real_surfaced(self):
        """A ground truth fake with low model score is explicitly recorded as a hard negative."""
        ground_truth = 1  # FAKE
        raw_prob = 0.02   # Missed by raw model

        decision = self.decision_system.decide(
            raw_fake_prob=raw_prob,
            ground_truth=ground_truth,
        )
        # If it reached LIKELY_REAL, it must be flagged as a hard negative
        if decision.prediction == DecisionState.LIKELY_REAL:
            self.assertTrue(decision.is_hard_negative)

    # -------------------------------------------------------------------------
    # Test 8: Model / Evidence Disagreement in Fusion
    # -------------------------------------------------------------------------
    def test_08_model_evidence_disagreement(self):
        """FusionAgent resolves conflict between primary specialist and metadata by abstaining."""
        fusion = FusionAgent()

        # Primary says REAL (low fake probability)
        primary_out = {
            "prediction": DecisionState.LIKELY_REAL,
            "verdict": "REAL",
            "fake_probability": 0.12,
            "confidence": 88.0,
            "raw_model_score": 0.12,
            "summary": "Primary model visual analysis leans real.",
        }

        # Metadata detected Photoshop / synthetic generation tag
        metadata_out = {
            "status": "Success",
            "anomalies_detected": True,
            "software_tag": "Adobe Firefly AI Splicer",
            "summary": "AI editing software signature detected in EXIF headers.",
        }

        # Linear mock invocation
        class MockContext:
            media_type = "image"
            image_agent_result = primary_out
            metadata_agent_result = metadata_out
            artifact_agent_result = None
            analysis_id = "test_conflict"

        fused = fusion.fuse(MockContext())
        self.assertEqual(fused["original_verdict"], DecisionState.UNCERTAIN)
        self.assertEqual(fused["primary_verdict"], DecisionState.REAL)
        self.assertTrue(fused["is_binary_fallback"])
        self.assertTrue(fused["disagreement_detected"])
        self.assertIn("evidence conflict", fused["reason"].lower())

    # -------------------------------------------------------------------------
    # Test 9: Normal Valid Inference Schema
    # -------------------------------------------------------------------------
    def test_09_normal_valid_inference(self):
        """ImageAgent returns structured schema separating raw score from final decision."""
        agent = ImageAgent()
        result = agent.analyze(self.clean_face_img_path, require_face=False)

        # Verify structured contract
        required_fields = [
            "prediction",
            "verdict",
            "confidence",
            "fake_probability",
            "real_probability",
            "raw_model_score",
            "uncertainty",
            "evidence",
            "reason",
            "summary",
            "quality_gate",
        ]
        for field in required_fields:
            self.assertIn(field, result, f"Field '{field}' missing from ImageAgent result.")

        self.assertIn(result["prediction"], [DecisionState.FAKE, DecisionState.LIKELY_REAL, DecisionState.UNCERTAIN])
        self.assertIsInstance(result["fake_probability"], float)
        self.assertIsInstance(result["raw_model_score"], float)

    # -------------------------------------------------------------------------
    # Test 10: Invalid / Corrupt Input
    # -------------------------------------------------------------------------
    def test_10_invalid_or_corrupt_input(self):
        """Corrupt or non-image file is caught gracefully and abstains to UNCERTAIN."""
        agent = ImageAgent()
        result = agent.analyze(self.corrupt_img_path, require_face=False)

        self.assertEqual(result["prediction"], DecisionState.UNCERTAIN)
        self.assertFalse(result["quality_gate"]["is_suitable"])

    # -------------------------------------------------------------------------
    # Test 11: Video with Mixed Frame Predictions -> UNCERTAIN
    # -------------------------------------------------------------------------
    def test_11_video_with_mixed_frame_predictions_abstains(self):
        """Video with contradictory frame evidence (some fake, some real) must abstain to UNCERTAIN."""
        temporal_conflict = True
        temporal_variance = 0.32  # High frame variance

        decision = self.decision_system.decide(
            raw_fake_prob=0.55,
            temporal_variance=temporal_variance,
            temporal_conflict=temporal_conflict,
        )

        self.assertEqual(decision.prediction, DecisionState.UNCERTAIN)
        self.assertIn("temporal inconsistency", decision.reason.lower())

    # -------------------------------------------------------------------------
    # Test 12: Video Temporal Inconsistency -> UNCERTAIN
    # -------------------------------------------------------------------------
    def test_12_video_temporal_inconsistency(self):
        """Video with high inter-frame jitter variance (> 0.22) must abstain."""
        decision = self.decision_system.decide(
            raw_fake_prob=0.88,
            temporal_variance=0.28,
            temporal_conflict=True,
        )
        self.assertEqual(decision.prediction, DecisionState.UNCERTAIN)

    # -------------------------------------------------------------------------
    # Test 13: API Response Schema
    # -------------------------------------------------------------------------
    def test_13_api_response_schema(self):
        """Simulates orchestrator pipeline and validates structured contract."""
        agent = ImageAgent()
        res = agent.analyze(self.clean_face_img_path, require_face=False)

        self.assertIn("prediction", res)
        self.assertIn("verdict", res)
        self.assertIn("fake_probability", res)
        self.assertIn("real_probability", res)
        self.assertIn("confidence", res)
        self.assertIn("uncertainty", res)
        self.assertIn("evidence", res)
        self.assertIn("reason", res)
        self.assertIsInstance(res["evidence"], list)

    # -------------------------------------------------------------------------
    # Test 14: Natural-Language Inference Response
    # -------------------------------------------------------------------------
    def test_14_natural_language_inference_response(self):
        """Verifies conversational Q&A answers are grounded in the inference result."""
        # 14.A: Uncertain state inquiry
        uncertain_res = {
            "prediction": DecisionState.UNCERTAIN,
            "confidence": 50.0,
            "reason": "Calibrated probability falls in the decision margin [0.35 - 0.65].",
            "evidence": ["Ambiguous high-frequency distribution"],
        }
        nlp_uncertain = generate_nlp_response(uncertain_res, "Is this image a deepfake?")
        self.assertTrue(nlp_uncertain["abstention"])
        self.assertIn("SatyaDrishti cannot determine this reliably", nlp_uncertain["answer"])
        self.assertIn("UNCERTAIN", nlp_uncertain["answer"])

        # 14.B: Fake state inquiry
        fake_res = {
            "prediction": DecisionState.FAKE,
            "confidence": 92.5,
            "fake_probability": 0.93,
            "evidence": ["High synthetic probability confirmed"],
        }
        nlp_fake = generate_nlp_response(fake_res, "Can you tell if this is fake?")
        self.assertFalse(nlp_fake["abstention"])
        self.assertIn("classified as FAKE", nlp_fake["answer"])
        self.assertIn("Deepfake detected", nlp_fake["answer"])

        # 14.C: Likely Real state inquiry (normalized to user-facing REAL)
        real_res = {
            "prediction": DecisionState.LIKELY_REAL,
            "confidence": 88.0,
            "fake_probability": 0.05,
            "evidence": ["Consistent sensor noise"],
        }
        nlp_real = generate_nlp_response(real_res, "Is this real or AI generated?")
        self.assertFalse(nlp_real["abstention"])
        self.assertEqual(nlp_real["primary_verdict"], "REAL")
        self.assertIn("REAL", nlp_real["answer"])
        self.assertNotIn("LIKELY REAL", nlp_real["answer"])
        self.assertNotIn("LIKELY_REAL", nlp_real["answer"])
        self.assertIn("does not prove authenticity", nlp_real["answer"])

        # 14.D: Explicit certified REAL state inquiry
        certified_res = {
            "prediction": DecisionState.REAL,
            "confidence": 92.0,
            "fake_probability": 0.05,
            "evidence": ["Certified authentic facial features"],
        }
        nlp_certified = generate_nlp_response(certified_res, "Is this authentic?")
        self.assertFalse(nlp_certified["abstention"])
        self.assertIn("REAL", nlp_certified["answer"])
        self.assertNotIn("LIKELY REAL", nlp_certified["answer"])
        self.assertNotIn("LIKELY_REAL", nlp_certified["answer"])
        self.assertIn("does not prove absolute authenticity", nlp_certified["answer"])

    # -------------------------------------------------------------------------
    # Test 15: Critical Regression Test: Actual Previously Misclassified Deepfake
    # -------------------------------------------------------------------------
    def test_15_regression_known_misclassified_deepfake_no_longer_likely_real(self):
        """
        CRITICAL REGRESSION TEST:
        Verified ground-truth deepfake ('deepy.jpeg') previously classified as:
        GROUND TRUTH = FAKE
        OLD MODEL DECISION = REAL (89.23% confidence)

        The new safety architecture MUST NOT classify this verified deepfake as LIKELY_REAL.
        Acceptable: UNCERTAIN or FAKE.
        Unacceptable: LIKELY_REAL or REAL.
        """
        img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "uploads",
            "570a97bc-c4a4-49c6-ad91-08f954e86067_deepy.jpeg"
        )
        if not os.path.exists(img_path):
            self.skipTest(f"Regression test image {img_path} not found.")

        agent = ImageAgent()
        res = agent.analyze(img_path)

        # Verified ground-truth is FAKE (do NOT alter based on model score)
        verified_ground_truth = "FAKE"
        self.assertEqual(verified_ground_truth, "FAKE")

        # Must NOT return LIKELY_REAL
        self.assertNotEqual(
            res["prediction"],
            DecisionState.LIKELY_REAL,
            f"CRITICAL SAFETY REGRESSION: Verified deepfake {img_path} was classified as LIKELY_REAL!"
        )
        self.assertNotEqual(
            res["prediction"],
            "REAL",
            f"CRITICAL SAFETY REGRESSION: Verified deepfake {img_path} was classified as REAL!"
        )
        self.assertIn(
            res["prediction"],
            (DecisionState.UNCERTAIN, DecisionState.FAKE),
            f"Expected UNCERTAIN or FAKE, got {res['prediction']}"
        )

    # -------------------------------------------------------------------------
    # Test 16: Critical Regression Test: Celeb-DF Synthesis Frame
    # -------------------------------------------------------------------------
    def test_16_regression_celeb_df_synthesis_frame_no_longer_likely_real(self):
        """
        CRITICAL REGRESSION TEST:
        Verified ground-truth synthesized deepfake frame from Celeb-DF.
        The system must detect localized compression ELA anomaly or abstain to UNCERTAIN.
        Must NEVER return LIKELY_REAL.
        """
        img_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "uploads",
            "test_celeb_fake_frame32.jpg"
        )
        if not os.path.exists(img_path):
            self.skipTest(f"Regression test frame {img_path} not found.")

        agent = ImageAgent()
        res = agent.analyze(img_path)

        verified_ground_truth = "FAKE"
        self.assertEqual(verified_ground_truth, "FAKE")

        self.assertNotEqual(res["prediction"], DecisionState.LIKELY_REAL)
        self.assertNotEqual(res["prediction"], "REAL")
        self.assertIn(res["prediction"], (DecisionState.UNCERTAIN, DecisionState.FAKE))


if __name__ == "__main__":
    unittest.main()
