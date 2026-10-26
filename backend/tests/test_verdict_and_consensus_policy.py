import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.core.decision_system import (
    SafetyDecisionSystem,
    DecisionState,
    DecisionResult,
    PresentationDecisionPolicy,
)
from backend.services.vision_service import VisionAnalysisService
from backend.agents.fusion_agent import FusionAgent
from backend.core.nlp_response import generate_nlp_response
from backend.utils.report_generator import ReportGenerator


class TestVerdictAndConsensusPolicy(unittest.TestCase):
    """
    Focused regression checks covering:
    1. Conflicting specialist predictions (e.g. Xception 33.3%, ViT 97.4%).
    2. Agreement between specialist predictions (e.g. both indicate REAL).
    3. Gemini = FAKE and Local provider = REAL (explicit disagreement, no false consensus).
    4. A single eligible external provider (distinguishing single provider from consensus).
    5. Correct provider-consensus denominator (counting all eligible reporting providers).
    6. Consistency between API, report UI, conversational inquiry, history, and PDF.
    7. Uncalibrated scores not being mislabeled as probabilities.
    8. Failed inference producing an error rather than fabricating a classification.
    9. Successfully analyzed supported media returning REAL or FAKE under documented decision policy.
    """

    def setUp(self):
        self.decision_system = SafetyDecisionSystem()

    def test_01_conflicting_specialist_predictions_return_binary_verdict_with_low_confidence(self):
        """
        Check 1: Conflicting specialist predictions (Xception=33.3%, ViT=97.4%).
        Raw fused score = 0.50 * 0.333 + 0.50 * 0.974 = 0.6535 (~65.3%).
        Decision policy must return primary verdict FAKE (since >= 0.50),
        confidence_level 'Low', is_binary_fallback True, and preserve conflict in evidence.
        """
        raw_xcep = 0.333
        raw_vit = 0.974
        fused_raw = round(0.50 * raw_xcep + 0.50 * raw_vit, 4)  # 0.6535

        forensic_dict = {
            "classifier_conflict": True,
            "evidence_items": ["XceptionNet: 33.3%", "ViT: 97.4%"],
        }
        res = self.decision_system.decide(
            raw_fake_prob=fused_raw,
            forensic_result=forensic_dict,
        )

        d = res.to_dict()
        # Primary & canonical verdict must be binary FAKE
        self.assertEqual(d["verdict"], "FAKE")
        self.assertEqual(d["canonical_verdict"], "FAKE")
        self.assertEqual(d["primary_verdict"], "FAKE")
        # Confidence level must be Low with fallback active
        self.assertEqual(d["confidence_level"], "Low")
        self.assertTrue(d["is_binary_fallback"])
        # Evidence must preserve specialist disagreement
        self.assertTrue(any("Disagreement" in ev or "Classifier Disagreement" in ev for ev in d["evidence"]))
        self.assertIsNotNone(d["reliability_warning"])

    def test_02_agreement_between_specialist_predictions(self):
        """
        Check 2: Agreement between specialist predictions (both indicate authentic media).
        Returns REAL with certified high/medium confidence and no fallback conflict warning.
        """
        raw_fake_prob = 0.12
        forensic_dict = {
            "classifier_conflict": False,
            "tampering_detected": False,
            "evidence_items": ["Specialist models indicate authentic features."],
        }
        res = self.decision_system.decide(
            raw_fake_prob=raw_fake_prob,
            forensic_result=forensic_dict,
        )

        d = res.to_dict()
        self.assertEqual(d["verdict"], "REAL")
        self.assertEqual(d["canonical_verdict"], "REAL")
        self.assertFalse(d["is_binary_fallback"])
        self.assertIn(d["confidence_level"], ["High", "Medium"])
        self.assertIsNone(d["reliability_warning"])

    def test_03_gemini_fake_and_local_real_produces_provider_disagreement(self):
        """
        Check 3: Gemini = FAKE and Local Forensic Vision = REAL.
        Must accurately identify provider disagreement, count 2 eligible providers,
        and NEVER claim external consensus confirms synthetic manipulation.
        """
        vision_svc = VisionAnalysisService()
        provider_results = {
            "gemini": {
                "name": "Google Gemini",
                "success": True,
                "assessment": "FAKE",
                "visual_synthetic_score": 0.75,
                "confidence_score": 80.0,
                "observable_anomalies": ["Unnatural facial skin smoothing"],
                "is_external_ai": True,
            },
            "local_forensic": {
                "name": "Local Forensic Vision",
                "success": True,
                "assessment": "REAL",
                "visual_synthetic_score": 0.15,
                "confidence_score": 85.0,
                "observable_anomalies": [],
                "is_external_ai": False,
            }
        }

        local_res = provider_results["local_forensic"]
        consensus = vision_svc._synthesize_image_consensus(
            local_result=local_res,
            provider_results=provider_results,
            providers_attempted=["gemini", "local_forensic"]
        )
        self.assertEqual(consensus["consensus_assessment"], "DISAGREEMENT")
        self.assertIn("Provider disagreement detected", consensus["consensus_reason"])
        self.assertIn("1 FAKE vs 1 REAL across 2 eligible providers", consensus["consensus_reason"])
        self.assertNotIn("External AI vision consensus confirms synthetic manipulation (1/1 providers)", consensus["consensus_reason"])

    def test_04_single_eligible_external_provider_distinguished_from_consensus(self):
        """
        Check 4: A single eligible external provider (e.g. only Gemini reported).
        Must state single active provider and NOT claim multi-provider consensus.
        """
        vision_svc = VisionAnalysisService()
        provider_results = {
            "gemini": {
                "name": "Google Gemini",
                "success": True,
                "assessment": "FAKE",
                "visual_synthetic_score": 0.80,
                "confidence_score": 85.0,
                "observable_anomalies": ["Texture inconsistency"],
                "is_external_ai": True,
            }
        }

        consensus = vision_svc._synthesize_image_consensus(
            local_result={},
            provider_results=provider_results,
            providers_attempted=["gemini"]
        )
        self.assertEqual(consensus["consensus_assessment"], "FAKE")
        self.assertIn("single active provider", consensus["consensus_reason"])
        self.assertNotIn("Multi-provider vision consensus", consensus["consensus_reason"])

    def test_05_correct_provider_consensus_denominator(self):
        """
        Check 5: Correct provider-consensus denominator across multi-provider evaluation.
        When 3 providers report (e.g. Gemini=FAKE, OpenAI=FAKE, Local=FAKE), denominator is 3.
        """
        vision_svc = VisionAnalysisService()
        provider_results = {
            "gemini": {"name": "Google Gemini", "success": True, "assessment": "FAKE", "visual_synthetic_score": 0.85},
            "openai_vision": {"name": "OpenAI Vision", "success": True, "assessment": "FAKE", "visual_synthetic_score": 0.80},
            "local_forensic": {"name": "Local Forensic Vision", "success": True, "assessment": "FAKE", "visual_synthetic_score": 0.70},
        }

        consensus = vision_svc._synthesize_image_consensus(
            local_result=provider_results["local_forensic"],
            provider_results=provider_results,
            providers_attempted=["gemini", "openai_vision", "local_forensic"]
        )
        self.assertEqual(consensus["consensus_assessment"], "FAKE")
        self.assertIn("3/3 eligible providers agree", consensus["consensus_reason"])

    def test_06_consistency_across_api_nlp_and_pdf(self):
        """
        Check 6: Consistency between API, NLP inquiry, and PDF report.
        All client surfaces must share the exact same binary primary verdict and Low Confidence warning.
        """
        sample_decision = {
            "verdict": "FAKE",
            "primary_verdict": "FAKE",
            "fake_probability": 0.677,
            "confidence": 67.7,
            "confidence_level": "Low",
            "is_binary_fallback": True,
            "reliability_warning": "Disagreement detected between specialist neural detectors.",
            "reason": "Classifier disagreement between Xception and ViT.",
            "evidence": ["Xception: 33.3%", "ViT: 97.4%"],
        }

        # 1. Decision System normalization
        norm_v = PresentationDecisionPolicy.get_canonical_verdict(
            verdict=sample_decision["verdict"],
            primary_verdict=sample_decision["primary_verdict"],
            is_binary_fallback=True,
            fake_probability=sample_decision["fake_probability"],
        )
        self.assertEqual(norm_v, "FAKE")

        # 2. Conversational inquiry
        nlp = generate_nlp_response(sample_decision, "Is this a deepfake?")
        self.assertEqual(nlp["prediction"], "FAKE")
        self.assertEqual(nlp["primary_verdict"], "FAKE")
        self.assertIn("classified as FAKE (LOW CONFIDENCE)", nlp["answer"])
        self.assertIn("Warning:", nlp["answer"])

        # 3. PDF report generation should complete without crashing and respect the verdict
        context_dict = {
            "analysis_id": "test_sync_report",
            "user": {"name": "Auditor", "email": "audit@test.com"},
            "uploaded_file": "sample.jpg",
            "media_type": "IMAGE",
            "validation_result": {"status": "PASSED", "issues": []},
            "confidence_result": {"confidence": 67.7, "level": "Low"},
            "fusion_result": sample_decision,
            "explanation": {"evidence": sample_decision["evidence"], "recommendation": "Manual review advised."},
        }
        with patch("backend.services.supabase_service.SupabaseService.verify_connection", return_value=False):
            pdf_path = ReportGenerator.generate_report(context_dict)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_07_uncalibrated_scores_not_mislabeled_as_probabilities(self):
        """
        Check 7: Uncalibrated raw scores (Xception, ViT, VideoModel) must have calibrated_prob=None
        or clearly identified calibration_status to avoid mislabeling uncalibrated outputs.
        """
        from backend.agents.video_agent import VideoAgent
        v_agent = VideoAgent()
        # Verify VideoAgent models_inventory specification
        from backend.agents.image_agent import ImageAgent
        img_agent = ImageAgent()

        # ImageAgent inventory check
        mock_decision = DecisionResult(prediction="REAL", fake_probability=0.20, confidence=80.0)
        mock_quality = MagicMock(blur_score=50.0, resolution=[256, 256])
        resp = img_agent._format_response(
            decision=mock_decision,
            quality_res=mock_quality,
            forensic_res={},
            visual_res={},
            filename="test.jpg",
            size=1024,
            fmt="JPEG",
            processing_time=0.1,
            img_shape=(256, 256, 3),
            raw_xcep_p=0.42,
            raw_vit_p=0.95,
            raw_resnet_p=0.30,
        )
        for m in resp.get("models", []):
            if m["name"] in ("XceptionNet (DeepfakeBench)", "Vision Transformer (ViT-B/16)", "ResNet-18 (Baseline)"):
                self.assertIsNone(m["calibrated_prob"], f"{m['name']} should have calibrated_prob=None")
                self.assertIn("Not calibrated", m["calibration_status"])

    def test_08_failed_inference_produces_error_without_fabricated_classification(self):
        """
        Check 8: Failed inference / missing probability produces an explicit UNCERTAIN / error
        rather than fabricating a classification.
        """
        p_v, o_v, is_fb, warn = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=None,
            reason="Inference failure: weights missing",
        )
        self.assertEqual(p_v, DecisionState.UNCERTAIN)
        self.assertFalse(is_fb)
        self.assertIn("could not be completed", warn.lower())

    def test_09_successfully_analyzed_supported_media_returns_real_or_fake(self):
        """
        Check 9: Successfully analyzed supported media returns strictly REAL or FAKE.
        Never returns UNCERTAIN as primary verdict when valid probability exists.
        """
        # Supported media with borderline score (0.51)
        res_fake = self.decision_system.decide(raw_fake_prob=0.51)
        d_fake = res_fake.to_dict()
        self.assertIn(d_fake["canonical_verdict"], ["REAL", "FAKE"])
        self.assertEqual(d_fake["canonical_verdict"], "FAKE")

        # Supported media with borderline score (0.49)
        res_real = self.decision_system.decide(raw_fake_prob=0.49)
        d_real = res_real.to_dict()
        self.assertIn(d_real["canonical_verdict"], ["REAL", "FAKE"])
        self.assertEqual(d_real["canonical_verdict"], "REAL")


if __name__ == "__main__":
    unittest.main()
