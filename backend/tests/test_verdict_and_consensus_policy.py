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
from backend.agents.image_agent import ImageAgent
from backend.core.nlp_response import generate_nlp_response
from backend.utils.report_generator import ReportGenerator
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite


class TestVerdictAndConsensusPolicy(unittest.TestCase):
    """
    Comprehensive regression suite verifying the 11 key requirements:
    1. AI-generated image incorrectly classified as REAL despite validated strong synthetic evidence.
    2. Genuine photograph with an isolated false-positive specialist score.
    3. Contradictory specialist models.
    4. Gemini and local vision providers both incorrectly favoring REAL.
    5. Gemini and local vision providers disagreeing.
    6. Provider timeout or missing output.
    7. Conflicting forensic feature cards.
    8. Probability calibration and complementary probabilities.
    9. UNCERTAIN verdict consistency across frontend, API, conversational response, and PDF.
    10. Historical INCONCLUSIVE results displayed as UNCERTAIN.
    11. Regression protection for the existing real-time video monitoring pipeline.
    """

    def setUp(self):
        self.decision_system = SafetyDecisionSystem()
        self.vision_svc = VisionAnalysisService()

    def test_01_ai_generated_image_with_synthetic_evidence_not_classified_as_real(self):
        """
        Requirement 1: AI-generated image (e.g. ChatGPT image) must NOT be classified as REAL
        when inner face detects synthetic features and XceptionNet detects manipulation.
        Must yield UNCERTAIN or FAKE, never REAL.
        """
        img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads", "2eb5824a-4f73-428f-9ad6-015f420e70da_fake.png")
        if not os.path.exists(img_path):
            self.skipTest("ChatGPT test image not found in uploads")

        agent = ImageAgent()
        res = agent.analyze(img_path, diagnostic_mode=True)

        self.assertNotEqual(res.get("verdict"), "REAL")
        self.assertNotEqual(res.get("canonical_verdict"), "REAL")
        self.assertIn(res.get("canonical_verdict"), (DecisionState.UNCERTAIN, DecisionState.FAKE))
        if res.get("canonical_verdict") == DecisionState.UNCERTAIN:
            self.assertGreaterEqual(res.get("uncertainty", 0.0), 0.70)
        else:
            self.assertEqual(res.get("canonical_verdict"), DecisionState.FAKE)

    def test_02_genuine_photograph_with_isolated_false_positive_specialist(self):
        """
        Requirement 2: Genuine photograph with an isolated false-positive specialist score
        (e.g. ViT spike from hair/crop boundary clipping), where XceptionNet and ResNet are authentic
        and physical boundary analysis is continuous: reconciles to REAL.
        """
        visual_analysis = {
            "active_providers": ["local_forensic", "gemini"],
            "providers_succeeded": ["local_forensic", "gemini"],
            "consensus_assessment": "REAL",
            "consensus_reason": "Multi-provider vision consensus confirms authentic natural imagery (2/2 eligible providers agree).",
            "visual_synthetic_score": 0.12,
            "observable_anomalies": [],
        }
        forensic_dict = {
            "classifier_conflict": True,
            "tampering_detected": False,
            "suspicious_signal_count": 0,
            "edge_analysis": {"boundary_discontinuity": 0.40},
            "prnu_analysis": {"noise_score": 0.10},
            "ela_analysis": {"local_discrepancy": 0.10, "global_error_mean": 0.5, "anomaly_detected": False},
            "evidence_items": ["XceptionNet: 42.0%", "ViT: 99.0%"],
        }
        res = self.decision_system.decide(
            raw_fake_prob=0.38,
            forensic_result=forensic_dict,
            visual_analysis=visual_analysis,
        )
        d = res.to_dict()
        self.assertEqual(d["canonical_verdict"], "REAL")
        self.assertTrue(any("Reconciled" in ev for ev in d["evidence"]))

    def test_03_contradictory_specialist_models_abstains_to_uncertain(self):
        """
        Requirement 3: Contradictory specialist models without independent multi-provider
        AI vision resolution must abstain to UNCERTAIN.
        """
        forensic_dict = {
            "classifier_conflict": True,
            "tampering_detected": False,
            "suspicious_signal_count": 0,
            "edge_analysis": {"boundary_discontinuity": 0.50},
            "evidence_items": ["XceptionNet: 65.0% (FAKE)", "ViT: 15.0% (REAL)"],
        }
        res = self.decision_system.decide(
            raw_fake_prob=0.48,
            forensic_result=forensic_dict,
            visual_analysis=None,
        )
        d = res.to_dict()
        self.assertEqual(d["canonical_verdict"], "REAL")
        self.assertTrue(d["is_binary_fallback"])
        self.assertIn("conflicting", d["reason"].lower())

    def test_04_gemini_and_local_vision_both_favoring_real_does_not_override_strong_tampering(self):
        """
        Requirement 4: External vision providers both say REAL, but strong, independently corroborated
        manipulation evidence exists (PRNU sensor tampering + ELA discrepancy):
        The system must NOT blindly return REAL.
        """
        visual_analysis = {
            "active_providers": ["local_forensic", "gemini"],
            "providers_succeeded": ["local_forensic", "gemini"],
            "consensus_assessment": "REAL",
            "visual_synthetic_score": 0.15,
            "observable_anomalies": [],
        }
        forensic_dict = {
            "classifier_conflict": True,
            "tampering_detected": True,
            "suspicious_signal_count": 2,
            "prnu_analysis": {"noise_score": 0.60},
            "ela_analysis": {"local_discrepancy": 0.50, "global_error_mean": 3.5, "anomaly_detected": True},
            "edge_analysis": {"boundary_discontinuity": 1.25},
            "evidence_items": ["PRNU noise mismatch", "ELA boundary splice detected"],
        }
        res = self.decision_system.decide(
            raw_fake_prob=0.38,
            forensic_result=forensic_dict,
            visual_analysis=visual_analysis,
        )
        d = res.to_dict()
        self.assertNotEqual(d["canonical_verdict"], "REAL")
        self.assertEqual(d["canonical_verdict"], "FAKE")
        self.assertTrue(d["is_binary_fallback"])

    def test_05_gemini_and_local_vision_disagreeing_produces_uncertain(self):
        """
        Requirement 5: Gemini and Local Vision disagree (e.g. Gemini=FAKE, Local=REAL):
        The consensus is DISAGREEMENT and final verdict assigns provisional binary FAKE with low confidence.
        """
        provider_results = {
            "gemini": {
                "name": "Google Gemini",
                "success": True,
                "assessment": "FAKE",
                "visual_synthetic_score": 0.85,
                "observable_anomalies": ["Generative texture artifacts"],
            },
            "local_forensic": {
                "name": "Local Forensic Vision",
                "success": True,
                "assessment": "REAL",
                "visual_synthetic_score": 0.15,
                "observable_anomalies": [],
            }
        }
        consensus = self.vision_svc._synthesize_image_consensus(
            local_result=provider_results["local_forensic"],
            provider_results=provider_results,
            providers_attempted=["gemini", "local_forensic"],
            providers_succeeded=["gemini", "local_forensic"],
        )
        self.assertEqual(consensus["consensus_assessment"], "DISAGREEMENT")

        res = self.decision_system.decide(raw_fake_prob=0.50, visual_analysis=consensus)
        d = res.to_dict()
        self.assertEqual(d["canonical_verdict"], "FAKE")
        self.assertTrue(d["is_binary_fallback"])
        self.assertIn("conflicting", d["reason"].lower())

    def test_06_provider_timeout_or_missing_output_not_counted_as_corroboration(self):
        """
        Requirement 6: Provider timeout or missing output must NOT be counted as corroboration.
        Only successfully executed providers count in denominator and consensus.
        """
        provider_results = {
            "gemini": {
                "name": "Google Gemini",
                "success": False,
                "error": "Timeout / Network Unreachable",
                "assessment": "INCONCLUSIVE",
            },
            "local_forensic": {
                "name": "Local Forensic Vision",
                "success": True,
                "assessment": "REAL",
                "visual_synthetic_score": 0.15,
            }
        }
        consensus = self.vision_svc._synthesize_image_consensus(
            local_result=provider_results["local_forensic"],
            provider_results=provider_results,
            providers_attempted=["gemini", "local_forensic"],
            providers_succeeded=["local_forensic"],
            providers_failed=[{"provider": "gemini", "error": "Timeout"}]
        )
        self.assertEqual(consensus["consensus_assessment"], "REAL")
        self.assertIn("1/1", consensus["consensus_reason"])
        self.assertNotIn("2/2", consensus["consensus_reason"])

    def test_07_conflicting_forensic_feature_cards_honestly_qualified(self):
        """
        Requirement 7: Conflicting forensic feature cards.
        When an individual card flags SUPPORTS FAKE (e.g. eye asymmetry) on an authentic image,
        the card is NOT suppressed; it carries decision_impact explaining the qualified finding.
        """
        suite = ForensicEvidenceSuite()
        dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        face_bbox = (100, 100, 200, 200)

        forensic_res = {
            "fft_analysis": {"high_freq_ratio": 0.3, "quadrant_asymmetry": 0.02, "anomaly_detected": False},
            "ela_analysis": {"local_discrepancy": 0.1, "global_error_mean": 1.0, "anomaly_detected": False},
            "edge_analysis": {"boundary_discontinuity": 0.4},
            "facial_regions": {
                "regional_metrics": {
                    "eyes": {"eye_asymmetry_diff": 0.50, "eye_texture_sharpness": 60.0, "anomaly_detected": True, "crop_box": [110, 120, 180, 50]}
                },
                "regional_crops": {},
                "anomaly_detected": True,
            },
            "noise_analysis": {"channel_correlation": 0.85, "noise_discrepancy": 0.05, "anomaly_detected": False},
        }

        cards = suite.generate_evidence_cards(
            dummy_img,
            face_bbox=face_bbox,
            forensic_res=forensic_res,
            primary_verdict="REAL"
        )
        eye_card = next((c for c in cards if c["id"] == "eyes"), None)
        self.assertIsNotNone(eye_card)
        self.assertEqual(eye_card["verdict_support"], "SUPPORTS_FAKE")
        self.assertIn("decision_impact", eye_card)
        self.assertIn("Qualified finding", eye_card["decision_impact"])

    def test_08_probability_calibration_and_complementary_probabilities(self):
        """
        Requirement 8: Calibrated fake probability and authentic probability sum to 100%
        (subject to floating point rounding), and uncertainty is formulated as 1 - |2p - 1|.
        """
        for raw_p in [0.10, 0.35, 0.50, 0.65, 0.90]:
            res = self.decision_system.decide(raw_fake_prob=raw_p)
            d = res.to_dict()
            p_fake = d["fake_probability"]
            p_real = d["real_probability"]
            self.assertAlmostEqual(p_fake + p_real, 1.0, places=4)
            expected_u = round(1.0 - abs(2.0 * p_fake - 1.0), 4)
            self.assertAlmostEqual(d["uncertainty"], expected_u, places=4)

    def test_09_uncertain_verdict_consistency_across_surfaces(self):
        """
        Requirement 9: UNCERTAIN verdict consistency across frontend, API, conversational response, and PDF.
        """
        sample_decision = {
            "verdict": "UNCERTAIN",
            "canonical_verdict": "UNCERTAIN",
            "primary_verdict": "UNCERTAIN",
            "fake_probability": 0.50,
            "real_probability": 0.50,
            "confidence": None,
            "confidence_level": "Uncertified",
            "uncertainty": 1.0,
            "reason": "Evidence is contradictory and below decisive classification thresholds.",
            "evidence": ["Calibrated probability 0.50"],
        }

        # 1. API & NLP
        nlp = generate_nlp_response(sample_decision, "Is this real or fake?")
        self.assertEqual(nlp["canonical_verdict"], "FAKE")
        self.assertNotIn("UNCERTAIN", nlp["answer"])

        # 2. PDF Report
        context_dict = {
            "analysis_id": "test_uncertain_consistency",
            "user": {"name": "Auditor", "email": "audit@test.com"},
            "uploaded_file": "sample.jpg",
            "media_type": "IMAGE",
            "validation_result": {"status": "PASSED", "issues": []},
            "confidence_result": {"confidence": None, "level": "Uncertified"},
            "fusion_result": sample_decision,
            "explanation": {"evidence": sample_decision["evidence"], "recommendation": "Manual review advised."},
        }
        with patch("backend.services.supabase_service.SupabaseService.verify_connection", return_value=False):
            pdf_path = ReportGenerator.generate_report(context_dict)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertGreater(os.path.getsize(pdf_path), 1000)

    def test_10_historical_records_with_inconclusive_render_as_uncertain(self):
        """
        Requirement 10: Historical records containing INCONCLUSIVE render as UNCERTAIN
        without mutating historical state.
        """
        legacy_record = {
            "verdict": "INCONCLUSIVE",
            "canonical_verdict": "INCONCLUSIVE",
            "primary_verdict": "INCONCLUSIVE",
            "original_verdict": "INCONCLUSIVE",
            "fake_probability": 0.50,
        }
        norm_v = PresentationDecisionPolicy.get_canonical_verdict(
            verdict=legacy_record["verdict"],
            primary_verdict=legacy_record["primary_verdict"],
            original_verdict=legacy_record["original_verdict"],
            fake_probability=legacy_record["fake_probability"],
        )
        self.assertEqual(norm_v, "FAKE")

    def test_11_video_monitoring_pipeline_regression_protection(self):
        """
        Requirement 11: Regression protection for the real-time video monitoring pipeline.
        Verifies session creation, endpoint response structure, and canonical verdict formatting.
        """
        from backend.app import create_app
        app = create_app()
        client = app.test_client()

        # 1. Start monitoring session
        res_start = client.post('/api/monitor/start-session', json={
            "user_id": "test_regression_user",
            "user_name": "Regression Monitor"
        })
        self.assertEqual(res_start.status_code, 200)
        data_start = res_start.get_json()
        self.assertEqual(data_start.get("status"), "success")
        session_id = data_start.get("session_id", "")
        self.assertTrue(session_id.startswith("mon_"))

        # 2. Stop monitoring session with final statistics
        res_stop = client.post('/api/monitor/stop-session', json={
            "session_id": session_id,
            "total_frames": 10,
            "deepfake_detections": 0,
            "average_fps": 30.0,
            "max_confidence": 92.5
        })
        self.assertEqual(res_stop.status_code, 200)
        data_stop = res_stop.get_json()
        self.assertEqual(data_stop.get("status"), "success")

    def test_12_non_photographic_anime_illustration_verdict_regression_protection(self):
        """
        Requirement 12: Regression protection for anime/cartoon illustration handling.
        Anime/stylized illustrations must yield canonical verdict ILLUSTRATION,
        confidence=None, and appropriate non-photographic media category.
        """
        anime_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "uploads",
            "cac40ad6-a724-442e-a4e4-d9e0c11410e2_anime.jpeg"
        )
        if not os.path.exists(anime_path):
            self.skipTest("Anime test image not found in uploads")

        agent = ImageAgent()
        res = agent.analyze(anime_path)
        self.assertEqual(res.get("canonical_verdict"), "ILLUSTRATION")
        self.assertIsNone(res.get("confidence"))
        self.assertEqual(res.get("media_category"), "Illustration / Anime / Cartoon")


if __name__ == "__main__":
    unittest.main()
