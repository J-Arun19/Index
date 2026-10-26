import unittest
from backend.core.decision_system import DecisionState, DecisionResult
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.fusion_agent import FusionAgent
from backend.preprocessing.quality_gate import QualityGateResult

class TestCanonicalResultContract(unittest.TestCase):
    def test_image_agent_canonical_contract_structure(self):
        """ImageAgent._format_response satisfies all canonical result contract requirements."""
        agent = ImageAgent()
        q_res = QualityGateResult(
            is_suitable=True,
            face_detected=True,
            face_count=1,
            primary_face_bbox=(50, 50, 150, 150),
            resolution=(1920, 1080),
            blur_score=142.5
        )
        decision = DecisionResult(
            prediction=DecisionState.UNCERTAIN,
            primary_verdict="UNCERTAIN",
            original_verdict=DecisionState.UNCERTAIN,
            is_binary_fallback=False,
            confidence=None,
            confidence_level="Uncertified",
            fake_probability=0.48,
            real_probability=0.52,
            raw_model_score=0.48,
            uncertainty=0.96,
            reason="Classifier disagreement between face-swap and synthetic models.",
            user_facing_statement="Classifier disagreement detected.",
            evidence=["Xception: 52%", "ViT: 2%"]
        )
        resp = agent._format_response(
            decision=decision,
            quality_res=q_res,
            forensic_res={"composite_forensic_score": 0.12, "ela_analysis": {"ela_score": 0.10}},
            visual_res={},
            filename="sample.jpg",
            size=10240,
            fmt="JPEG",
            processing_time=0.15,
            img_shape=(1080, 1920, 3),
            face_bbox=(50, 50, 150, 150),
            raw_xcep_p=0.52,
            raw_vit_p=0.02,
            raw_resnet_p=0.35
        )

        required_keys = [
            "verdict", "canonical_verdict", "media_type", "task_type",
            "models", "measurements", "supporting_evidence", "contradicting_evidence",
            "unavailable_checks", "decision", "limitations", "is_provisional"
        ]
        for key in required_keys:
            self.assertIn(key, resp, f"Canonical key '{key}' missing from ImageAgent response")

        self.assertEqual(resp["canonical_verdict"], DecisionState.UNCERTAIN)
        self.assertEqual(resp["media_type"], "image")
        self.assertIsInstance(resp["models"], list)
        self.assertEqual(len(resp["models"]), 3)

        # Check model structure
        m0 = resp["models"][0]
        self.assertIn("name", m0)
        self.assertIn("task", m0)
        self.assertIn("predicted_class", m0)
        self.assertIn("raw_score", m0)
        self.assertIn("calibrated_prob", m0)
        self.assertIn("calibration_status", m0)
        self.assertIn("status", m0)
        self.assertIn("limitations", m0)

        # Check measurements structure
        meas = resp["measurements"]
        self.assertIn("resolution", meas)
        self.assertIn("blur", meas)
        self.assertIn("ela", meas)
        self.assertIn("face_detection", meas)
        self.assertEqual(meas["resolution"]["formatted"], "1920 × 1080 px")

        # Check decision object
        dec = resp["decision"]
        self.assertIn("method", dec)
        self.assertIn("score", dec)
        self.assertIn("score_type", dec)
        self.assertIn("confidence", dec)
        self.assertIn("confidence_type", dec)
        self.assertIn("criteria_met", dec)
        self.assertIn("reason", dec)

    def test_fusion_agent_preserves_canonical_contract(self):
        """FusionAgent forwards and enriches canonical contract elements."""
        fa = FusionAgent()
        primary_out = {
            "prediction": DecisionState.UNCERTAIN,
            "verdict": DecisionState.UNCERTAIN,
            "canonical_verdict": DecisionState.UNCERTAIN,
            "primary_verdict": "UNCERTAIN",
            "fake_probability": 0.48,
            "confidence": None,
            "models": [{"name": "TestModel", "task": "TestTask", "predicted_class": "REAL"}],
            "measurements": {"resolution": {"formatted": "800 × 600 px"}},
            "supporting_evidence": ["Face detected"],
            "contradicting_evidence": ["Model disagreement"],
            "unavailable_checks": ["Lip sync unavailable"],
            "decision": {"method": "SafetyDecisionSystem", "criteria_met": False},
            "limitations": ["Test limitation"],
            "media_type": "image",
            "task_type": "ai_generated_and_facial_deepfake_detection",
        }
        fused = fa.fuse("ImageAgent", primary_out)
        self.assertEqual(fused["canonical_verdict"], DecisionState.UNCERTAIN)
        self.assertIn("models", fused)
        self.assertEqual(len(fused["models"]), 1)
        self.assertIn("measurements", fused)
        self.assertIn("supporting_evidence", fused)
        self.assertIn("contradicting_evidence", fused)
        self.assertIn("unavailable_checks", fused)
        self.assertIn("decision", fused)
        self.assertIn("limitations", fused)

    def test_resolution_never_renders_object_object(self):
        """Ensure dictionary resolution formatting in report generator avoids [object Object]."""
        from backend.utils.report_generator import ReportGenerator
        res_dict = {"width": 1920, "height": 1080}
        w = res_dict.get("width")
        h = res_dict.get("height")
        formatted = f"{w} × {h} px"
        self.assertEqual(formatted, "1920 × 1080 px")
        self.assertNotEqual(formatted, "[object Object]")

if __name__ == "__main__":
    unittest.main()
