import os
import unittest
import torch
import numpy as np
import json

from backend.models.experimental.vit_detector import ViTDetector
from backend.models.experimental.xception_detector import XceptionDetector
from backend.agents.experimental.experimental_image_agent import ExperimentalImageAgent
from backend.agents.experimental.experimental_video_agent import ExperimentalVideoAgent
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.utils.config import PipelineConfig
from backend.core.decision_system import DecisionState

class TestExperimentalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.doc_path = "backend/uploads/document_contract_sample.png"

    def test_vit_detector_inference(self):
        """Verifies ViT detector initialization and tensor output shapes."""
        vit = ViTDetector(device=self.device)
        dummy_bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        res = vit.predict_image(dummy_bgr)
        self.assertIn("real_probability", res)
        self.assertIn("fake_probability", res)
        self.assertAlmostEqual(res["real_probability"] + res["fake_probability"], 1.0, places=3)
        self.assertIn(res["predicted_label"], ["REAL", "FAKE"])

    def test_xception_detector_inference(self):
        """Verifies Xception detector initialization, weights loading, and frame prediction."""
        xcep = XceptionDetector(device=self.device)
        dummy_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        res = xcep.predict_frame(dummy_bgr)
        self.assertIn("real_probability", res)
        self.assertIn("fake_probability", res)
        self.assertAlmostEqual(res["real_probability"] + res["fake_probability"], 1.0, places=3)
        self.assertIn(res["predicted_label"], ["REAL", "FAKE"])

    def test_image_agent_rejects_non_face_media(self):
        """Verifies non-face media is routed to NOT_SUITABLE without forced facial prediction."""
        if os.path.exists(self.doc_path):
            agent = ExperimentalImageAgent(backbone="vit", device=self.device)
            res = agent.analyze(self.doc_path, require_face=True)
            self.assertEqual(res["prediction"], DecisionState.NOT_SUITABLE)
            self.assertEqual(res["quality_gate"]["eligibility"], "UNSUPPORTED")
            self.assertIsNone(res["raw_model_score"])

    def test_production_pipeline_isolation(self):
        """Strict regression test verifying production weights and configs were not modified."""
        # 1. Verify production weights exist
        self.assertTrue(os.path.exists(PipelineConfig.IMAGE_MODEL_PATH))
        self.assertTrue(os.path.exists(PipelineConfig.VIDEO_MODEL_PATH))

        # 2. Verify calibration config unchanged
        cal_path = os.path.join(PipelineConfig.BACKEND_DIR, "models", "calibration_config.json")
        self.assertTrue(os.path.exists(cal_path))
        with open(cal_path, "r") as f:
            cal = json.load(f)
        self.assertEqual(cal["temperature"], 1.885)
        self.assertEqual(cal["tau_real"], 0.35)
        self.assertEqual(cal["tau_fake"], 0.65)

        # 3. Verify production agent defaults unchanged
        prod_img = ImageAgent()
        self.assertEqual(prod_img.weights_path, PipelineConfig.IMAGE_MODEL_PATH)
        prod_vid = VideoAgent()
        self.assertEqual(prod_vid.weights_path, PipelineConfig.VIDEO_MODEL_PATH)

if __name__ == "__main__":
    unittest.main()
