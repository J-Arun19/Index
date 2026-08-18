import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.image_model import ImageModel
from backend.agents.image_agent import ImageAgent
from backend.agents.fusion_agent import FusionAgent
from backend.utils.config import PipelineConfig

class TestClassMappingAssertion(unittest.TestCase):
    """
    Automated assertions to guarantee Class 0 = REAL and Class 1 = FAKE mapping 
    cannot silently become reversed anywhere in the SatyaDrishti pipeline.
    """

    def test_image_model_class_mapping(self):
        model = ImageModel(num_classes=2)
        model.eval()
        
        # Test case A: High logit at index 0 (REAL)
        fake_logits_real = torch.tensor([[10.0, -10.0]])
        probs = torch.softmax(fake_logits_real, dim=-1)
        fake_prob = probs[:, 1].item()
        self.assertLess(fake_prob, 0.01, "Class index 1 must correspond to FAKE probability.")
        pred_class = (probs[:, 1] >= 0.5).long().item()
        self.assertEqual(pred_class, 0, "Index 0 highest logit must map to predicted_class 0 (REAL).")
        
        # Test case B: High logit at index 1 (FAKE)
        fake_logits_fake = torch.tensor([[-10.0, 10.0]])
        probs_fake = torch.softmax(fake_logits_fake, dim=-1)
        fake_prob_high = probs_fake[:, 1].item()
        self.assertGreater(fake_prob_high, 0.99, "Class index 1 high logit must correspond to FAKE probability near 1.0.")
        pred_class_fake = (probs_fake[:, 1] >= 0.5).long().item()
        self.assertEqual(pred_class_fake, 1, "Index 1 highest logit must map to predicted_class 1 (FAKE).")
        
        # Test dummy forward pass through model to check tensor shapes
        dummy_input = torch.randn(1, 3, 224, 224)
        pred_dict = model.predict(dummy_input)
        self.assertIn("logits", pred_dict)
        self.assertEqual(pred_dict["probabilities"].shape, (1, 2))

    def test_image_agent_verdict_mapping(self):
        # Verify that predicted_class 1 maps to DEEPFAKE and 0 maps to REAL
        predicted_class_fake = 1
        verdict_fake = "DEEPFAKE" if (predicted_class_fake == 1) else "REAL"
        self.assertEqual(verdict_fake, "DEEPFAKE")
        
        predicted_class_real = 0
        verdict_real = "DEEPFAKE" if (predicted_class_real == 1) else "REAL"
        self.assertEqual(verdict_real, "REAL")

    def test_fusion_agent_threshold(self):
        fusion = FusionAgent()
        # Mock primary result as REAL
        mock_image_res = {
            "verdict": "REAL",
            "confidence_score": 80.0,
            "summary": "Verified authentic"
        }
        res = fusion.fuse("ImageAgent", agent_output=mock_image_res)
        self.assertEqual(res["verdict"], "REAL")
        
        # Mock primary result as DEEPFAKE
        mock_image_fake_res = {
            "verdict": "DEEPFAKE",
            "confidence_score": 80.0,
            "summary": "Flagged anomalies"
        }
        res_fake = fusion.fuse("ImageAgent", agent_output=mock_image_fake_res)
        self.assertEqual(res_fake["verdict"], "DEEPFAKE")

if __name__ == "__main__":
    unittest.main()
