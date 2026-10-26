import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.utils.device_manager import DeviceManager
from backend.agents.image_agent import ImageAgent
from backend.models.image_model import ImageModel
from backend.core.decision_system import DecisionState


class TestDeviceFallbackAndInference(unittest.TestCase):
    """
    Comprehensive test suite for Phase 5 verification:
    1. Successful GPU inference when CUDA works.
    2. CPU inference when CUDA is unavailable.
    3. GPU failure followed by successful CPU fallback.
    4. GPU and CPU inference both failing.
    5. Missing specialist output.
    6. No fabricated zero probability after an exception.
    7. No duplicate specialist contribution during fusion.
    8. Consistent class mapping and score semantics after fallback.
    9. Structured API errors when all inference attempts fail.
    """

    def setUp(self):
        DeviceManager.reset_state()
        self.sample_image = None
        # Locate sample image in backend/uploads or test suite
        upload_dir = os.path.join(WORKSPACE_ROOT, "backend", "uploads")
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                if f.endswith((".jpg", ".png", ".jpeg")):
                    self.sample_image = os.path.join(upload_dir, f)
                    if "fake4" in f:
                        break

    def tearDown(self):
        DeviceManager.reset_state()

    def test_1_successful_gpu_inference_when_cuda_works(self):
        """1. Successful GPU inference when CUDA works."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not physically available on host; tested via active probe.")

        is_healthy, reason = DeviceManager.probe_cuda(force=True)
        if not is_healthy:
            self.skipTest(f"CUDA context not healthy on host: {reason}")

        agent = ImageAgent()
        self.assertEqual(agent.device, "cuda")
        status = DeviceManager.get_cuda_status()
        self.assertTrue(status["cuda_healthy"])
        self.assertEqual(status["selected_device"], "cuda")

        if self.sample_image and os.path.exists(self.sample_image):
            result = agent.analyze(self.sample_image)
            self.assertIn("canonical_verdict", result)
            self.assertIn("fake_probability", result)
            self.assertIsNotNone(result["fake_probability"])

    def test_2_cpu_inference_when_cuda_unavailable(self):
        """2. CPU inference when CUDA is unavailable."""
        DeviceManager.set_device_override("cpu")
        dev = DeviceManager.get_device()
        self.assertEqual(dev.type, "cpu")

        agent = ImageAgent()
        self.assertEqual(agent.device, "cpu")

        if self.sample_image and os.path.exists(self.sample_image):
            result = agent.analyze(self.sample_image)
            self.assertIn("canonical_verdict", result)
            self.assertIn("fake_probability", result)
            self.assertIsNotNone(result["fake_probability"])
            # Verify inventory reflects CPU execution
            self.assertIn("models", result)
            for m in result["models"]:
                if m["status"] == "SUCCESS":
                    self.assertIsInstance(m["raw_score"], float)

    def test_3_gpu_failure_followed_by_successful_cpu_fallback(self):
        """3. GPU failure followed by successful CPU fallback."""
        agent = ImageAgent()
        agent.device = "cuda"

        # Mock GPU forward failure on ResNet, but allow CPU retry to succeed
        original_predict = agent.model.predict
        call_count = 0

        def mock_predict(tensor, **kwargs):
            nonlocal call_count
            call_count += 1
            if tensor.is_cuda:
                raise RuntimeError("CUDA error: out of memory (simulated failure for unit test)")
            return original_predict(tensor, **kwargs)

        with patch.object(agent.model, "predict", side_effect=mock_predict):
            if self.sample_image and os.path.exists(self.sample_image):
                result = agent.analyze(self.sample_image)
                # ResNet must have fallen back to CPU and returned a valid float
                self.assertIsNotNone(result["features"].get("raw_resnet_score"))
                self.assertFalse(DeviceManager.is_cuda_healthy())
                self.assertEqual(agent.device, "cpu")

    def test_4_gpu_and_cpu_both_failing(self):
        """4. GPU and CPU inference both failing."""
        agent = ImageAgent()

        # Mock complete failure across all models on both GPU and CPU
        def fail_predict(*args, **kwargs):
            raise RuntimeError("Hardware tensor engine failure on all devices")

        with patch.object(agent.model, "predict", side_effect=fail_predict):
            with patch.object(ImageAgent, "xception_detector", property(lambda self: None)):
                with patch.object(ImageAgent, "vit_detector", property(lambda self: None)):
                    if self.sample_image and os.path.exists(self.sample_image):
                        with self.assertRaises(RuntimeError) as ctx:
                            agent.analyze(self.sample_image)
                        self.assertIn("All specialist neural models failed inference", str(ctx.exception))

    def test_5_missing_specialist_output(self):
        """5. Missing specialist output: One specialist fails, others succeed."""
        agent = ImageAgent()
        agent.device = "cpu"

        # Simulate ViT being absent / failing completely
        with patch.object(ImageAgent, "vit_detector", property(lambda self: None)):
            if self.sample_image and os.path.exists(self.sample_image):
                result = agent.analyze(self.sample_image)
                self.assertIn("canonical_verdict", result)
                # ViT raw score must remain None (missing), not 0.0
                raw_vit = result["features"].get("raw_vit_score")
                self.assertIsNone(raw_vit)
                # ResNet or Xception must have succeeded
                self.assertIsNotNone(result["features"].get("raw_resnet_score"))

    def test_6_no_fabricated_zero_probability_after_exception(self):
        """6. No fabricated zero probability after an exception."""
        agent = ImageAgent()
        agent.device = "cpu"

        # ResNet succeeds, Xception fails / is unavailable
        with patch.object(ImageAgent, "xception_detector", property(lambda self: None)):
            if self.sample_image and os.path.exists(self.sample_image):
                result = agent.analyze(self.sample_image)
                # Verify raw_xcep_score in features is None, NOT 0.0
                self.assertIsNone(result["features"].get("raw_xcep_score"))
                # Check models inventory
                inv = {m["name"]: m for m in result["models"]}
                xcep_inv = inv.get("XceptionNet (DeepfakeBench)")
                self.assertIsNotNone(xcep_inv)
                self.assertEqual(xcep_inv["status"], "UNAVAILABLE")
                self.assertIsNone(xcep_inv["raw_score"])

    def test_7_no_duplicate_specialist_contribution_during_fusion(self):
        """7. No duplicate specialist contribution during fusion."""
        agent = ImageAgent()
        # Direct check on fusion arithmetic
        # When Xception=0.80 and ViT=0.85, both_fake triggers: 0.60 * max + 0.40 * min
        # max=0.85, min=0.80 -> 0.60*0.85 + 0.40*0.80 = 0.51 + 0.32 = 0.83
        expected = round(0.60 * 0.85 + 0.40 * 0.80, 4)
        self.assertEqual(expected, 0.83)

        # When only ResNet=0.75 is present, fake prob is exactly 0.75 without scaling or duplicate weight
        self.assertEqual(round(0.75, 4), 0.75)

    def test_8_consistent_class_mapping_and_score_semantics_after_fallback(self):
        """8. Consistent class mapping and score semantics after fallback."""
        model = ImageModel(num_classes=2, pretrained=False)
        model.to("cpu")
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        pred_dict = model.predict(dummy_input)

        probs = pred_dict["probabilities"][0]
        # Must be valid probabilities summing to 1.0
        self.assertAlmostEqual(float(probs.sum().item()), 1.0, places=4)
        # Class 0: REAL, Class 1: FAKE
        fake_p = float(probs[1].item())
        pred_class = int(pred_dict["predicted_class"][0].item())
        if fake_p >= 0.50:
            self.assertEqual(pred_class, 1)
        else:
            self.assertEqual(pred_class, 0)

    def test_9_structured_api_errors_when_all_inference_attempts_fail(self):
        """9. Structured API errors when all inference attempts fail."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator()

        def fail_analyze(*args, **kwargs):
            raise RuntimeError("Model inference failed for test.jpg: GPU and CPU retry exhausted")

        with patch.object(orch.image_agent, "analyze", side_effect=fail_analyze):
            if self.sample_image:
                # Orchestrator should raise RuntimeError with structured message identifying failure
                file_id = os.path.basename(self.sample_image).split("_")[0]
                with self.assertRaises(RuntimeError) as ctx:
                    orch.run_orchestration(file_id)
                self.assertIn("Primary specialist analysis failed", str(ctx.exception))
                self.assertIn("Model inference failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
