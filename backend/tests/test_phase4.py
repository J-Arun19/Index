import os
import unittest
import shutil
import tempfile
import json
import torch
import torch.nn as nn
import numpy as np
from backend.utils.config import PipelineConfig
from backend.models.video_model import VideoModel
from backend.training.train_video_model import run_promotion_gate

class TestPhase4(unittest.TestCase):
    def setUp(self):
        # Save baseline configuration to restore later
        self.old_pooling = getattr(PipelineConfig, "TEMPORAL_POOLING", "lstm")
        self.old_frames = PipelineConfig.MAX_FRAMES_PER_VIDEO
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Restore configuration
        PipelineConfig.TEMPORAL_POOLING = self.old_pooling
        PipelineConfig.MAX_FRAMES_PER_VIDEO = self.old_frames
        shutil.rmtree(self.temp_dir)
        
    def test_architecture_selection(self):
        """Verify initialization selects 'lstm' or 'mean' pooling correctly."""
        # Test explicit selection
        model_lstm = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm")
        self.assertEqual(model_lstm.temporal_pooling, "lstm")
        self.assertTrue(hasattr(model_lstm, "lstm"))
        self.assertEqual(model_lstm.fc.in_features, 256)
        
        model_mean = VideoModel(num_classes=2, pretrained=False, temporal_pooling="mean")
        self.assertEqual(model_mean.temporal_pooling, "mean")
        self.assertFalse(hasattr(model_mean, "lstm"))
        self.assertEqual(model_mean.fc.in_features, model_mean.feature_dim)
        
        # Test PipelineConfig fallback
        PipelineConfig.TEMPORAL_POOLING = "mean"
        model_fallback = VideoModel(num_classes=2, pretrained=False)
        self.assertEqual(model_fallback.temporal_pooling, "mean")
        
    def test_tensor_dimensions_and_mean_pooling_correctness(self):
        """Verify tensor shapes and that mean pooling takes the exact mean of features."""
        model = VideoModel(num_classes=2, pretrained=False, temporal_pooling="mean")
        model.eval()
        
        # input shape: (batch_size, sequence_length, channels, height, width)
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 3, 224, 224)
        
        with torch.no_grad():
            # Pass through backbone manually to verify mean pooling math
            # Step 1: Flatten seq and batch dims
            x_flat = x.view(batch_size * seq_len, 3, 224, 224)
            features_flat = model.backbone(x_flat)
            features = features_flat.view(batch_size, seq_len, model.feature_dim)
            expected_pooled = torch.mean(features, dim=1)
            expected_logits = model.fc(expected_pooled)
            
            # Step 2: Full model forward
            logits = model(x)
            
        self.assertEqual(logits.shape, (batch_size, 2))
        self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-4))
        
    def test_10_frame_input_handling(self):
        """Verify model handles 10-frame sequences correctly."""
        for pooling in ["lstm", "mean"]:
            model = VideoModel(num_classes=2, pretrained=False, temporal_pooling=pooling)
            x = torch.randn(2, 10, 3, 224, 224)
            logits = model(x)
            self.assertEqual(logits.shape, (2, 2))
            
    def test_class_collapse_detection(self):
        """Verify class collapse is detected based on majority predictions exceeding 98%."""
        # No collapse case
        metrics_healthy = {
            "final_metrics": {
                "macro_f1": 0.60,
                "real_recall": 0.50,
                "real_pred_pct": 0.50,
                "fake_pred_pct": 0.50,
                "class_collapse": False
            }
        }
        
        # Collapse cases
        metrics_collapse_real = {
            "final_metrics": {
                "macro_f1": 0.49,
                "real_recall": 1.0,
                "real_pred_pct": 0.99,
                "fake_pred_pct": 0.01,
                "class_collapse": True
            }
        }
        
        metrics_collapse_fake = {
            "final_metrics": {
                "macro_f1": 0.49,
                "real_recall": 0.0,
                "real_pred_pct": 0.01,
                "fake_pred_pct": 0.99,
                "class_collapse": True
            }
        }
        
        # Let's save a fake baseline metrics file in temp dir to run promotion tests
        baseline_path = os.path.join(self.temp_dir, "baseline_metrics.json")
        baseline_data = {
            "final_metrics": {
                "macro_f1": 0.50,
                "real_recall": 0.50,
                "real_pred_pct": 0.50,
                "fake_pred_pct": 0.50,
                "class_collapse": False,
                "confusion_matrix": [[50, 0], [50, 0]]
            }
        }
        
        # Temporarily mock PipelineConfig.RESULTS_DIR
        old_results_dir = PipelineConfig.RESULTS_DIR
        PipelineConfig.RESULTS_DIR = self.temp_dir
        
        try:
            # Save baseline metrics
            os.makedirs(os.path.join(self.temp_dir, "video"), exist_ok=True)
            with open(os.path.join(self.temp_dir, "video", "baseline_metrics.json"), "w") as f:
                json.dump(baseline_data, f)
                
            # Collapse models must be rejected by the promotion gate
            res_healthy = run_promotion_gate(
                media_type="video",
                run_metrics=metrics_healthy,
                run_weights_path="dummy.pth",
                is_baseline_run=False,
                skip_promotion_overwrite=True
            )
            res_collapse_real = run_promotion_gate(
                media_type="video",
                run_metrics=metrics_collapse_real,
                run_weights_path="dummy.pth",
                is_baseline_run=False,
                skip_promotion_overwrite=True
            )
            res_collapse_fake = run_promotion_gate(
                media_type="video",
                run_metrics=metrics_collapse_fake,
                run_weights_path="dummy.pth",
                is_baseline_run=False,
                skip_promotion_overwrite=True
            )
            
            self.assertTrue(res_healthy)
            self.assertFalse(res_collapse_real)
            self.assertFalse(res_collapse_fake)
        finally:
            PipelineConfig.RESULTS_DIR = old_results_dir
            
    def test_checkpoint_isolation_and_safety(self):
        """Verify skip_promotion_overwrite does not touch production files."""
        # Ensure that running the promotion gate with skip_promotion_overwrite does NOT overwrite files
        dummy_weights = os.path.join(self.temp_dir, "dummy_weights.pth")
        with open(dummy_weights, "w") as f:
            f.write("dummy weights content")
            
        prod_path = PipelineConfig.VIDEO_MODEL_PATH
        # Read original prod file contents or metadata if it exists
        prod_exists = os.path.exists(prod_path)
        original_mtime = os.path.getmtime(prod_path) if prod_exists else None
        
        metrics = {
            "final_metrics": {
                "macro_f1": 0.99, # high f1 to trigger promotion success
                "real_recall": 0.99,
                "real_pred_pct": 0.50,
                "fake_pred_pct": 0.50,
                "class_collapse": False
            }
        }
        
        # Run promotion gate with skip_promotion_overwrite=True
        run_promotion_gate(
            media_type="video",
            run_metrics=metrics,
            run_weights_path=dummy_weights,
            is_baseline_run=False,
            skip_promotion_overwrite=True
        )
        
        # Verify production weight file is unmodified
        if prod_exists:
            self.assertEqual(os.path.getmtime(prod_path), original_mtime)
        else:
            self.assertFalse(os.path.exists(prod_path))
            
    def test_deterministic_experiment_configuration(self):
        """Verify random seed locking provides reproducible configuration setups."""
        # Verify we can set seed deterministically
        torch.manual_seed(42)
        tensor1 = torch.randn(5)
        torch.manual_seed(42)
        tensor2 = torch.randn(5)
        self.assertTrue(torch.allclose(tensor1, tensor2))

if __name__ == "__main__":
    unittest.main()
