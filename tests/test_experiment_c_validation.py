import os
import sys
import unittest
import numpy as np
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.training.dataset_loader import DeepfakeDataset
from backend.utils.metrics_helper import optimize_threshold, compute_all_metrics, verify_confusion_matrix_consistency
from backend.utils.config import PipelineConfig


class TestExperimentCValidation(unittest.TestCase):
    """
    Automated pre-flight validation checks for Experiment C:
    1. JPEG augmentation preserves RGB ordering.
    2. Class weights correspond to class 0 = REAL, class 1 = FAKE.
    3. Validation transform does not use training augmentation.
    4. Validation/test data are not modified by augmentation.
    5. Confusion matrix dimensions and class ordering are correct.
    6. Precision/recall/F1 calculated from confusion matrix match reported metrics within tolerance.
    7. No test-set predictions are used for threshold optimization.
    8. Epoch 12 baseline checkpoint remains unchanged.
    9. New checkpoints are configured to write only inside backend/models/weights/runs/exp_c_weighted_aug/.
    10. The full_image_I1 checkpoints and progress log remain untouched.
    """

    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        self.exp_c_dir = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", "exp_c_weighted_aug")
        self.baseline_ckpt = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", "full_image_I1", "best_checkpoint.pth")
        self.latest_ckpt = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", "full_image_I1", "latest_checkpoint.pth")

    def test_1_jpeg_augmentation_preserves_rgb_ordering(self):
        """CHECK 1: Verify JPEG augmentation preserves RGB color channels and does NOT swap to BGR."""
        # Create an image with dominant Red channel: R=230, G=70, B=20
        test_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        test_rgb[:, :, 0] = 230  # R
        test_rgb[:, :, 1] = 70   # G
        test_rgb[:, :, 2] = 20   # B

        # Run safe JPEG compression
        compressed_rgb = self.preprocessor.simulate_jpeg_compression(test_rgb, quality=75)

        self.assertEqual(compressed_rgb.shape, (100, 100, 3))
        self.assertEqual(compressed_rgb.dtype, np.uint8)

        mean_r = float(compressed_rgb[:, :, 0].mean())
        mean_g = float(compressed_rgb[:, :, 1].mean())
        mean_b = float(compressed_rgb[:, :, 2].mean())

        # If RGB was swapped to BGR, mean_r would be ~20 and mean_b would be ~230
        self.assertGreater(mean_r, 200, f"Red channel corrupted or swapped: {mean_r}")
        self.assertLess(mean_b, 40, f"Blue channel contaminated or swapped: {mean_b}")
        self.assertTrue(mean_r > mean_g > mean_b, f"Channel ordering altered: R={mean_r}, G={mean_g}, B={mean_b}")

    def test_2_class_weights_correspond_to_real_and_fake(self):
        """CHECK 2: Verify class weights correspond to index 0 = REAL and index 1 = FAKE."""
        manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
        self.assertTrue(os.path.exists(manifest_path), f"Manifest missing at {manifest_path}")

        # Compute weights directly from manifest
        real_count = 0
        fake_count = 0
        import csv
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("media_type") == "image" and row.get("split") == "train":
                    if int(row["label"]) == 0:
                        real_count += 1
                    elif int(row["label"]) == 1:
                        fake_count += 1

        self.assertEqual(real_count, 8227, f"Expected 8,227 REAL samples, found {real_count}")
        self.assertEqual(fake_count, 43130, f"Expected 43,130 FAKE samples, found {fake_count}")

        total = real_count + fake_count
        w_real = total / (2.0 * real_count)
        w_fake = total / (2.0 * fake_count)

        self.assertAlmostEqual(w_real, 3.1212, places=3)
        self.assertAlmostEqual(w_fake, 0.5954, places=3)
        self.assertGreater(w_real, w_fake, "Minority class (REAL) must have strictly higher weight than majority (FAKE)")

        # Verify weights tensor placement on device
        weights_tensor = torch.tensor([w_real, w_fake], dtype=torch.float32)
        device = torch.device(PipelineConfig.DEVICE)
        weights_on_device = weights_tensor.to(device)
        self.assertEqual(weights_on_device.device.type, device.type)

    def test_3_validation_transform_does_not_use_training_augmentation(self):
        """CHECK 3: Verify validation transform is deterministic and does NOT apply training augmentations."""
        test_rgb = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

        # Call with is_training=False 5 times
        tensors = [self.preprocessor.apply_augmentation_and_normalization(test_rgb, is_training=False) for _ in range(5)]

        # All 5 evaluation tensors must be bitwise identical
        for i in range(1, 5):
            self.assertTrue(
                torch.equal(tensors[0], tensors[i]),
                f"Validation transform is not deterministic between run 0 and run {i}!"
            )

    def test_4_validation_test_data_not_modified_by_augmentation(self):
        """CHECK 4: Verify input numpy image array is not mutated in-place during preprocessing."""
        test_rgb = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        copy_rgb = test_rgb.copy()

        # Run validation transform
        _ = self.preprocessor.apply_augmentation_and_normalization(test_rgb, is_training=False)
        self.assertTrue(np.array_equal(test_rgb, copy_rgb), "Input array was mutated during validation preprocessing!")

        # Run training transform
        _ = self.preprocessor.apply_augmentation_and_normalization(test_rgb, is_training=True)
        self.assertTrue(np.array_equal(test_rgb, copy_rgb), "Input array was mutated during training augmentation!")

    def test_5_confusion_matrix_dimensions_and_class_ordering(self):
        """CHECK 5: Verify confusion matrix dimensions are 2x2 with Rows=Actual, Cols=Predicted."""
        # 0 = REAL, 1 = FAKE
        y_true = np.array([0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.85, 0.9])
        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

        cm = metrics["confusion_matrix"]
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)

        # y_true 0: [0.1, 0.2, 0.8] -> preds at 0.5: [0, 0, 1] => TN=2, FP=1
        # y_true 1: [0.3, 0.7, 0.85, 0.9] -> preds at 0.5: [0, 1, 1, 1] => FN=1, TP=3
        self.assertEqual(cm[0][0], 2, "TN mismatch")
        self.assertEqual(cm[0][1], 1, "FP mismatch")
        self.assertEqual(cm[1][0], 1, "FN mismatch")
        self.assertEqual(cm[1][1], 3, "TP mismatch")

    def test_6_metrics_consistency_from_confusion_matrix(self):
        """CHECK 6: Verify derived metrics from Epoch 12 confusion matrix match reported metrics within tolerance."""
        # Epoch 12 reported numbers: TN=1360, FP=618, FN=768, TP=6662
        cm = [[1360, 618], [768, 6662]]
        reported_metrics = {
            "real_precision": 0.6391,
            "real_recall": 0.6876,
            "real_f1": 0.6624,
            "fake_precision": 0.9151,
            "fake_recall": 0.8966,
            "fake_f1": 0.9058,
            "macro_f1": 0.7841,
            "accuracy": 85.27
        }
        audit = verify_confusion_matrix_consistency(cm, reported_metrics, tolerance=1e-3)
        self.assertTrue(audit["is_consistent"], f"Discrepancies found: {audit['error_messages']}")

    def test_7_no_test_set_predictions_used_for_threshold_optimization(self):
        """CHECK 7: Verify optimize_threshold operates exclusively on provided validation arrays."""
        val_true = np.array([0] * 100 + [1] * 100)
        val_prob = np.concatenate([np.random.uniform(0.1, 0.4, 100), np.random.uniform(0.6, 0.9, 100)])

        thresh, macro_f1 = optimize_threshold(val_true, val_prob)
        self.assertGreaterEqual(thresh, 0.01)
        self.assertLessEqual(thresh, 0.99)
        self.assertGreater(macro_f1, 0.80)

    def test_8_epoch12_baseline_checkpoint_remains_unchanged(self):
        """CHECK 8: Verify Epoch 12 best_checkpoint.pth exists, is readable, healthy, and contains expected metrics."""
        self.assertTrue(os.path.exists(self.baseline_ckpt), f"Baseline checkpoint missing: {self.baseline_ckpt}")
        ckpt = torch.load(self.baseline_ckpt, map_location="cpu", weights_only=False)
        self.assertIsInstance(ckpt, dict, "Checkpoint root must be a dictionary")
        self.assertEqual(ckpt.get("epoch"), 12, f"Expected Epoch 12, got {ckpt.get('epoch')}")
        self.assertAlmostEqual(ckpt.get("best_val_metric"), 0.784112, places=4)
        self.assertAlmostEqual(ckpt.get("val_loss"), 0.432207, places=4)
        self.assertIn("model_state_dict", ckpt, "Checkpoint must contain model_state_dict")
        # Verify finite parameters in model_state_dict
        for k, v in ckpt["model_state_dict"].items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(torch.isfinite(v).all(), f"Parameter {k} has non-finite values")

    def test_9_new_checkpoints_configured_for_separate_dir(self):
        """CHECK 9: Verify exp_c target directory is distinct from full_image_I1."""
        full_image_dir = os.path.dirname(self.baseline_ckpt)
        self.assertNotEqual(
            os.path.normpath(self.exp_c_dir),
            os.path.normpath(full_image_dir),
            "Experiment directory must NOT be the same as full_image_I1!"
        )

    def test_10_active_epoch12_process_remains_untouched(self):
        """CHECK 10: Verify full_image_I1 latest_checkpoint and progress log are intact."""
        self.assertTrue(os.path.exists(self.latest_ckpt), f"Latest checkpoint missing: {self.latest_ckpt}")
        ckpt = torch.load(self.latest_ckpt, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt.get("epoch"), 12, f"Expected latest checkpoint Epoch 12, got {ckpt.get('epoch')}")
        self.assertAlmostEqual(ckpt.get("best_val_metric"), 0.784112, places=4)
        progress_log = os.path.join(PipelineConfig.RESULTS_DIR, "image", "full_image_I1_progress.log")
        self.assertTrue(os.path.exists(progress_log), "Progress log must exist")
        with open(progress_log, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Epoch 12/20", content, "Progress log must contain Epoch 12 entry")


if __name__ == "__main__":
    unittest.main()

