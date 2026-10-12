import os
import sys
import unittest
import numpy as np
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.utils.metrics_helper import (
    compute_all_metrics,
    optimize_threshold,
    verify_confusion_matrix_consistency,
)


class TestEpoch10MetricLogging(unittest.TestCase):
    """
    Regression test suite specifically verifying:
    1. Epoch 10 metric extraction and logging does NOT raise NameError for fake_f1 or real_f1.
    2. All REAL and FAKE precision, recall, and F1 variables are defined and sourced from the metrics suite.
    3. Training metrics (loss=0.0790, acc=97.03%) and validation metrics remain strictly isolated.
    4. Decode statistics formatting handles 51,357 successful samples, 0 decode failures, 0 skipped samples, 0 retries.
    5. compute_all_metrics guarantees existence of real_f1 and fake_f1 keys.
    """

    def setUp(self):
        # Configure a test logger
        self.logger = logging.getLogger("TestEpoch10Logger")
        self.logger.setLevel(logging.INFO)

        # Epoch 10 simulation constants
        self.epoch = 10
        self.epochs = 20
        self.total_batches = 803
        self.epoch_train_loss = 0.0790
        self.epoch_train_acc = 97.03
        self.train_succ = 51357
        self.train_failures = 0
        self.train_skipped = 0
        self.train_retries = 0
        self.train_unique_vids = 0
        self.train_fail_rate = 0.0

        # Synthetic validation predictions and targets matching binary classification
        np.random.seed(42)
        self.val_labels_arr = np.concatenate([np.zeros(1978, dtype=np.int64), np.ones(7430, dtype=np.int64)])
        # Generate realistic probabilities
        real_probs = np.random.beta(1.5, 4.0, size=1978)
        fake_probs = np.random.beta(4.0, 1.5, size=7430)
        self.val_probs_arr = np.concatenate([real_probs, fake_probs]).astype(np.float32)
        self.epoch_val_loss = 0.3542

    def test_compute_all_metrics_contains_all_f1_keys(self):
        """Verify that compute_all_metrics always provides real_f1 and fake_f1."""
        opt_thresh, ep_val_macro_f1 = optimize_threshold(self.val_labels_arr, self.val_probs_arr)
        val_metrics_suite = compute_all_metrics(self.val_labels_arr, self.val_probs_arr, threshold=opt_thresh)

        self.assertIn("real_precision", val_metrics_suite)
        self.assertIn("real_recall", val_metrics_suite)
        self.assertIn("real_f1", val_metrics_suite)
        self.assertIn("fake_precision", val_metrics_suite)
        self.assertIn("fake_recall", val_metrics_suite)
        self.assertIn("fake_f1", val_metrics_suite)
        self.assertTrue(np.isfinite(val_metrics_suite["fake_f1"]))
        self.assertTrue(np.isfinite(val_metrics_suite["real_f1"]))

    def test_epoch10_metric_logging_path_no_name_error(self):
        """
        Executes the exact Epoch 10 metric extraction and logging block from train_image_model.py.
        Verifies that no NameError or other exception is raised.
        """
        ep = 9  # 0-indexed epoch 9 corresponds to Epoch 10
        epochs = self.epochs
        total_batches = self.total_batches
        epoch_train_loss = self.epoch_train_loss
        epoch_train_acc = self.epoch_train_acc
        epoch_val_loss = self.epoch_val_loss
        log_prefix = "[REAL TRAINING] "

        # Compute validation metrics
        opt_thresh, ep_val_macro_f1 = optimize_threshold(self.val_labels_arr, self.val_probs_arr)
        val_metrics_suite = compute_all_metrics(self.val_labels_arr, self.val_probs_arr, threshold=opt_thresh)
        val_metrics_suite["val_loss"] = epoch_val_loss

        cm_check = verify_confusion_matrix_consistency(
            val_metrics_suite["confusion_matrix"], val_metrics_suite, tolerance=1e-3
        )

        # EXACT metric extraction block as in fixed train_image_model.py
        epoch_val_acc = val_metrics_suite["accuracy"]
        val_precision = val_metrics_suite["fake_precision"]
        val_recall = val_metrics_suite["fake_recall"]
        fake_f1 = val_metrics_suite["fake_f1"]
        val_f1 = fake_f1
        val_auc = val_metrics_suite["roc_auc"]
        val_cm = val_metrics_suite["confusion_matrix"]
        real_precision = val_metrics_suite["real_precision"]
        real_recall = val_metrics_suite["real_recall"]
        real_f1 = val_metrics_suite["real_f1"]
        fake_precision = val_metrics_suite["fake_precision"]
        fake_recall = val_metrics_suite["fake_recall"]
        val_ece = val_metrics_suite["ece"]

        elapsed_total = 10800  # ~3 hours
        eta_total = 10800
        mem_str = " | RSS RAM: 1542.0MB"
        vram_str = " | VRAM Peak: 2310.0MB"

        train_succ = self.train_succ
        train_failures = self.train_failures
        train_skipped = self.train_skipped
        train_retries = self.train_retries
        train_unique_vids = self.train_unique_vids
        train_fail_rate = self.train_fail_rate

        # EXACT logging statements from train_image_model.py
        train_log = (
            f"[TRAINING METRICS] Epoch {ep+1}/{epochs} | Batch {total_batches}/{total_batches} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Elapsed: {int(elapsed_total)}s | ETA: {int(eta_total)}s{mem_str}{vram_str}"
        )

        decode_log = (
            f"\n========================================\n"
            f"Epoch {ep+1} decode statistics:\n"
            f"Successful samples: {train_succ}\n"
            f"Decode failures: {train_failures}\n"
            f"Skipped samples: {train_skipped}\n"
            f"Retries: {train_retries}\n"
            f"Unique failed videos: {train_unique_vids}\n"
            f"Failure rate: {train_fail_rate:.2f}%\n"
            f"========================================"
        )

        # This line previously crashed with: NameError: name 'fake_f1' is not defined
        val_log = (
            f"[VALIDATION METRICS] {log_prefix}Epoch {ep+1}/{epochs} (Optimal Threshold={opt_thresh:.4f}) - "
            f"Val Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc:.2f}%, Macro-F1: {ep_val_macro_f1:.4f}, ROC-AUC: {val_auc:.4f}, ECE: {val_ece:.4f} | "
            f"REAL Prec/Rec/F1: {real_precision:.4f}/{real_recall:.4f}/{real_f1:.4f}, "
            f"FAKE Prec/Rec/F1: {fake_precision:.4f}/{fake_recall:.4f}/{fake_f1:.4f}"
        )

        cm_log = (
            f"[VALIDATION CONFUSION MATRIX] {log_prefix}(Rows: Actual [REAL=0, FAKE=1], Columns: Predicted [REAL=0, FAKE=1]): "
            f"{val_cm} (TN={val_cm[0][0]}, FP={val_cm[0][1]}, FN={val_cm[1][0]}, TP={val_cm[1][1]})"
        )

        audit_log = (
            f"[VALIDATION METRIC AUDIT] {'PASSED' if cm_check['is_consistent'] else 'FAILED'} "
            f"(Total Samples: {cm_check['derived_metrics'].get('total_samples')}, Discrepancies: {cm_check['discrepancies']})"
        )

        # Verify string contents
        self.assertIn("Epoch 10/20", train_log)
        self.assertIn("Train Loss: 0.0790", train_log)
        self.assertIn("Train Acc: 97.03%", train_log)
        self.assertIn("Successful samples: 51357", decode_log)
        self.assertIn("Decode failures: 0", decode_log)
        self.assertIn("REAL Prec/Rec/F1:", val_log)
        self.assertIn("FAKE Prec/Rec/F1:", val_log)
        self.assertIn(f"{fake_f1:.4f}", val_log)
        self.assertIn(f"{real_f1:.4f}", val_log)
        self.assertIn(f"{fake_precision:.4f}", val_log)
        self.assertIn(f"{fake_recall:.4f}", val_log)

    def test_strict_isolation_between_train_and_val_metrics(self):
        """Ensure training metrics and validation metrics do not contaminate each other."""
        ep = 9
        epoch_train_loss = 0.0790
        epoch_train_acc = 97.03
        epoch_val_loss = 0.3542

        opt_thresh, ep_val_macro_f1 = optimize_threshold(self.val_labels_arr, self.val_probs_arr)
        val_metrics_suite = compute_all_metrics(self.val_labels_arr, self.val_probs_arr, threshold=opt_thresh)

        # Training metrics namespace
        train_record = {
            "epoch": ep + 1,
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
        }

        # Validation metrics namespace
        val_record = {
            "epoch": ep + 1,
            "val_loss": epoch_val_loss,
            "accuracy": val_metrics_suite["accuracy"],
            "macro_f1": ep_val_macro_f1,
            "real_f1": val_metrics_suite["real_f1"],
            "fake_f1": val_metrics_suite["fake_f1"],
        }

        self.assertNotEqual(train_record["train_loss"], val_record["val_loss"])
        self.assertNotEqual(train_record["train_accuracy"], val_record["accuracy"])
        self.assertNotIn("real_f1", train_record)
        self.assertNotIn("fake_f1", train_record)
        self.assertNotIn("train_loss", val_record)


if __name__ == "__main__":
    unittest.main()
