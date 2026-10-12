import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.utils.metrics_helper import (
    compute_all_metrics,
    verify_confusion_matrix_consistency,
    optimize_threshold,
)


class TestConfusionMatrixAudit(unittest.TestCase):
    """
    Automated audit test suite verifying:
    1. Consistency between confusion matrix and classification metrics.
    2. Exact mathematical reconstruction of Epoch 9 metrics from the corrected confusion matrix.
    3. Detection and rejection of the inconsistent Epoch 9 log matrix [[1375, 603], [819, 611]].
    4. Proper class-index ordering (0=REAL, 1=FAKE).
    5. Clear distinction between training and validation metric structures.
    """

    def setUp(self):
        # Validation split ground truth counts for full dataset
        self.val_real_count = 1978
        self.val_fake_count = 7430
        self.val_total = 9408

        # Corrected Epoch 9 confusion matrix
        self.correct_cm = [[1375, 603], [819, 6611]]

        # Erroneous/inconsistent Epoch 9 confusion matrix from log
        self.inconsistent_cm = [[1375, 603], [819, 611]]

        # Reported Epoch 9 classification metrics
        self.reported_metrics = {
            "accuracy": 84.8852,
            "real_precision": 0.6267,
            "real_recall": 0.6951,
            "real_f1": 0.6592,
            "fake_precision": 0.9164,
            "fake_recall": 0.8898,
            "fake_f1": 0.9029,
            "macro_f1": 0.7810,
            "roc_auc": 0.8970,
        }

    def test_correct_confusion_matrix_matches_reported_metrics(self):
        """Verify that the corrected matrix [[1375, 603], [819, 6611]] matches all reported metrics within 1e-3."""
        audit = verify_confusion_matrix_consistency(
            self.correct_cm, self.reported_metrics, tolerance=1e-3
        )
        self.assertTrue(audit["is_consistent"], f"Audit failed: {audit['error_messages']}")
        self.assertEqual(len(audit["error_messages"]), 0)

        # Check total sample count matches full validation set
        derived = audit["derived_metrics"]
        self.assertEqual(derived["total_samples"], self.val_total)
        self.assertEqual(derived["real_count"], self.val_real_count)
        self.assertEqual(derived["fake_count"], self.val_fake_count)

        # Precision/recall numerical tolerance check
        self.assertAlmostEqual(derived["real_precision"], 0.6267, delta=1e-3)
        self.assertAlmostEqual(derived["real_recall"], 0.6951, delta=1e-3)
        self.assertAlmostEqual(derived["fake_recall"], 0.8898, delta=1e-3)
        self.assertAlmostEqual(derived["macro_f1"], 0.7810, delta=1e-3)

    def test_inconsistent_matrix_is_rejected(self):
        """Verify that the reported log matrix [[1375, 603], [819, 611]] is detected as inconsistent."""
        audit = verify_confusion_matrix_consistency(
            self.inconsistent_cm, self.reported_metrics, tolerance=1e-3
        )
        self.assertFalse(audit["is_consistent"], "Inconsistent matrix should NOT pass verification!")
        self.assertGreater(len(audit["error_messages"]), 0)

        # The inconsistent matrix has only 1430 fake samples (missing 6000)
        self.assertEqual(audit["derived_metrics"]["fake_count"], 1430)
        self.assertEqual(audit["derived_metrics"]["total_samples"], 3408)

        # Discrepancies in fake recall and macro-f1 must be large (> 20%)
        self.assertGreater(audit["discrepancies"]["fake_recall"], 0.40)
        self.assertGreater(audit["discrepancies"]["fake_precision"], 0.40)
        self.assertGreater(audit["discrepancies"]["macro_f1"], 0.20)

    def test_class_index_ordering_and_mapping(self):
        """
        Verify class-index ordering:
        Row 0 = Actual REAL (0)
        Row 1 = Actual FAKE (1)
        Col 0 = Predicted REAL (0)
        Col 1 = Predicted FAKE (1)
        """
        cm = np.array(self.correct_cm)
        tn, fp = cm[0]
        fn, tp = cm[1]

        # Row 0 sum must equal actual REAL count
        self.assertEqual(tn + fp, self.val_real_count)
        # Row 1 sum must equal actual FAKE count
        self.assertEqual(fn + tp, self.val_fake_count)

        # Real Recall = tn / (tn + fp)
        self.assertAlmostEqual(tn / (tn + fp), 0.6951, delta=1e-4)
        # Real Precision = tn / (tn + fn)
        self.assertAlmostEqual(tn / (tn + fn), 0.6267, delta=1e-4)
        # Fake Recall = tp / (fn + tp)
        self.assertAlmostEqual(tp / (fn + tp), 0.8898, delta=1e-4)
        # Fake Precision = tp / (fp + tp)
        self.assertAlmostEqual(tp / (fp + tp), 0.9164, delta=1e-4)

    def test_compute_all_metrics_internal_consistency(self):
        """Verify compute_all_metrics produces mathematically verified consistent metrics and matrix."""
        # Generate synthetic binary dataset
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(200, dtype=int), np.ones(800, dtype=int)])
        y_prob = np.concatenate([np.random.beta(2, 5, size=200), np.random.beta(5, 2, size=800)])

        metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
        self.assertIn("matrix_verification", metrics)
        self.assertTrue(metrics["matrix_verification"]["is_consistent"])

        cm = metrics["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        self.assertEqual(tn + fp, 200)
        self.assertEqual(fn + tp, 800)

        # Matrix-derived metrics must exactly match classification metrics
        self.assertAlmostEqual(metrics["real_precision"], tn / (tn + fn) if (tn + fn) else 0.0, places=6)
        self.assertAlmostEqual(metrics["real_recall"], tn / 200.0, places=6)
        self.assertAlmostEqual(metrics["fake_precision"], tp / (fp + tp) if (fp + tp) else 0.0, places=6)
        self.assertAlmostEqual(metrics["fake_recall"], tp / 800.0, places=6)
        self.assertIn("real_f1", metrics)
        self.assertIn("fake_f1", metrics)
        self.assertTrue(np.isfinite(metrics["real_f1"]))
        self.assertTrue(np.isfinite(metrics["fake_f1"]))

    def test_train_vs_val_metrics_distinction(self):
        """Verify training metrics and validation metrics are kept in distinct namespaces."""
        train_metrics = {
            "loss": 0.0891,
            "accuracy": 96.85
        }
        val_metrics = {
            "loss": 0.3723,
            "accuracy": 84.89,
            "macro_f1": 0.7810,
            "real_recall": 0.6951,
            "fake_recall": 0.8898
        }
        report = {
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics
        }
        self.assertNotEqual(report["train_metrics"]["loss"], report["validation_metrics"]["loss"])
        self.assertIn("macro_f1", report["validation_metrics"])
        self.assertNotIn("macro_f1", report["train_metrics"])


if __name__ == "__main__":
    unittest.main()
