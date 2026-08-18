import os
import unittest
import numpy as np
import torch
import torch.nn as nn
from backend.training.video_representation_diagnostics import DiagnosticsPipeline

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(224*224*3, 512),
            nn.Identity()
        )
        # Custom identity replacement to simulate outputting feature vectors
        class IdentityFC(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x
        self.backbone = IdentityFC()
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(256, 2)
        
    def forward(self, x):
        b, s, c, h, w = x.size()
        # Mock spatial features: project flattened frames to 512-dim
        flat_x = x.view(b * s, -1)
        # Linear projection to 512-dim
        proj = torch.zeros((b * s, 512), device=x.device)
        proj[:, :min(flat_x.shape[1], 512)] = flat_x[:, :512]
        features = proj.view(b, s, 512)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])

class TestRepresentationDiagnostics(unittest.TestCase):
    def setUp(self):
        self.pipeline = DiagnosticsPipeline(os.getcwd())
        
    def test_centroid_calculations(self):
        """Test centroid and variance calculation helper on known dummy arrays."""
        features = np.array([
            [1.0, 2.0],
            [1.5, 2.5],
            [10.0, 20.0],
            [10.5, 20.5]
        ])
        labels = np.array([0, 0, 1, 1])
        
        res = self.pipeline.evaluate_separability(features, labels)
        
        # Real Centroid: [1.25, 2.25]
        # Fake Centroid: [10.25, 20.25]
        # Centroid Distance = sqrt(9^2 + 18^2) = sqrt(81 + 324) = sqrt(405) ≈ 20.1246
        self.assertAlmostEqual(res["euclidean_centroid_distance"], 20.1246117975, places=4)
        
        # Within-class variance
        # Real Variance: var([1, 2]) + var([1.5, 2.5]) ?
        # Real coords: [[1, 2], [1.5, 2.5]]. Centroid = [1.25, 2.25]
        # Real var along axes:
        # axis 0: var([1.0, 1.5]) = 0.25^2 = 0.0625
        # axis 1: var([2.0, 2.5]) = 0.25^2 = 0.0625
        # Mean real var = 0.0625
        # Fake var along axes:
        # axis 0: var([10.0, 10.5]) = 0.0625
        # axis 1: var([20.0, 20.5]) = 0.0625
        # Mean fake var = 0.0625
        # Within-class var = 0.0625
        self.assertAlmostEqual(res["within_class_variance"], 0.0625, places=4)
        
    def test_centroid_edge_cases(self):
        """Test separability calculations under missing classes or zero variance."""
        features = np.array([[1.0, 1.0], [2.0, 2.0]])
        labels_empty = np.array([0, 0])
        
        res = self.pipeline.evaluate_separability(features, labels_empty)
        self.assertEqual(res["euclidean_centroid_distance"], 0.0)
        self.assertEqual(res["cosine_centroid_distance"], 0.0)
        self.assertEqual(res["within_class_variance"], 0.0)
        self.assertEqual(res["between_class_variance"], 0.0)

    def test_train_and_eval_probes(self):
        """Test linear probe training and validation metrics."""
        train_x = np.array([
            [1.0, 1.0], [1.1, 0.9], [1.2, 1.2],
            [10.0, 10.0], [9.9, 10.1], [10.2, 9.8]
        ])
        train_y = np.array([0, 0, 0, 1, 1, 1])
        
        val_x = np.array([
            [1.15, 1.05],
            [9.85, 10.15]
        ])
        val_y = np.array([0, 1])
        
        res = self.pipeline.train_and_eval_probes(train_x, train_y, val_x, val_y)
        
        # Test that all expected classifiers are returned
        for name in ["Logistic Regression", "Linear SVM", "Ridge", "Nearest Neighbors"]:
            self.assertIn(name, res)
            self.assertEqual(res[name]["accuracy"], 1.0)
            self.assertEqual(res[name]["macro_f1"], 1.0)
            self.assertEqual(res[name]["real_recall"], 1.0)
            self.assertEqual(res[name]["fake_recall"], 1.0)

    def test_train_val_separation_check(self):
        """Ensure no overlap exists between train and validation splits in the pipeline."""
        if os.path.exists(self.pipeline.pilot_manifest_path):
            train_samples, val_samples = self.pipeline.load_pilot_samples()
            train_paths = {s["file_path"] for s in train_samples}
            val_paths = {s["file_path"] for s in val_samples}
            
            # Check disjointness
            intersection = train_paths.intersection(val_paths)
            self.assertEqual(len(intersection), 0, f"Train and validation sets share paths: {intersection}")

if __name__ == "__main__":
    unittest.main()
