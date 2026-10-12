import os
import sys
import json
import hashlib
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class TestCalibrationAndExpansion(unittest.TestCase):

    def test_ff_splits_strictly_disjoint(self):
        """Verify 0 sequence overlap between FF++ train, val, and test splits."""
        with open("backend/training/splits/train.json") as f:
            train_pairs = json.load(f)
        with open("backend/training/splits/val.json") as f:
            val_pairs = json.load(f)
        with open("backend/training/splits/test.json") as f:
            test_pairs = json.load(f)

        train_seqs = {str(s).zfill(3) for pair in train_pairs for s in pair}
        val_seqs = {str(s).zfill(3) for pair in val_pairs for s in pair}
        test_seqs = {str(s).zfill(3) for pair in test_pairs for s in pair}

        self.assertEqual(len(train_seqs.intersection(val_seqs)), 0, "Train-Val leak detected!")
        self.assertEqual(len(train_seqs.intersection(test_seqs)), 0, "Train-Test leak detected!")
        self.assertEqual(len(val_seqs.intersection(test_seqs)), 0, "Val-Test leak detected!")

    def test_manifests_exist_and_disjoint(self):
        """Verify validation and locked test manifests exist and share 0 duplicate files."""
        val_path = "research/experimental_fusion/datasets/validation_manifest_v2.json"
        test_path = "research/experimental_fusion/datasets/locked_test_manifest_v2.json"

        self.assertTrue(os.path.exists(val_path))
        self.assertTrue(os.path.exists(test_path))

        with open(val_path) as f:
            val_data = json.load(f)
        with open(test_path) as f:
            test_data = json.load(f)

        val_vids = {v["filepath"] for v in val_data["videos"]}
        test_vids = {v["filepath"] for v in test_data["videos"]}
        val_hashes = {v["sha256"] for v in val_data["videos"]}
        test_hashes = {v["sha256"] for v in test_data["videos"]}

        # Zero path overlap
        self.assertEqual(len(val_vids.intersection(test_vids)), 0, "Video path overlap between val and test!")
        # Zero hash overlap
        self.assertEqual(len(val_hashes.intersection(test_hashes)), 0, "SHA-256 hash overlap between val and test!")

        # Verify counts
        self.assertEqual(len(val_data["videos"]), 60)
        self.assertEqual(len(test_data["videos"]), 200)
        self.assertEqual(len(test_data["images"]), 207)

    def test_fusion_calibration_config_valid(self):
        """Verify isolated calibration config contains fitted parameters from validation split."""
        cfg_path = "backend/models/experimental/fusion_calibration_config.json"
        self.assertTrue(os.path.exists(cfg_path))

        with open(cfg_path) as f:
            cfg = json.load(f)

        self.assertEqual(cfg["fitted_on"], "validation_split_only")
        self.assertGreater(cfg["sbi"]["temperature"], 0.5)
        self.assertGreater(cfg["xception"]["temperature"], 0.5)
        self.assertEqual(cfg["decision_thresholds"]["tau_real"], 0.35)
        self.assertEqual(cfg["decision_thresholds"]["tau_fake"], 0.65)
        self.assertEqual(cfg["decision_thresholds"]["min_confidence"], 60.0)

    def test_phase1_legacy_manifests_unmodified(self):
        """Verify recorded regression baseline hashes for Phase 1 artifacts."""
        expected_hashes = {
            "scratch/image_eval_manifest.json": "48413326aa5a185ac1096594a47353ec70da96899ac6109a89049c8941b8f58a",
            "research/experimental_fusion/fusion_benchmark_results.json": "f8dd080435801edbea81cfa7c2b7c0a0a7a3ff6a77dee8366c924a1da972e5d7",
        }
        for path, exp_h in expected_hashes.items():
            self.assertTrue(os.path.exists(path))
            act_h = sha256_file(path)
            self.assertEqual(act_h, exp_h, f"Phase 1 regression artifact {path} was mutated!")

if __name__ == "__main__":
    unittest.main()
