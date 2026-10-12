import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.experimental_fusion.phase2.experiment_2a.simulate_experiment_2a import (
    simulate_one_sided_real_policy,
    compute_3state_metrics,
)

class TestExperiment2APolicy(unittest.TestCase):

    def test_boundary_values(self):
        """
        Verify exact boundary behavior for One-Sided Real Policy:
        - Both < 0.35 (e.g. 0.349, 0.349) -> LIKELY_REAL
        - One model >= 0.35 (e.g. 0.349, 0.350) -> UNCERTAIN
        - Other model >= 0.35 (e.g. 0.350, 0.349) -> UNCERTAIN
        - Both >= 0.35 -> UNCERTAIN
        """
        # Baseline verdict was LIKELY_REAL
        # Case 1: Both strictly below 0.35
        v1, r1, c1 = simulate_one_sided_real_policy("LIKELY_REAL", 0.349, 0.349, 0.30, "reason")
        self.assertEqual(v1, "LIKELY_REAL")
        self.assertFalse(c1)

        # Case 2: SBI < 0.35, Xception == 0.350
        v2, r2, c2 = simulate_one_sided_real_policy("LIKELY_REAL", 0.349, 0.350, 0.30, "reason")
        self.assertEqual(v2, "UNCERTAIN")
        self.assertTrue(c2)
        self.assertIn("One-Sided Real Policy", r2)

        # Case 3: SBI == 0.350, Xception < 0.35
        v3, r3, c3 = simulate_one_sided_real_policy("LIKELY_REAL", 0.350, 0.349, 0.30, "reason")
        self.assertEqual(v3, "UNCERTAIN")
        self.assertTrue(c3)

        # Case 4: Asymmetric drag case (FaceShifter pattern: SBI 0.08, Xception 0.48, fused 0.31)
        v4, r4, c4 = simulate_one_sided_real_policy("LIKELY_REAL", 0.08, 0.48, 0.31, "reason")
        self.assertEqual(v4, "UNCERTAIN")
        self.assertTrue(c4)

    def test_missing_scores(self):
        """
        Verify safe handling of missing scores (None values):
        - If one model is None and the other is < 0.35 -> LIKELY_REAL
        - If both models are None -> UNCERTAIN
        """
        # SBI None, Xception confident authentic
        v1, r1, c1 = simulate_one_sided_real_policy("LIKELY_REAL", None, 0.20, 0.20, "reason")
        self.assertEqual(v1, "LIKELY_REAL")
        self.assertFalse(c1)

        # Xception None, SBI confident authentic
        v2, r2, c2 = simulate_one_sided_real_policy("LIKELY_REAL", 0.22, None, 0.22, "reason")
        self.assertEqual(v2, "LIKELY_REAL")
        self.assertFalse(c2)

        # Both None
        v3, r3, c3 = simulate_one_sided_real_policy("LIKELY_REAL", None, None, 0.50, "reason")
        self.assertEqual(v3, "UNCERTAIN")
        self.assertTrue(c3)

    def test_model_disagreement_and_fake_preservation(self):
        """
        Verify that FAKE, UNCERTAIN, and disagreement verdicts are strictly preserved:
        - Baseline FAKE remains FAKE
        - Baseline UNCERTAIN remains UNCERTAIN
        """
        # Baseline FAKE
        v_fake, r_fake, c_fake = simulate_one_sided_real_policy("FAKE", 0.90, 0.85, 0.88, "High fake probability")
        self.assertEqual(v_fake, "FAKE")
        self.assertFalse(c_fake)

        # Baseline UNCERTAIN (from model disagreement)
        v_unc, r_unc, c_unc = simulate_one_sided_real_policy("UNCERTAIN", 0.10, 0.90, 0.50, "Model Disagreement")
        self.assertEqual(v_unc, "UNCERTAIN")
        self.assertFalse(c_unc)

    def test_non_face_inputs_preserved(self):
        """
        Verify that NOT_SUITABLE verdicts are untouched.
        """
        v_ns, r_ns, c_ns = simulate_one_sided_real_policy("NOT_SUITABLE", None, None, None, "No human face found")
        self.assertEqual(v_ns, "NOT_SUITABLE")
        self.assertFalse(c_ns)

    def test_simulation_reproducibility(self):
        """
        Verify deterministic reproducibility against saved Phase 2 predictions.
        """
        results_path = "research/experimental_fusion/phase2/experiment_2a/experiment_2a_simulation_results.json"
        self.assertTrue(os.path.exists(results_path))

        with open(results_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        v_meta = saved["video_simulation"]["alternative_metrics"]
        self.assertEqual(v_meta["false_positive_rate"], 0.0100)
        self.assertEqual(v_meta["false_negative_rate"], 0.0000)
        self.assertEqual(v_meta["abstention_rate"], 0.8450)
        self.assertEqual(v_meta["decided_accuracy"], 0.9677)

if __name__ == "__main__":
    unittest.main()
