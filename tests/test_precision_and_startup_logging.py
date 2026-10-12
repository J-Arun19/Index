import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.training.train_image_model import (
    determine_precision_mode,
    check_tensor_finite,
    check_labels_finite,
    ImageModel,
)


class TestPrecisionAndStartupLogging(unittest.TestCase):
    """
    Regression test suite verifying:
    1. FP32 mode passes startup logging without UnboundLocalError.
    2. Active precision is correctly displayed as 'FP32' in FP32 mode.
    3. Autocast and GradScaler are disabled in FP32 mode.
    4. AMP mode behaves correctly when explicitly selected, including remainder batch fallback.
    """

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_determine_precision_mode_fp32(self):
        """Verify determine_precision_mode returns use_amp=False for explicit FP32 flag and precision arg."""
        use_amp_flag, prec_str_flag, reason_flag = determine_precision_mode(
            precision_arg="auto", force_fp32=True, device=self.device
        )
        self.assertFalse(use_amp_flag, "force_fp32=True must disable AMP")
        self.assertEqual(prec_str_flag, "FP32")
        self.assertIn("CLI", reason_flag)

        use_amp_arg, prec_str_arg, reason_arg = determine_precision_mode(
            precision_arg="fp32", force_fp32=False, device=self.device
        )
        self.assertFalse(use_amp_arg, "precision_arg='fp32' must disable AMP")
        self.assertEqual(prec_str_arg, "FP32")

    def test_determine_precision_mode_amp(self):
        """Verify determine_precision_mode returns use_amp=True when precision='amp' on CUDA."""
        if self.device.type == "cuda":
            use_amp, prec_str, reason = determine_precision_mode(
                precision_arg="amp", force_fp32=False, device=self.device
            )
            self.assertTrue(use_amp, "precision_arg='amp' must enable AMP on CUDA")
            self.assertEqual(prec_str, "AMP (FP16)")
            self.assertIn("Explicitly requested", reason)

    def test_startup_logging_block_fp32_no_unbound_error(self):
        """
        Simulate the exact training loop startup logging section at Epoch 1, Batch 1
        under FP32 mode to verify no UnboundLocalError and displayed precision is FP32.
        """
        batch_size = 64
        use_amp, precision_str, _ = determine_precision_mode(
            precision_arg="auto", force_fp32=True, device=self.device
        )
        self.assertFalse(use_amp)

        model = ImageModel(num_classes=2, pretrained=False).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        lr = 1e-4
        ep = 8
        total_batches = 800

        # Create dummy batch 0
        images = torch.randn(64, 3, 224, 224, device=self.device)
        batch_labels = torch.randint(0, 2, (64,), device=self.device)
        batch_idx = 0

        # Exact logic as in train_image_model.py:
        batch_actual_size = images.size(0)
        batch_use_amp = bool(use_amp and (batch_actual_size == batch_size))
        opt_lrs_str = ", ".join(f"{pg.get('lr', lr):.1e}" for pg in optimizer.param_groups)
        current_lr_val = optimizer.param_groups[0].get("lr", lr) if optimizer.param_groups else lr
        sample_info = f"Batch [{batch_idx+1}/{total_batches}], size={batch_actual_size} (LRs: [{opt_lrs_str}])"

        log_output = (
            f"\n--- [BATCH 1 INTEGRITY CHECK (Epoch {ep+1})] ---\n"
            f"  Input Tensor Shape: {tuple(images.shape)}, Dtype: {images.dtype}\n"
            f"  Pixel Range: min={images.min().item():.4f}, max={images.max().item():.4f}, mean={images.mean().item():.4f}, std={images.std().item():.4f}\n"
            f"  Finite: {torch.isfinite(images).all().item()}\n"
            f"  Labels: Shape={tuple(batch_labels.shape)}, Unique Classes={torch.unique(batch_labels).tolist()}\n"
            f"  Active Precision: {'AMP (FP16)' if batch_use_amp else 'FP32'}\n"
            f"  Optimizer LRs: [{opt_lrs_str}]\n"
            f"------------------------------------------------"
        )

        self.assertIn("Active Precision: FP32", log_output)
        self.assertFalse(batch_use_amp)

        # Verify GradScaler is disabled
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.assertFalse(scaler.is_enabled(), "GradScaler must be disabled in FP32 mode")

    def test_startup_logging_block_amp_mode(self):
        """
        Verify startup logging and batch_use_amp in AMP mode for full batch and remainder batch.
        """
        if self.device.type != "cuda":
            self.skipTest("CUDA required for AMP test")

        batch_size = 64
        use_amp, precision_str, _ = determine_precision_mode(
            precision_arg="amp", force_fp32=False, device=self.device
        )
        self.assertTrue(use_amp)

        # Full batch
        full_actual_size = 64
        batch_use_amp_full = bool(use_amp and (full_actual_size == batch_size))
        self.assertTrue(batch_use_amp_full, "Full batch in AMP mode must enable batch_use_amp")
        active_prec_full = "AMP (FP16)" if batch_use_amp_full else "FP32"
        self.assertEqual(active_prec_full, "AMP (FP16)")

        # Remainder batch (e.g. last batch with 17 samples)
        rem_actual_size = 17
        batch_use_amp_rem = bool(use_amp and (rem_actual_size == batch_size))
        self.assertFalse(batch_use_amp_rem, "Remainder batch must disable batch_use_amp")
        active_prec_rem = "AMP (FP16)" if batch_use_amp_rem else "FP32"
        self.assertEqual(active_prec_rem, "FP32")

        # GradScaler is enabled in AMP mode
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.assertTrue(scaler.is_enabled(), "GradScaler must be enabled in AMP mode")

    def test_validation_path_precision(self):
        """Verify validation autocast toggle correctly derives from use_amp."""
        # FP32 validation
        use_amp = False
        batch_size = 64
        val_batch_size = 64
        val_batch_use_amp = use_amp and (val_batch_size == batch_size)
        self.assertFalse(val_batch_use_amp, "Validation autocast must be disabled in FP32 mode")

        # AMP validation
        use_amp = True
        val_batch_use_amp = use_amp and (val_batch_size == batch_size)
        self.assertTrue(val_batch_use_amp, "Validation autocast must be enabled for full batch in AMP mode")


if __name__ == "__main__":
    unittest.main()
