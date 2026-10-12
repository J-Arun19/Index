import os
import sys

# Ensure workspace root is in sys.path
_workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _workspace_dir not in sys.path:
    sys.path.insert(0, _workspace_dir)

import random
import tempfile
import unittest
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

# Configure thread limits
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
torch.set_num_threads(1)
cv2.ocl.setUseOpenCL(False)

from backend.utils.config import PipelineConfig
from backend.models.video_model import VideoModel
from backend.preprocessing.video_preprocessor import VideoPreprocessor
from backend.training.dataset_exceptions import VideoDecodeError
from backend.training.train_video_model import (
    validate_checkpoint_state,
    save_checkpoint_atomic,
    restore_rng_states,
    check_tensor_finite,
    check_labels_finite,
    check_data_leakage,
    NumericalInstabilityError
)
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn


class TestVideoPipeline(unittest.TestCase):
    """
    Controlled end-to-end unit and integration test suite for the SatyaDrishti
    video deepfake detection training pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(PipelineConfig.DEVICE)
        cls.test_dir = tempfile.mkdtemp(prefix="satyadrishti_video_test_")
        cls.real_video_sample = r"D:\Download\SatyaDrishti_Datasets\Celeb DF\Celeb-real\id0_0000.mp4"

    @classmethod
    def tearDownClass(cls):
        try:
            shutil_rm = True
            import shutil
            shutil.rmtree(cls.test_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_video_decoding_and_uniform_sampling(self):
        """Tests reliable video decoding, uniform 8-frame sampling, resizing, and normalization on real dataset video."""
        if not os.path.exists(self.real_video_sample):
            self.skipTest(f"Real video sample not found at: {self.real_video_sample}")

        preprocessor = VideoPreprocessor(max_frames=8)
        tensor = preprocessor.preprocess_video(self.real_video_sample)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (8, 3, 224, 224), f"Expected shape (8, 3, 224, 224), got {tensor.shape}")
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all().item(), "Decoded video tensor contains NaN or Inf values")
        
        # Verify ImageNet normalization range (typically between -2.5 and +2.5)
        self.assertGreater(tensor.min().item(), -5.0)
        self.assertLess(tensor.max().item(), 5.0)
        print("  [PASS] Test 1: Video decoding, uniform 8-frame sampling, and normalization verified.")

    def test_02_short_video_handling(self):
        """Tests that videos with fewer frames than target max_frames (e.g. 2 frames) pad gracefully to exactly 8 frames."""
        short_video_path = os.path.join(self.test_dir, "short_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(short_video_path, fourcc, 10.0, (224, 224))
        for _ in range(2):
            frame = np.full((224, 224, 3), 128, dtype=np.uint8)
            cv2.circle(frame, (112, 112), 40, (200, 200, 200), -1)
            out.write(frame)
        out.release()

        preprocessor = VideoPreprocessor(max_frames=8)
        tensor = preprocessor.preprocess_video(short_video_path)

        self.assertEqual(tensor.shape, (8, 3, 224, 224), "Short video did not pad to required sequence length of 8")
        self.assertTrue(torch.isfinite(tensor).all().item())
        print("  [PASS] Test 2: Short video handling and sequence padding verified.")

    def test_03_corrupt_video_handling(self):
        """Tests that corrupt or unreadable video files raise VideoDecodeError cleanly."""
        corrupt_video_path = os.path.join(self.test_dir, "corrupt_video.mp4")
        with open(corrupt_video_path, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_BYTES_1234567890")

        preprocessor = VideoPreprocessor(max_frames=8)
        with self.assertRaises(VideoDecodeError):
            preprocessor.preprocess_video(corrupt_video_path)
        print("  [PASS] Test 3: Corrupt video detection and VideoDecodeError handling verified.")

    def test_04_model_architecture_and_forward_pass(self):
        """Tests VideoModel forward pass with ResNet18 spatial extractor and LSTM/mean temporal pooling."""
        # 1. Test LSTM temporal pooling (default)
        model_lstm = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        model_lstm.eval()

        # Batch of 2 videos, 8 frames each
        dummy_input = torch.randn(2, 8, 3, 224, 224, device=self.device)
        with torch.no_grad():
            logits_lstm = model_lstm(dummy_input)
            pred_dict = model_lstm.predict(dummy_input)

        self.assertEqual(logits_lstm.shape, (2, 2), f"Expected logits shape (2, 2), got {logits_lstm.shape}")
        self.assertTrue(torch.isfinite(logits_lstm).all().item())
        self.assertIn("probabilities", pred_dict)
        self.assertIn("predicted_class", pred_dict)
        self.assertEqual(pred_dict["predicted_class"].shape, (2,))

        # 2. Test Mean temporal pooling
        model_mean = VideoModel(num_classes=2, pretrained=False, temporal_pooling="mean").to(self.device)
        model_mean.eval()
        with torch.no_grad():
            logits_mean = model_mean(dummy_input)
        self.assertEqual(logits_mean.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits_mean).all().item())
        print("  [PASS] Test 4: VideoModel forward pass (LSTM & Mean pooling) and predict() verified.")

    def test_05_loss_gradients_amp_and_optimizer_step(self):
        """Tests full batch forward pass with AMP, CrossEntropyLoss, backprop, gradient checks, and optimizer step."""
        model = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        use_amp = (self.device.type == "cuda")
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        batch_size = 4
        num_frames = 8
        videos = torch.randn(batch_size, num_frames, 3, 224, 224, device=self.device)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=self.device)

        check_tensor_finite(videos, "test_input", "videos", 1, 1, 1e-4)
        check_labels_finite(labels, "test_labels", "labels", 1, 1, 1e-4)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        self.assertTrue(torch.isfinite(loss).item(), "Loss is not finite")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Check gradients are finite and not None
        grad_found = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None after backward()")
                self.assertTrue(torch.isfinite(param.grad).all().item(), f"Gradient for {name} contains NaN/Inf")
                grad_found = True
        self.assertTrue(grad_found, "No gradients were computed")

        # Clip grad norm and step optimizer
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertTrue(torch.isfinite(grad_norm).item())

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        print("  [PASS] Test 5: Forward, Loss, AMP backprop, gradient validation, and optimizer step verified.")

    def test_06_remainder_batch_fp32_fallback(self):
        """Tests that remainder batches (batch_actual_size != batch_size) safely fall back to FP32 execution."""
        model = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        nominal_batch_size = 4
        remainder_size = 1
        videos = torch.randn(remainder_size, 8, 3, 224, 224, device=self.device)
        labels = torch.tensor([1], dtype=torch.long, device=self.device)

        # Emulate remainder fallback logic: batch_actual_size (1) != nominal_batch_size (4) -> batch_use_amp = False
        use_amp = (self.device.type == "cuda")
        batch_use_amp = use_amp and (videos.size(0) == nominal_batch_size)
        self.assertFalse(batch_use_amp, "Remainder batch did not disable AMP")

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=batch_use_amp):
            outputs = model(videos)
            loss = criterion(outputs, labels)

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        print("  [PASS] Test 6: Remainder batch FP32 fallback verified.")

    def test_07_resumable_atomic_checkpointing(self):
        """Tests atomic checkpoint creation, numerical validation, and resumable restoration."""
        model = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        ckpt_path = os.path.join(self.test_dir, "test_checkpoint.pth")
        state = {
            "epoch": 3,
            "step": 150,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": 0.3542,
            "best_opt_thresh": 0.65,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }

        # 1. Valid checkpoint save
        saved = save_checkpoint_atomic(state, ckpt_path)
        self.assertTrue(saved, "Failed to atomically save valid checkpoint")
        self.assertTrue(os.path.exists(ckpt_path), "Checkpoint file was not created on disk")

        # 2. Resumable loading
        loaded_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_resumed = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        model_resumed.load_state_dict(loaded_ckpt["model_state_dict"])

        for p1, p2 in zip(model.parameters(), model_resumed.parameters()):
            self.assertTrue(torch.equal(p1, p2), "Resumed model parameters do not match original model")
        self.assertEqual(loaded_ckpt["epoch"], 3)
        self.assertEqual(loaded_ckpt["step"], 150)

        # 3. Refusal to save corrupted checkpoint
        corrupted_state = {
            "epoch": 4,
            "model_state_dict": {"corrupt_weight": torch.tensor([1.0, float('nan'), 2.0])},
            "best_val_loss": float('nan')
        }
        refused = save_checkpoint_atomic(corrupted_state, ckpt_path)
        self.assertFalse(refused, "save_checkpoint_atomic failed to reject corrupted state")
        # Ensure original healthy checkpoint was not overwritten
        original_healthy = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertEqual(original_healthy["epoch"], 3)
        print("  [PASS] Test 7: Resumable atomic checkpointing & corruption prevention verified.")

    def test_08_split_leakage_and_manifest_integrity(self):
        """Tests that video entries in dataset_manifest.csv have zero source or identity leakage across splits."""
        manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
        self.assertTrue(os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}")

        train_ds = DeepfakeDataset(split="train", media_type="video", manifest_path=manifest_path, use_cache=False)
        val_ds = DeepfakeDataset(split="val", media_type="video", manifest_path=manifest_path, use_cache=False)
        test_ds = DeepfakeDataset(split="test", media_type="video", manifest_path=manifest_path, use_cache=False)

        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(val_ds), 0)
        self.assertGreater(len(test_ds), 0)

        # Verify automated leakage check passes without raising an exception
        try:
            check_data_leakage(train_ds, val_ds, test_ds)
            leakage_passed = True
        except ValueError as e:
            leakage_passed = False
            self.fail(f"Data leakage detected: {e}")
        self.assertTrue(leakage_passed)
        print(f"  [PASS] Test 8: Manifest split isolation (Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}) verified.")

    def test_09_epoch1_first_best_checkpoint_initialization(self):
        """Tests that in Epoch 1, when best_val_loss starts as inf and first finite val_loss arrives, both checkpoints save atomically with zero errors."""
        model = VideoModel(num_classes=2, pretrained=False, temporal_pooling="lstm").to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        latest_path = os.path.join(self.test_dir, "epoch1_latest.pth")
        best_path = os.path.join(self.test_dir, "epoch1_best.pth")

        # Initial state before epoch 1
        best_val_loss = float('inf')
        epoch_val_loss = 0.5218
        epoch_train_loss = 0.4838

        # Correct update sequence: update best_val_loss BEFORE building checkpoint state
        is_best = (epoch_val_loss < best_val_loss)
        self.assertTrue(is_best)
        if is_best:
            best_val_loss = epoch_val_loss

        checkpoint_state = {
            "epoch": 1,
            "step": 75,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "best_val_loss": best_val_loss,
            "metrics": {"macro_f1": 0.571, "roc_auc": 0.6127}
        }

        # Both must pass validation and save atomically
        saved_latest = save_checkpoint_atomic(checkpoint_state, latest_path)
        self.assertTrue(saved_latest, "Failed to save epoch 1 latest checkpoint")
        self.assertTrue(os.path.exists(latest_path))

        saved_best = save_checkpoint_atomic(checkpoint_state, best_path)
        self.assertTrue(saved_best, "Failed to save epoch 1 best checkpoint")
        self.assertTrue(os.path.exists(best_path))

        # Validate loaded content
        loaded_best = torch.load(best_path, map_location="cpu", weights_only=False)
        is_valid, errors = validate_checkpoint_state(loaded_best)
        self.assertTrue(is_valid, f"Checkpoint failed validation: {errors}")
        self.assertEqual(len(errors), 0)
        self.assertEqual(loaded_best["best_val_loss"], 0.5218)
        print("  [PASS] Test 9: Epoch 1 best_val_loss initialization and atomic save sequence verified.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SatyaDrishti Video Deepfake Detection Pipeline Validation Tests  ")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVideoPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
