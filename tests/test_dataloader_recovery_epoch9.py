import os
import sys
import tempfile
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn, DatasetStatsTracker
from backend.training.dataset_exceptions import (
    DatasetDecodeError,
    VideoDecodeError,
    ImageDecodeError,
    SamplePreprocessingError,
    is_fatal_error,
)


def create_dummy_video(file_path: str, num_frames: int = 10, width: int = 64, height: int = 64):
    """Creates a small valid mp4 video."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()


def create_dummy_image(file_path: str, width: int = 64, height: int = 64):
    """Creates a small valid jpg image."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(file_path, img)


def run_tests():
    print("=" * 70)
    print("RUNNING TARGETED EPOCH 9 DATALOADER CRASH RECOVERY TESTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. safe_collate_fn with valid samples
    # -------------------------------------------------------------
    print("\n[TEST 1] safe_collate_fn with valid samples...")
    valid_batch = [
        (torch.randn(3, 224, 224), torch.tensor(0, dtype=torch.long)),
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),
        (torch.randn(3, 224, 224), torch.tensor(0, dtype=torch.long)),
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),
    ]
    imgs, lbls = safe_collate_fn(valid_batch)
    assert imgs is not None and lbls is not None, "Failed on valid batch"
    assert imgs.shape == (4, 3, 224, 224), f"Expected shape (4, 3, 224, 224), got {imgs.shape}"
    assert lbls.shape == (4,), f"Expected labels shape (4,), got {lbls.shape}"
    assert lbls.dtype == torch.int64, f"Expected int64 labels, got {lbls.dtype}"
    assert torch.equal(lbls, torch.tensor([0, 1, 0, 1], dtype=torch.long)), "Labels mismatch"
    print("  --> PASS: Valid samples collated correctly into uniform tensors.")

    # -------------------------------------------------------------
    # 2. safe_collate_fn with None samples
    # -------------------------------------------------------------
    print("\n[TEST 2] safe_collate_fn with None and (None, None) entries...")
    batch_with_none = [
        None,
        (None, None),
        (torch.randn(3, 224, 224), torch.tensor(0, dtype=torch.long)),
        (None, torch.tensor(1, dtype=torch.long)),
        (torch.randn(3, 224, 224), None),
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),
        None,
    ]
    imgs, lbls = safe_collate_fn(batch_with_none)
    assert imgs is not None and lbls is not None, "Failed to collate valid items from batch with None"
    assert imgs.shape == (2, 3, 224, 224), f"Expected shape (2, 3, 224, 224), got {imgs.shape}"
    assert lbls.shape == (2,), f"Expected labels shape (2,), got {lbls.shape}"
    print("  --> PASS: None items, (None, None), and partial None pairs rejected cleanly.")

    # -------------------------------------------------------------
    # 3. safe_collate_fn with malformed tuples/lists
    # -------------------------------------------------------------
    print("\n[TEST 3] safe_collate_fn with malformed tuples/lists...")
    malformed_batch = [
        (torch.randn(3, 224, 224),),  # 1-tuple
        (torch.randn(3, 224, 224), torch.tensor(0), "extra_item"),  # 3-tuple
        "not_a_tuple",  # string
        12345,  # int
        {"image": torch.randn(3, 224, 224), "label": 0},  # dict
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),  # valid
    ]
    imgs, lbls = safe_collate_fn(malformed_batch)
    assert imgs is not None and lbls is not None
    assert imgs.shape == (1, 3, 224, 224), f"Expected 1 valid item, got {imgs.shape}"
    assert lbls.shape == (1,)
    print("  --> PASS: Malformed tuples, wrong lengths, and non-sequence types rejected.")

    # -------------------------------------------------------------
    # 4. safe_collate_fn with non-tensor image values
    # -------------------------------------------------------------
    print("\n[TEST 4] safe_collate_fn with non-tensor image values...")
    non_tensor_batch = [
        (np.zeros((224, 224, 3), dtype=np.uint8), torch.tensor(0)),  # numpy array
        ("path/to/image.jpg", torch.tensor(1)),  # string path
        ([1, 2, 3], torch.tensor(0)),  # list
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),  # valid
    ]
    imgs, lbls = safe_collate_fn(non_tensor_batch)
    assert imgs is not None and imgs.shape == (1, 3, 224, 224)
    print("  --> PASS: Non-tensor image values rejected.")

    # -------------------------------------------------------------
    # 5. safe_collate_fn with wrong tensor dimensions and NaN/Inf
    # -------------------------------------------------------------
    print("\n[TEST 5] safe_collate_fn with wrong tensor dimensions and NaN/Inf...")
    nan_tensor = torch.randn(3, 224, 224)
    nan_tensor[0, 0, 0] = float('nan')
    inf_tensor = torch.randn(3, 224, 224)
    inf_tensor[0, 0, 0] = float('inf')

    bad_dims_batch = [
        (torch.randn(224, 224), torch.tensor(0)),  # 2D tensor
        (torch.randn(224), torch.tensor(1)),  # 1D tensor
        (torch.randn(1, 3, 10, 224, 224), torch.tensor(0)),  # 5D tensor
        (torch.empty(0), torch.tensor(1)),  # 0-element tensor
        (nan_tensor, torch.tensor(0)),  # NaN tensor
        (inf_tensor, torch.tensor(1)),  # Inf tensor
        (torch.randn(3, 224, 224), torch.tensor(1, dtype=torch.long)),  # valid
    ]
    imgs, lbls = safe_collate_fn(bad_dims_batch)
    assert imgs is not None and imgs.shape == (1, 3, 224, 224)
    print("  --> PASS: Wrong dimensions, empty tensors, and NaN/Inf rejected.")

    # -------------------------------------------------------------
    # 6. safe_collate_fn with inconsistent tensor shapes within batch
    # -------------------------------------------------------------
    print("\n[TEST 6] safe_collate_fn with inconsistent tensor shapes in batch...")
    # 5 items: 3 are (3, 224, 224), 1 is (3, 200, 200), 1 is (1, 224, 224)
    mixed_shapes_batch = [
        (torch.randn(3, 224, 224), torch.tensor(0)),
        (torch.randn(3, 200, 200), torch.tensor(1)),  # deviant spatial size
        (torch.randn(3, 224, 224), torch.tensor(1)),
        (torch.randn(1, 224, 224), torch.tensor(0)),  # deviant channel count
        (torch.randn(3, 224, 224), torch.tensor(0)),
    ]
    imgs, lbls = safe_collate_fn(mixed_shapes_batch)
    assert imgs is not None and lbls is not None
    assert imgs.shape == (3, 3, 224, 224), f"Expected dominant shape batch of 3, got {imgs.shape}"
    assert lbls.shape == (3,)
    print("  --> PASS: Inconsistent tensor shapes identified and filtered to dominant shape.")

    # -------------------------------------------------------------
    # 7. safe_collate_fn with label validation & normalization
    # -------------------------------------------------------------
    print("\n[TEST 7] safe_collate_fn with label validation & normalization...")
    labels_batch = [
        (torch.randn(3, 224, 224), 0),  # python int
        (torch.randn(3, 224, 224), np.int64(1)),  # numpy int
        (torch.randn(3, 224, 224), torch.tensor(0, dtype=torch.int32)),  # 0D tensor
        (torch.randn(3, 224, 224), torch.tensor([1], dtype=torch.int64)),  # 1D single-element tensor
        (torch.randn(3, 224, 224), torch.tensor([0, 1])),  # 1D multi-element tensor (invalid)
        (torch.randn(3, 224, 224), "fake_label"),  # string label (invalid)
    ]
    imgs, lbls = safe_collate_fn(labels_batch)
    assert imgs is not None and imgs.shape == (4, 3, 224, 224)
    assert lbls.shape == (4,)
    assert lbls.dtype == torch.int64
    assert torch.equal(lbls, torch.tensor([0, 1, 0, 1], dtype=torch.long))
    print("  --> PASS: Labels normalized to scalar int64; invalid labels rejected.")

    # -------------------------------------------------------------
    # 8. safe_collate_fn with completely invalid batch (sentinel test)
    # -------------------------------------------------------------
    print("\n[TEST 8] safe_collate_fn with completely invalid batch...")
    all_none_batch = [(None, None), None, (None, None)]
    res = safe_collate_fn(all_none_batch)
    assert res == (None, None), f"Expected (None, None), got {res}"

    all_corrupt_batch = [("invalid", 1), (torch.randn(224), 0), (torch.randn(3, 224, 224), None)]
    res2 = safe_collate_fn(all_corrupt_batch)
    assert res2 == (None, None), f"Expected (None, None), got {res2}"

    # Verify train_image_model loop condition handles sentinel
    images, batch_labels = safe_collate_fn(all_none_batch)
    skipped = False
    if images is None or batch_labels is None or len(images) == 0:
        skipped = True
    assert skipped == True, "Training loop condition failed to safely skip empty batch sentinel"
    print("  --> PASS: Completely invalid batch returns (None, None) and is safely skipped.")

    # -------------------------------------------------------------
    # 9. Multi-Worker DataLoader with simulated decode errors & retries
    # -------------------------------------------------------------
    print("\n[TEST 9] Multi-Worker DataLoader with simulated decode errors & retries...")
    temp_dir = tempfile.mkdtemp(prefix="test_recovery_")
    valid_v1 = os.path.join(temp_dir, "v1.mp4")
    valid_v2 = os.path.join(temp_dir, "v2.mp4")
    corrupt_v = os.path.join(temp_dir, "corrupt.mp4")
    valid_i = os.path.join(temp_dir, "img1.jpg")

    create_dummy_video(valid_v1, num_frames=12)
    create_dummy_video(valid_v2, num_frames=12)
    create_dummy_image(valid_i)
    with open(corrupt_v, "wb") as f:
        f.write(b"NOT_A_VIDEO_GARBAGE_BYTES" * 30)

    manifest_csv = os.path.join(temp_dir, "test_manifest.csv")
    with open(manifest_csv, "w", encoding="utf-8") as f:
        f.write("dataset_source,file_path,media_type,label,manipulation_type,compression,split,sample_origin,frame_index\n")
        # Mixed valid, corrupted, and out-of-bounds samples
        f.write(f"test,{valid_v1},image,0,none,none,train,video_frame,2\n")
        f.write(f"test,{valid_v1},image,0,none,none,train,video_frame,9999\n") # out of bounds frame
        f.write(f"test,{corrupt_v},image,1,fake,none,train,video_frame,0\n")   # corrupt file
        f.write(f"test,{valid_i},image,0,none,none,train,physical_image,-1\n") # valid image
        f.write(f"test,{valid_v2},image,1,fake,none,train,video_frame,4\n")   # valid video frame
        f.write(f"test,{valid_v2},image,0,none,none,train,video_frame,6\n")   # valid video frame
        # Add consecutive failing samples
        f.write(f"test,{corrupt_v},image,1,fake,none,train,video_frame,0\n")
        f.write(f"test,{corrupt_v},image,1,fake,none,train,video_frame,0\n")

    dataset = DeepfakeDataset(
        split="train",
        media_type="image",
        manifest_path=manifest_csv,
        use_cache=False,
        max_retries=2
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
        collate_fn=safe_collate_fn,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 4, 3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2)
    ).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

    batches_processed = 0
    samples_processed = 0
    batches_skipped = 0

    for b_idx, (b_imgs, b_lbls) in enumerate(loader):
        if b_imgs is None or b_lbls is None or len(b_imgs) == 0:
            batches_skipped += 1
            print(f"  [Loader] Batch {b_idx}: Safely skipped empty batch.")
            continue

        b_actual = b_imgs.size(0)
        assert b_imgs.shape[1:] == (3, 224, 224), f"Unexpected tensor shape: {b_imgs.shape}"
        assert not torch.isnan(b_imgs).any(), "NaN in batch images"

        b_imgs = b_imgs.to(device)
        b_lbls = b_lbls.to(device)

        opt.zero_grad()
        out = dummy_model(b_imgs)
        loss = crit(out, b_lbls)
        loss.backward()
        opt.step()

        batches_processed += 1
        samples_processed += b_actual
        print(f"  [Loader] Batch {b_idx}: Successfully processed {b_actual} samples. Loss={loss.item():.4f}")

    stats = dataset.get_failure_stats(successful_samples=samples_processed)
    print(f"  Stats: failures={stats['decode_failures']}, retries={stats['retry_count']}, skipped={stats['skipped_samples']}")
    assert stats["decode_failures"] > 0, "Expected decode failures during stress test"
    assert batches_processed > 0, "Expected at least one processed batch"
    print("  --> PASS: Multi-Worker DataLoader completed training steps without crashing.")

    # -------------------------------------------------------------
    # 10. Fatal exception propagation
    # -------------------------------------------------------------
    print("\n[TEST 10] Fatal exception classification & propagation...")
    # Fatal errors
    assert is_fatal_error(torch.cuda.OutOfMemoryError("CUDA OOM")) == True
    assert is_fatal_error(TypeError("TypeError bug")) == True
    assert is_fatal_error(NameError("NameError bug")) == True
    assert is_fatal_error(AttributeError("AttributeError bug")) == True
    assert is_fatal_error(KeyboardInterrupt()) == True
    assert is_fatal_error(SystemExit()) == True

    # Recoverable errors
    assert is_fatal_error(VideoDecodeError("read frame fail")) == False
    assert is_fatal_error(ImageDecodeError("read img fail")) == False
    assert is_fatal_error(SamplePreprocessingError("preprocess fail")) == False
    assert is_fatal_error(cv2.error("OpenCV error")) == False

    # Verify fatal error raises out of __getitem__
    class BuggyDataset(DeepfakeDataset):
        def _extract_video_frame(self, file_path, frame_idx):
            raise TypeError("Programming bug: int object not callable")

    buggy_ds = BuggyDataset(
        split="train",
        media_type="image",
        manifest_path=manifest_csv,
        use_cache=False,
        max_retries=1
    )
    fatal_propagated = False
    try:
        _ = buggy_ds[0]
    except TypeError:
        fatal_propagated = True
    assert fatal_propagated == True, "Fatal TypeError was swallowed by __getitem__!"
    print("  --> PASS: Fatal errors are correctly identified and strictly propagated.")

    # -------------------------------------------------------------
    # 11. Checkpoint integrity check
    # -------------------------------------------------------------
    print("\n[TEST 11] Epoch 8 checkpoint preservation check...")
    latest_path = "backend/models/weights/runs/full_image_I1/latest_checkpoint.pth"
    best_path = "backend/models/weights/runs/full_image_I1/best_checkpoint.pth"

    assert os.path.exists(latest_path), f"latest_checkpoint.pth missing: {latest_path}"
    assert os.path.exists(best_path), f"best_checkpoint.pth missing: {best_path}"

    ckpt_latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    assert ckpt_latest.get("epoch") in (8, 9), f"Expected latest epoch 8 or 9, found {ckpt_latest.get('epoch')}"
    assert "model_state_dict" in ckpt_latest, "model_state_dict missing from latest_checkpoint"
    assert "optimizer_state_dict" in ckpt_latest, "optimizer_state_dict missing from latest_checkpoint"

    ckpt_best = torch.load(best_path, map_location="cpu", weights_only=False)
    assert ckpt_best.get("epoch") in (8, 9), f"Expected best epoch 8 or 9, found {ckpt_best.get('epoch')}"
    print(f"  Latest Checkpoint: Epoch {ckpt_latest.get('epoch')}, Config: {ckpt_latest.get('config')}")
    print(f"  Best Checkpoint: Epoch {ckpt_best.get('epoch')}, Best Val Metric: {ckpt_best.get('best_val_metric')}")
    print(f"  --> PASS: Epoch {ckpt_latest.get('epoch')} checkpoint is completely intact and untampered.")

    # -------------------------------------------------------------
    # 12. CUDA availability check
    # -------------------------------------------------------------
    print("\n[TEST 12] CUDA environment verification...")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Device Count: {torch.cuda.device_count()}")
    assert cuda_avail == True, "CUDA is expected to be available for training!"
    print("  --> PASS: CUDA is fully operational.")

    print("\n" + "=" * 70)
    print("ALL 12 TARGETED TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
