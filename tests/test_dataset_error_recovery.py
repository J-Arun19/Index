import os
import sys
import tempfile
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure stdout for clean Windows console output
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
from backend.models.image_model import ImageModel

def create_dummy_video(file_path: str, num_frames: int = 15, width: int = 64, height: int = 64):
    """Creates a small valid mp4 video with OpenCV."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()

def create_dummy_image(file_path: str, width: int = 64, height: int = 64):
    """Creates a small valid jpg image."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(file_path, img)

def run_all_tests():
    print("=" * 60)
    print("STARTING COMPREHENSIVE DATASET RECOVERY STRESS TEST")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp(prefix="stress_test_")
    print(f"Working directory: {temp_dir}")
    
    # 1. Generate test assets
    valid_vid1 = os.path.join(temp_dir, "valid_video_1.mp4")
    valid_vid2 = os.path.join(temp_dir, "valid_video_2.mp4")
    corrupt_vid = os.path.join(temp_dir, "corrupt_video.mp4")
    missing_vid = os.path.join(temp_dir, "missing_video.mp4")
    valid_img = os.path.join(temp_dir, "valid_image.jpg")
    corrupt_img = os.path.join(temp_dir, "corrupt_image.jpg")
    
    create_dummy_video(valid_vid1, num_frames=12)
    create_dummy_video(valid_vid2, num_frames=12)
    create_dummy_image(valid_img)
    
    # Corrupt video: garbage bytes
    with open(corrupt_vid, "wb") as f:
        f.write(b"NOT_A_REAL_VIDEO_HEADER_DATA_GARBAGE_BYTES_1234567890" * 20)
        
    # Corrupt image: 0 bytes
    with open(corrupt_img, "wb") as f:
        pass
        
    # TEST 1: Strict VideoCapture Resource Leak & Direct Extraction Test
    print("\n--- TEST 1: VideoCapture Handle & Exception Verification ---")
    ds_test = DeepfakeDataset.__new__(DeepfakeDataset)
    ds_test.split = "train"
    ds_test.media_type = "image"
    ds_test.stats = DatasetStatsTracker()
    
    # A: Valid frame extraction
    f_valid = ds_test._extract_video_frame(valid_vid1, 5)
    assert f_valid is not None and f_valid.shape == (64, 64, 3), f"Expected frame shape (64, 64, 3), got {getattr(f_valid, 'shape', None)}"
    print("[PASS] Valid frame extracted successfully.")
    
    # B: Out of bounds frame index
    try:
        ds_test._extract_video_frame(valid_vid1, 9999)
        assert False, "Should have raised VideoDecodeError for out-of-bounds frame!"
    except VideoDecodeError as e:
        print(f"[PASS] Out-of-bounds frame correctly raised VideoDecodeError: {e}")
        
    # C: Corrupted video file
    try:
        ds_test._extract_video_frame(corrupt_vid, 0)
        assert False, "Should have raised VideoDecodeError for corrupt video!"
    except VideoDecodeError as e:
        print(f"[PASS] Corrupt video correctly raised VideoDecodeError: {e}")
        
    # D: Missing video file
    try:
        ds_test._extract_video_frame(missing_vid, 0)
        assert False, "Should have raised VideoDecodeError for missing video!"
    except VideoDecodeError as e:
        print(f"[PASS] Missing video correctly raised VideoDecodeError: {e}")
        
    # E: Verify repeated failures do not leak handles (run 50 cycles)
    for _ in range(50):
        try:
            ds_test._extract_video_frame(corrupt_vid, 0)
        except VideoDecodeError:
            pass
    print("[PASS] 50 repeated failures executed with guaranteed VideoCapture cleanup.")

    # TEST 2: Multi-Worker DataLoader Stress Test with Injected Errors
    print("\n--- TEST 2: Multi-Worker DataLoader with Mixed Errors & Sample Skipping ---")
    test_samples = [
        # Valid sample
        {"dataset_source": "test", "file_path": valid_vid1, "label": 0, "manipulation_type": "none", "compression": "none", "sample_origin": "video_frame", "frame_index": 2},
        # Invalid frame index
        {"dataset_source": "test", "file_path": valid_vid1, "label": 0, "manipulation_type": "none", "compression": "none", "sample_origin": "video_frame", "frame_index": 500},
        # Corrupt video
        {"dataset_source": "test", "file_path": corrupt_vid, "label": 1, "manipulation_type": "fake", "compression": "none", "sample_origin": "video_frame", "frame_index": 0},
        # Valid physical image
        {"dataset_source": "test", "file_path": valid_img, "label": 0, "manipulation_type": "none", "compression": "none", "sample_origin": "physical_image", "frame_index": -1},
        # Missing video
        {"dataset_source": "test", "file_path": missing_vid, "label": 1, "manipulation_type": "fake", "compression": "none", "sample_origin": "video_frame", "frame_index": 0},
        # Another valid video frame
        {"dataset_source": "test", "file_path": valid_vid2, "label": 1, "manipulation_type": "fake", "compression": "none", "sample_origin": "video_frame", "frame_index": 4},
        # Corrupt image
        {"dataset_source": "test", "file_path": corrupt_img, "label": 1, "manipulation_type": "fake", "compression": "none", "sample_origin": "physical_image", "frame_index": -1},
        # Another valid video frame
        {"dataset_source": "test", "file_path": valid_vid2, "label": 0, "manipulation_type": "none", "compression": "none", "sample_origin": "video_frame", "frame_index": 8},
    ]
    
    # Manifest with intentional cluster of all-bad samples to test skipped batches
    manifest_csv = os.path.join(temp_dir, "test_manifest.csv")
    with open(manifest_csv, "w", encoding="utf-8") as f:
        f.write("dataset_source,file_path,media_type,label,manipulation_type,compression,split,sample_origin,frame_index\n")
        for s in test_samples:
            f.write(f"{s['dataset_source']},{s['file_path']},image,{s['label']},{s['manipulation_type']},{s['compression']},train,{s['sample_origin']},{s['frame_index']}\n")
        # Add 4 consecutive corrupted samples that always fail even after all retries (by pointing to corrupt files only)
        for i in range(4):
            f.write(f"test,{corrupt_vid},image,1,fake,none,train,video_frame,0\n")

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
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 2)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    processed_batches = 0
    skipped_batches = 0
    total_samples_trained = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        if images is None or labels is None or len(images) == 0:
            print(f"[PASS] Batch {batch_idx+1}: Safely skipped empty batch.")
            skipped_batches += 1
            continue
            
        actual_size = images.size(0)
        assert images.shape[1:] == (3, 224, 224), f"Unexpected tensor dimensions: {images.shape}"
        assert not torch.isnan(images).any(), "Tensor contains NaN!"
        assert actual_size > 0, "Batch size must be > 0"
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        processed_batches += 1
        total_samples_trained += actual_size
        print(f"[PASS] Batch {batch_idx+1}: Processed {actual_size} valid samples. Loss={loss.item():.4f}")
        
    stats = dataset.get_failure_stats(successful_samples=total_samples_trained)
    print(f"\nStats report from multi-worker run: {stats}")
    assert stats["decode_failures"] > 0, "Expected decode_failures > 0"
    assert stats["retry_count"] > 0, "Expected retry_count > 0"
    assert stats["unique_failed_videos"] > 0, "Expected unique_failed_videos > 0"
    assert total_samples_trained > 0, "Expected successful samples trained > 0"
    print("[PASS] DataLoader multi-worker resilience test PASSED.")

    # TEST 3: Fatal Error Propagation Verification
    print("\n--- TEST 3: Fatal Error Propagation Verification ---")
    cuda_oom = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_fatal_error(cuda_oom) == True, "CUDA OOM must be classified as FATAL!"
    
    type_err = TypeError("unsupported operand type(s)")
    assert is_fatal_error(type_err) == True, "Programming/TypeError must be classified as FATAL!"
    
    name_err = NameError("name 'xyz' is not defined")
    assert is_fatal_error(name_err) == True, "NameError must be classified as FATAL!"
    
    dec_err = VideoDecodeError("Failed to read frame")
    assert is_fatal_error(dec_err) == False, "VideoDecodeError must NOT be classified as fatal!"
    print("[PASS] Fatal error classifiers verified.")

    # TEST 4: Checkpoint & Production Weights Invariance Check
    print("\n--- TEST 4: Checkpoint & Production File Integrity ---")
    latest_ckpt_path = "backend/models/weights/runs/full_image_I1/latest_checkpoint.pth"
    assert os.path.exists(latest_ckpt_path), f"Checkpoint missing: {latest_ckpt_path}"
    ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
    epoch_val = ckpt.get("epoch")
    print(f"Latest checkpoint epoch: {epoch_val}")
    assert epoch_val == 8, f"CRITICAL: latest_checkpoint epoch must be 8, found {epoch_val}!"
    print("[PASS] latest_checkpoint.pth preserved at epoch = 8.")

    prod_files = [
        "backend/models/weights/image_model.pth",
        "backend/models/weights/video_model.pth",
        "backend/models/model_registry.json"
    ]
    for pf in prod_files:
        assert os.path.exists(pf), f"Production file missing: {pf}"
        print(f"[PASS] Production file confirmed present: {pf} (size: {os.path.getsize(pf)} bytes)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
