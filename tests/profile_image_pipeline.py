import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.utils.config import PipelineConfig
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from backend.models.image_model import ImageModel
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.face_detector import FaceDetector
from backend.training.train_image_model import get_cuda_memory_mb, get_gpu_utilization, get_process_memory_mb


def profile_cpu_preprocessing_micro(dataset: DeepfakeDataset, num_samples: int = 50):
    """
    Measures the exact CPU time breakdown per sample:
    1. Video extraction / disk read
    2. Haar Cascade face detection
    3. Crop, resize (224x224), BGR->RGB
    4. Augmentation & PyTorch normalization
    Also tests bypassing Haar Cascade (direct resize) to quantify the Haar bottleneck.
    """
    print("\n" + "=" * 75)
    print(f"STAGE 1: CPU PREPROCESSING MICRO-BENCHMARK ({num_samples} Representative Samples)")
    print("=" * 75)

    detector = FaceDetector()
    preprocessor = ImagePreprocessor(face_detector=detector)

    indices = list(range(min(num_samples, len(dataset.samples))))

    t_disk_reads = []
    t_haar_detects = []
    t_crop_resizes = []
    t_aug_norms = []
    t_totals = []
    t_no_haar_totals = []
    faces_detected_count = 0

    for idx in indices:
        sample = dataset.samples[idx]
        file_path = sample["file_path"]
        origin = sample.get("sample_origin", "physical_image")
        frame_idx = sample.get("frame_index", -1)

        t0 = time.perf_counter()

        # Step 1: Disk Read / Video Frame Extraction
        t_disk_start = time.perf_counter()
        frame = None
        if origin == "video_frame":
            cap = None
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
            finally:
                if cap is not None:
                    cap.release()
        else:
            frame = cv2.imread(file_path)
        t_disk_end = time.perf_counter()

        if frame is None:
            continue

        dur_disk = (t_disk_end - t_disk_start) * 1000.0  # ms
        t_disk_reads.append(dur_disk)

        # Step 2: Haar Cascade Face Detection
        t_haar_start = time.perf_counter()
        faces = detector.detect_faces(frame)
        t_haar_end = time.perf_counter()
        dur_haar = (t_haar_end - t_haar_start) * 1000.0  # ms
        t_haar_detects.append(dur_haar)

        if faces:
            faces_detected_count += 1
            frame_cropped = detector.crop_face(frame, faces[0])
            if frame_cropped is None or getattr(frame_cropped, "size", 0) == 0:
                frame_cropped = frame
        else:
            frame_cropped = frame

        # Step 3: Resize to 224x224 & BGR -> RGB
        t_resize_start = time.perf_counter()
        img_resized = cv2.resize(frame_cropped, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        t_resize_end = time.perf_counter()
        dur_resize = (t_resize_end - t_resize_start) * 1000.0  # ms
        t_crop_resizes.append(dur_resize)

        # Step 4: Augmentation & Tensor Normalization
        t_aug_start = time.perf_counter()
        tensor = preprocessor.apply_augmentation_and_normalization(img_rgb, is_training=True)
        t_aug_end = time.perf_counter()
        dur_aug = (t_aug_end - t_aug_start) * 1000.0  # ms
        t_aug_norms.append(dur_aug)

        total_sample_ms = (time.perf_counter() - t0) * 1000.0
        t_totals.append(total_sample_ms)

        # Baseline comparison: time WITHOUT Haar cascade (direct resize)
        dur_no_haar = dur_disk + dur_resize + dur_aug
        t_no_haar_totals.append(dur_no_haar)

    avg_disk = np.mean(t_disk_reads) if t_disk_reads else 0.0
    avg_haar = np.mean(t_haar_detects) if t_haar_detects else 0.0
    avg_resize = np.mean(t_crop_resizes) if t_crop_resizes else 0.0
    avg_aug = np.mean(t_aug_norms) if t_aug_norms else 0.0
    avg_total = np.mean(t_totals) if t_totals else 0.0
    avg_no_haar = np.mean(t_no_haar_totals) if t_no_haar_totals else 0.0

    print(f"Samples Evaluated:             {len(t_totals)}")
    print(f"Face Detection Hit Rate:       {faces_detected_count}/{len(t_totals)} ({faces_detected_count / max(1, len(t_totals)) * 100:.1f}%)")
    print("-" * 75)
    print(f"1. Video Decode / Disk Read:   {avg_disk:6.2f} ms ({avg_disk / max(1e-5, avg_total) * 100:5.1f}%)")
    print(f"2. Haar Face Detection (CPU):  {avg_haar:6.2f} ms ({avg_haar / max(1e-5, avg_total) * 100:5.1f}%) [BOTTLENECK]")
    print(f"3. Crop + Resize (224x224):    {avg_resize:6.2f} ms ({avg_resize / max(1e-5, avg_total) * 100:5.1f}%)")
    print(f"4. Augmentation & Normalization: {avg_aug:6.2f} ms ({avg_aug / max(1e-5, avg_total) * 100:5.1f}%)")
    print("-" * 75)
    print(f"TOTAL CPU Time Per Sample:     {avg_total:6.2f} ms ({1000.0 / max(1e-5, avg_total):.2f} samples/sec/core)")
    print(f"TOTAL Without Haar Cascade:    {avg_no_haar:6.2f} ms ({1000.0 / max(1e-5, avg_no_haar):.2f} samples/sec/core)")
    print(f"Haar Cascade Speedup Potential: {avg_total / max(1e-5, avg_no_haar):.2f}x faster if Haar bypassed")
    print("=" * 75 + "\n")

    return {
        "avg_disk_ms": avg_disk,
        "avg_haar_ms": avg_haar,
        "avg_resize_ms": avg_resize,
        "avg_aug_ms": avg_aug,
        "avg_total_ms": avg_total,
        "avg_no_haar_ms": avg_no_haar,
        "face_hit_rate": faces_detected_count / max(1, len(t_totals))
    }


def benchmark_training_configuration(
    dataset: DeepfakeDataset,
    batch_size: int,
    num_workers: int,
    num_batches: int = 5,
    warmup_batches: int = 2
) -> dict:
    """
    Safely benchmarks a single training configuration:
    - FP32 enabled (no autocast)
    - Measures data loading wait time vs GPU compute time
    - Monitors RAM (process working set) and GPU VRAM
    - Does NOT modify any checkpoint or trained weights
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # In-memory evaluation model (pretrained backbone, eval BN for layers 1-3)
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.train()
    for name, module in model.backbone.named_modules():
        if ("layer4" not in name and "fc" not in name) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create subset to ensure small representative footprint
    total_needed = (warmup_batches + num_batches) * batch_size
    subset_indices = list(range(min(total_needed, len(dataset))))
    subset = Subset(dataset, subset_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
        "collate_fn": safe_collate_fn
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(subset, **loader_kwargs)

    data_wait_times = []
    gpu_compute_times = []
    processed_samples = 0
    batch_count = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    start_rss = get_process_memory_mb()
    t_wait_start = time.perf_counter()

    for idx, (images, labels) in enumerate(loader):
        t_wait_end = time.perf_counter()
        wait_ms = (t_wait_end - t_wait_start) * 1000.0

        if images is None or labels is None:
            t_wait_start = time.perf_counter()
            continue

        actual_b = images.size(0)

        # Warmup vs measured
        is_warmup = (idx < warmup_batches)

        t_comp_start = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        # FP32 forward + backward (explicitly disable AMP)
        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t_comp_end = time.perf_counter()
        comp_ms = (t_comp_end - t_comp_start) * 1000.0

        if not is_warmup:
            data_wait_times.append(wait_ms)
            gpu_compute_times.append(comp_ms)
            processed_samples += actual_b
            batch_count += 1

        if batch_count >= num_batches:
            break

        t_wait_start = time.perf_counter()

    avg_wait = np.mean(data_wait_times) if data_wait_times else 0.0
    avg_comp = np.mean(gpu_compute_times) if gpu_compute_times else 0.0
    total_batch_time = avg_wait + avg_comp

    samples_per_sec = (batch_size / (total_batch_time / 1000.0)) if total_batch_time > 0 else 0.0
    batches_per_sec = (1.0 / (total_batch_time / 1000.0)) if total_batch_time > 0 else 0.0

    # Theoretical GPU duty cycle (% of total loop time the GPU was actually calculating)
    gpu_duty_cycle = (avg_comp / max(1e-5, total_batch_time)) * 100.0

    vram_info = get_cuda_memory_mb(device)
    end_rss = get_process_memory_mb()
    gpu_util = get_gpu_utilization()

    full_epoch_samples = 51357
    est_full_epoch_sec = full_epoch_samples / max(1e-5, samples_per_sec)
    est_full_epoch_hours = est_full_epoch_sec / 3600.0

    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "precision": "FP32",
        "avg_data_wait_ms": avg_wait,
        "avg_gpu_compute_ms": avg_comp,
        "total_batch_ms": total_batch_time,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": batches_per_sec,
        "gpu_duty_cycle_pct": gpu_duty_cycle,
        "gpu_util_smi": gpu_util,
        "vram_peak_mb": vram_info["peak"],
        "rss_ram_mb": end_rss,
        "rss_delta_mb": end_rss - start_rss,
        "est_epoch_hours": est_full_epoch_hours,
        "est_epoch_min": est_full_epoch_sec / 60.0
    }


def run_full_profiling_suite():
    print("=" * 80)
    print("      SATYADRISHTI IMAGE PIPELINE PROFILER & BOTTLENECK ANALYSIS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Precision Mode:   FP32 (Autocast Disabled)")

    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    print(f"Dataset Size:     {len(dataset.samples)} samples")

    # 1. Micro-profiling of CPU preprocessing
    micro_stats = profile_cpu_preprocessing_micro(dataset, num_samples=30)

    # 2. Benchmark Matrix: BS 64 vs 32, Workers 0 vs 2
    matrix_configs = [
        {"batch_size": 64, "num_workers": 0},
        {"batch_size": 64, "num_workers": 2},
        {"batch_size": 32, "num_workers": 0},
        {"batch_size": 32, "num_workers": 2},
    ]

    print("\n" + "=" * 80)
    print("STAGE 2: MULTI-CONFIGURATION BENCHMARK MATRIX (FP32)")
    print("=" * 80)

    results = []
    for cfg in matrix_configs:
        bs = cfg["batch_size"]
        nw = cfg["num_workers"]
        print(f"\n--> Running Benchmark: Batch Size = {bs}, DataLoader Workers = {nw} (FP32)...")
        res = benchmark_training_configuration(
            dataset=dataset,
            batch_size=bs,
            num_workers=nw,
            num_batches=4,
            warmup_batches=1
        )
        results.append(res)
        print(f"    Data Wait: {res['avg_data_wait_ms']:.1f} ms | GPU Compute: {res['avg_gpu_compute_ms']:.1f} ms")
        print(f"    Throughput: {res['samples_per_sec']:.2f} img/s | GPU Duty Cycle: {res['gpu_duty_cycle_pct']:.1f}%")
        print(f"    Peak VRAM: {res['vram_peak_mb']:.1f} MB | Process RAM (RSS): {res['rss_ram_mb']:.1f} MB")
        print(f"    Estimated Epoch Time: {res['est_epoch_hours']:.2f} hrs ({res['est_epoch_min']:.1f} mins)")

    # Print Formatted Comparison Table
    print("\n" + "=" * 105)
    print(f"{'BS':<4} | {'WORKERS':<7} | {'DATA WAIT':<12} | {'GPU COMP':<10} | {'THROUGHPUT':<13} | {'GPU DUTY':<9} | {'RAM (RSS)':<10} | {'EST. EPOCH':<12}")
    print("=" * 105)
    for r in results:
        print(
            f"{r['batch_size']:<4} | "
            f"{r['num_workers']:<7} | "
            f"{r['avg_data_wait_ms']:>8.1f} ms | "
            f"{r['avg_gpu_compute_ms']:>6.1f} ms | "
            f"{r['samples_per_sec']:>7.2f} img/s | "
            f"{r['gpu_duty_cycle_pct']:>7.1f}% | "
            f"{r['rss_ram_mb']:>7.1f} MB | "
            f"{r['est_epoch_hours']:>5.2f} hours"
        )
    print("=" * 105)

    return micro_stats, results


if __name__ == "__main__":
    run_full_profiling_suite()
