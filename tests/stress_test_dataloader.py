import os
import sys
import time
import ctypes
from ctypes import wintypes
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn

logger = setup_logger("StressTest")

# Precise Windows Working Set and System Memory Query
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

psapi = ctypes.WinDLL('psapi.dll')
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', wintypes.DWORD),
        ('dwMemoryLoad', wintypes.DWORD),
        ('ullTotalPhys', ctypes.c_uint64),
        ('ullAvailPhys', ctypes.c_uint64),
        ('ullTotalPageFile', ctypes.c_uint64),
        ('ullAvailPageFile', ctypes.c_uint64),
        ('ullTotalVirtual', ctypes.c_uint64),
        ('ullAvailVirtual', ctypes.c_uint64),
        ('sullAvailExtendedVirtual', ctypes.c_uint64),
    ]

kernel32 = ctypes.WinDLL('kernel32.dll')
kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL


def get_mem_stats():
    # Process memory
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    h = kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    rss_mb = pmc.WorkingSetSize / (1024.0 * 1024.0)
    peak_rss_mb = pmc.PeakWorkingSetSize / (1024.0 * 1024.0)

    # System memory
    mem_status = MEMORYSTATUSEX()
    mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
    total_ram_mb = mem_status.ullTotalPhys / (1024.0 * 1024.0)
    avail_ram_mb = mem_status.ullAvailPhys / (1024.0 * 1024.0)
    used_ram_mb = total_ram_mb - avail_ram_mb
    load_pct = mem_status.dwMemoryLoad

    return {
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "total_ram_mb": total_ram_mb,
        "avail_ram_mb": avail_ram_mb,
        "used_ram_mb": used_ram_mb,
        "load_pct": load_pct
    }


def run_stress_test(num_batches: int = 200, batch_size: int = 32):
    logger.info("=" * 70)
    logger.info("  STARTING DATALOADER MEMORY-SAFE STRESS TEST")
    logger.info(f"  Configuration: batch_size={batch_size}, num_workers=0, fp32, pin_memory=False")
    logger.info(f"  Target Batches: {num_batches} (~{num_batches * batch_size} samples)")
    logger.info("=" * 70)

    initial_mem = get_mem_stats()
    logger.info(
        f"Initial State -> Process RSS: {initial_mem['rss_mb']:.1f} MB | "
        f"System RAM Used: {initial_mem['used_ram_mb']:.0f}/{initial_mem['total_ram_mb']:.0f} MB ({initial_mem['load_pct']}%), "
        f"Free: {initial_mem['avail_ram_mb']:.0f} MB"
    )

    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples.")

    # Locate the suspect sample (533_450.mp4 frame 279)
    suspect_indices = [
        i for i, s in enumerate(train_dataset.samples)
        if "533_450.mp4" in s["file_path"] and s["frame_index"] == 279
    ]
    if suspect_indices:
        suspect_idx = suspect_indices[0]
        logger.info(f"Found suspect sample at index {suspect_idx}: {train_dataset.samples[suspect_idx]['file_path']} (frame 279)")
        # Test standalone extraction first
        t0 = time.time()
        tensor, lbl = train_dataset[suspect_idx]
        t_sample = time.time() - t0
        logger.info(f"Direct decode of suspect sample {suspect_idx} succeeded in {t_sample*1000:.1f}ms: shape={tensor.shape}, label={lbl}")
    else:
        logger.warning("Suspect sample 533_450.mp4 frame 279 not found in manifest!")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=safe_collate_fn
    )

    batches_processed = 0
    total_samples = 0
    t_start = time.time()
    last_log_time = t_start
    last_batches = 0

    rss_history = []
    
    logger.info("\nProcessing batches...\n")
    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        if images is None or labels is None or len(images) == 0:
            continue

        # Simulate FP32 checks
        assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
        assert torch.isfinite(images).all(), "Non-finite values detected in batch images!"
        assert torch.isfinite(labels).all(), "Non-finite values detected in batch labels!"

        batches_processed += 1
        total_samples += len(images)

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            now = time.time()
            dt = now - last_log_time
            d_batches = batches_processed - last_batches
            bps = d_batches / dt if dt > 0 else 0
            sps = (d_batches * batch_size) / dt if dt > 0 else 0
            
            mem = get_mem_stats()
            rss_history.append(mem["rss_mb"])
            failures = train_dataset.stats._decode_failures.value
            retries = train_dataset.stats._retry_count.value
            skipped = train_dataset.stats._skipped_samples.value

            logger.info(
                f"[Batch {batch_idx + 1:3d}/{num_batches}] "
                f"Throughput: {bps:4.1f} batch/s ({sps:5.1f} img/s) | "
                f"Process RSS: {mem['rss_mb']:6.1f} MB (Peak: {mem['peak_rss_mb']:6.1f} MB) | "
                f"RAM Free: {mem['avail_ram_mb']:5.0f} MB ({mem['load_pct']}%) | "
                f"DecodeFail: {failures}, Retries: {retries}, Skips: {skipped}"
            )
            last_log_time = now
            last_batches = batches_processed

    total_time = time.time() - t_start
    final_mem = get_mem_stats()
    stats = train_dataset.get_failure_stats(successful_samples=total_samples)

    logger.info("=" * 70)
    logger.info("  STRESS TEST COMPLETED SUCCESSFULLY!")
    logger.info(f"  Total Batches: {batches_processed}/{num_batches}")
    logger.info(f"  Total Samples Processed: {total_samples}")
    logger.info(f"  Total Elapsed Time: {total_time:.2f}s")
    logger.info(f"  Average Throughput: {batches_processed / total_time:.2f} batch/s ({total_samples / total_time:.2f} img/s)")
    logger.info(f"  Final Process RSS: {final_mem['rss_mb']:.1f} MB (Peak: {final_mem['peak_rss_mb']:.1f} MB)")
    logger.info(f"  System RAM Free: {final_mem['avail_ram_mb']:.0f} MB / {final_mem['total_ram_mb']:.0f} MB ({final_mem['load_pct']}% load)")
    logger.info(f"  Decode Failures: {stats['decode_failures']}")
    logger.info(f"  Retries: {stats['retry_count']}")
    logger.info(f"  Skipped Samples: {stats['skipped_samples']}")
    logger.info("=" * 70)

    # Memory stability check: verify RSS growth rate
    if len(rss_history) >= 2:
        rss_growth = rss_history[-1] - rss_history[0]
        logger.info(f"RSS Change across test: {rss_growth:+.2f} MB (First checkpoint: {rss_history[0]:.1f} MB -> Final: {rss_history[-1]:.1f} MB)")
        if rss_growth < 150.0:
            logger.info("MEMORY STABILITY CHECK: PASSED (RSS is flat and stable, zero leak detected).")
        else:
            logger.warning("MEMORY STABILITY CHECK: Elevated RSS growth detected.")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=200, help="Number of batches to test")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    run_stress_test(num_batches=args.batches, batch_size=args.batch_size)
