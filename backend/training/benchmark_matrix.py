import os
import sys
import time
import json
import traceback
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.training.dataset_loader import DeepfakeDataset
from backend.models.image_model import ImageModel
from backend.training.train_image_model import get_cuda_memory_mb, get_gpu_utilization, get_process_memory_mb

logger = setup_logger("BenchmarkMatrix")

def run_single_config_benchmark(
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int,
    target_batches: int = 200
) -> dict:
    config_name = f"BS={batch_size}, Workers={num_workers}, Persist={persistent_workers}, Prefetch={prefetch_factor}"
    logger.info("==================================================")
    logger.info(f"STARTING BENCHMARK: {config_name}")
    logger.info("==================================================")
    
    result = {
        "config_name": config_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "batches_per_sec": 0.0,
        "samples_per_sec": 0.0,
        "gpu_utilization": 0.0,
        "allocated_vram_mb": 0.0,
        "peak_vram_mb": 0.0,
        "cpu_rss_mb": 0.0,
        "avg_data_wait_ms": 0.0,
        "avg_gpu_compute_ms": 0.0,
        "failures_retries": 0,
        "exceptions_crashes": [],
        "stability_status": "STABLE"
    }
    
    device = torch.device(PipelineConfig.DEVICE)
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    
    try:
        train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
        model = ImageModel(num_classes=2, pretrained=True).to(device)
        model.train()
        
        # Split backbone/head LR split as per pipeline setup
        for name, param in model.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        use_amp = (device.type == "cuda")
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        loader_kwargs = {
            "batch_size": min(batch_size, len(train_dataset)),
            "num_workers": num_workers,
            "pin_memory": (device.type == "cuda")
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            if prefetch_factor > 0:
                loader_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            
        processed_batches = 0
        processed_samples = 0
        
        data_wait_times = []
        gpu_compute_times = []

        overall_start = time.time()
        loader_iter = iter(train_loader)
        
        t_wait_start = time.perf_counter()
        
        for batch_idx in range(target_batches):
            try:
                images, batch_labels = next(loader_iter)
            except StopIteration:
                break
            except Exception as e:
                err_str = f"DataLoader next() Exception at batch {batch_idx+1}: {str(e)}"
                logger.error(err_str)
                result["exceptions_crashes"].append(err_str)
                result["stability_status"] = "UNSTABLE/UNSAFE"
                break
                
            t_wait_end = time.perf_counter()
            data_wait_times.append((t_wait_end - t_wait_start) * 1000.0)
            
            t_comp_start = time.perf_counter()
            
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                
            t_comp_end = time.perf_counter()
            gpu_compute_times.append((t_comp_end - t_comp_start) * 1000.0)
            
            processed_batches += 1
            processed_samples += images.size(0)
            
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == target_batches:
                vram_info = get_cuda_memory_mb(device)
                logger.info(
                    f"[{config_name}] Batch [{batch_idx+1}/{target_batches}] | "
                    f"Last Wait: {data_wait_times[-1]:.1f}ms | Last Comp: {gpu_compute_times[-1]:.1f}ms | "
                    f"Peak VRAM: {vram_info['peak']:.1f}MB"
                )
            
            t_wait_start = time.perf_counter()
            
        overall_elapsed = time.time() - overall_start
        
        if processed_batches > 0:
            result["batches_per_sec"] = processed_batches / max(1e-5, overall_elapsed)
            result["samples_per_sec"] = processed_samples / max(1e-5, overall_elapsed)
            result["avg_data_wait_ms"] = float(np.mean(data_wait_times)) if data_wait_times else 0.0
            result["avg_gpu_compute_ms"] = float(np.mean(gpu_compute_times)) if gpu_compute_times else 0.0
            
        vram_info = get_cuda_memory_mb(device)
        result["allocated_vram_mb"] = vram_info["allocated"]
        result["peak_vram_mb"] = vram_info["peak"]
        result["gpu_utilization"] = get_gpu_utilization()
        result["cpu_rss_mb"] = get_process_memory_mb()
        
        failures = getattr(train_dataset, "failed_sample_count", 0)
        retries = getattr(train_dataset, "retry_count", 0)
        result["failures_retries"] = failures + retries
        
        # Check if peak VRAM exceeded safe threshold (3.5 GB = 3584 MB)
        if result["peak_vram_mb"] > 3584.0:
            result["stability_status"] = "UNSAFE (VRAM > 3.5GB)"
            
        logger.info(
            f"FINISHED {config_name} | Status: {result['stability_status']} | "
            f"{result['batches_per_sec']:.2f} b/s ({result['samples_per_sec']:.2f} s/s) | "
            f"Peak VRAM: {result['peak_vram_mb']:.1f}MB | GPU Util: {result['gpu_utilization']:.0f}% | "
            f"DataWait: {result['avg_data_wait_ms']:.1f}ms | GPUComp: {result['avg_gpu_compute_ms']:.1f}ms"
        )
        
    except Exception as e:
        err_msg = f"Fatal configuration execution error: {str(e)}\n{traceback.format_exc()}"
        logger.error(err_msg)
        result["exceptions_crashes"].append(str(e))
        result["stability_status"] = "UNSTABLE/CRASHED"
        
    return result

def run_benchmark_matrix():
    configs = [
        # (batch_size, num_workers, persistent_workers, prefetch_factor)
        (8, 0, False, 0),
        (8, 2, True, 2),
        (8, 4, True, 2),
        (16, 0, False, 0),
        (16, 2, True, 2),
        (16, 4, True, 2),
        (16, 4, True, 4),
        (24, 0, False, 0),
        (24, 2, True, 2),
        (24, 4, True, 2),
    ]
    
    results = []
    logger.info(f"Starting Multi-Configuration Benchmark Matrix across {len(configs)} candidate configurations...")
    
    for bs, nw, pw, pf in configs:
        res = run_single_config_benchmark(
            batch_size=bs,
            num_workers=nw,
            persistent_workers=pw,
            prefetch_factor=pf,
            target_batches=100
        )
        results.append(res)
        
    # Rank results by throughput (samples/sec) descending
    ranked_results = sorted(results, key=lambda x: x["samples_per_sec"], reverse=True)
    
    output_path = os.path.join(PipelineConfig.RESULTS_DIR, "image", "benchmark_matrix_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ranked_results, f, indent=4)
        
    print("\n" + "="*110)
    print(f"{'RANK':<5} | {'CONFIGURATION':<42} | {'SAMPLES/SEC':<11} | {'GPU UTIL':<8} | {'PEAK VRAM':<10} | {'DATA WAIT':<10} | {'GPU COMP':<9} | {'STATUS':<15}")
    print("="*110)
    
    for rank, r in enumerate(ranked_results, 1):
        config_fmt = f"BS={r['batch_size']}, Workers={r['num_workers']}, Persist={r['persistent_workers']}, Prefetch={r['prefetch_factor']}"
        gpu_u = f"{r['gpu_utilization']:.0f}%" if r['gpu_utilization'] >= 0 else "N/A"
        print(f"{rank:<5} | {config_fmt:<42} | {r['samples_per_sec']:<11.2f} | {gpu_u:<8} | {r['peak_vram_mb']:<10.1f} MB | {r['avg_data_wait_ms']:<10.1f} ms | {r['avg_gpu_compute_ms']:<9.1f} ms | {r['stability_status']:<15}")
    print("="*110 + "\n")
    
    logger.info(f"Benchmark matrix completed. Full results saved to: {output_path}")

if __name__ == "__main__":
    run_benchmark_matrix()
