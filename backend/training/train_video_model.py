import os

# Configure OpenBLAS and numerical libraries to conservative thread limits BEFORE importing numpy/torch/cv2
# Prevents host memory allocation failure and thread thrashing on CPU
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import argparse
import random
import json
import shutil
import time
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

# PyTorch thread settings: conservative thread execution to reduce host RAM pressure
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

# Explicitly disable OpenCV OpenCL execution path and limit thread pool
cv2.ocl.setUseOpenCL(False)
if hasattr(cv2, "setNumThreads"):
    cv2.setNumThreads(1)
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

import sys
_workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _workspace_dir not in sys.path:
    sys.path.insert(0, _workspace_dir)

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.dataset_loader import DatasetLoader
from backend.training.video_dataset import VideoDataset
from backend.models.video_model import VideoModel
from backend.preprocessing.video_preprocessor import VideoPreprocessor
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from backend.utils.metrics_helper import compute_ece, optimize_threshold, compute_all_metrics

logger = setup_logger("TrainVideoModel")


# ==============================================================================
# 1. System Profiling & Diagnostics Helpers
# ==============================================================================

def get_process_memory_mb() -> float:
    """Returns current process Working Set (RSS) memory in MB on Windows/Linux."""
    try:
        import ctypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)
            ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def get_cuda_memory_mb(device) -> dict:
    """Returns current CUDA allocated, reserved, and peak VRAM in MB."""
    if not torch.cuda.is_available() or (isinstance(device, torch.device) and device.type != 'cuda'):
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(device) / (1024.0 * 1024.0),
        "reserved": torch.cuda.memory_reserved(device) / (1024.0 * 1024.0),
        "peak": torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    }


def get_gpu_utilization() -> float:
    """Safely queries GPU utilization percentage using nvidia-smi."""
    try:
        import subprocess
        cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True, timeout=2, stderr=subprocess.DEVNULL).decode().strip()
        return float(out.split()[0])
    except Exception:
        return -1.0


# ==============================================================================
# 2. Numerical Stability Guards
# ==============================================================================

class NumericalInstabilityError(RuntimeError):
    """Exception raised when numerical instability (NaN/Inf) is detected during training."""
    def __init__(self, stage: str, tensor_name: str, epoch: int, batch_idx: int, lr: float, details: str = "", sample_info: str = ""):
        self.stage = stage
        self.tensor_name = tensor_name
        self.epoch = epoch
        self.batch_idx = batch_idx
        self.lr = lr
        self.details = details
        self.sample_info = sample_info
        super().__init__(
            f"Numerical instability (NaN/Inf) detected in stage '{stage}' "
            f"for tensor '{tensor_name}' at Epoch {epoch}, Batch {batch_idx} (LR: {lr}). "
            f"{details} {sample_info}".strip()
        )


def check_tensor_finite(
    tensor: torch.Tensor,
    stage: str,
    tensor_name: str,
    epoch: int,
    batch_idx: int,
    lr: float,
    sample_info: str = ""
) -> None:
    """Validates that a tensor contains only finite numbers, raising NumericalInstabilityError on failure."""
    if tensor is None or not torch.is_tensor(tensor):
        return
    if bool(torch.isfinite(tensor).all()):
        return

    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())
    total_elements = tensor.numel()

    finite_vals = tensor[~(nan_mask | inf_mask)]
    if finite_vals.numel() > 0:
        min_v = f"{finite_vals.min().item():.6e}"
        max_v = f"{finite_vals.max().item():.6e}"
        mean_v = f"{finite_vals.mean().item():.6e}"
    else:
        min_v, max_v, mean_v = "None", "None", "None"

    diag_details = (
        f"\n[NUMERICAL INSTABILITY DETECTED]\n"
        f"  Stage: {stage} | Component: {tensor_name}\n"
        f"  Epoch: {epoch} | Batch: {batch_idx} | Learning Rate: {lr}\n"
        f"  Total Elements: {total_elements} | NaN Count: {nan_count} | Inf Count: {inf_count}\n"
        f"  Finite Elements Stats -> Min: {min_v}, Max: {max_v}, Mean: {mean_v}\n"
    )
    logger.error(diag_details)
    raise NumericalInstabilityError(stage, tensor_name, epoch, batch_idx, lr, diag_details, sample_info)


def check_labels_finite(
    labels: torch.Tensor,
    stage: str,
    tensor_name: str,
    epoch: int,
    batch_idx: int,
    lr: float,
    num_classes: int = 2,
    sample_info: str = ""
) -> None:
    """Validates that target labels are finite and within [0, num_classes-1]."""
    if labels is None or not torch.is_tensor(labels):
        return
    if bool(torch.isfinite(labels).all()) and (labels >= 0).all() and (labels < num_classes).all():
        return
    msg = f"Target labels out of bounds [0, {num_classes-1}] or non-finite: min={labels.min().item()}, max={labels.max().item()}"
    logger.error(msg)
    raise NumericalInstabilityError(stage, tensor_name, epoch, batch_idx, lr, msg, sample_info)


# ==============================================================================
# 3. Resumable, Atomic Checkpointing
# ==============================================================================

def validate_checkpoint_state(state: dict) -> tuple:
    """
    Validates that all metrics, model parameters, buffers, and optimizer states
    within a checkpoint dictionary are completely finite (no NaN, +Inf, -Inf).
    Returns (is_valid: bool, list_of_errors: list).
    """
    errors = []
    if not isinstance(state, dict):
        return False, ["Checkpoint root is not a dictionary"]

    for metric_name in ["train_loss", "val_loss", "best_val_loss"]:
        val = state.get(metric_name)
        if val is not None:
            if isinstance(val, (float, int)):
                if not np.isfinite(val):
                    errors.append(f"Scalar metric '{metric_name}' is non-finite: {val}")
            elif isinstance(val, torch.Tensor):
                if not torch.isfinite(val).all():
                    errors.append(f"Tensor metric '{metric_name}' is non-finite: {val}")

    model_state = state.get("model_state_dict", {})
    if isinstance(model_state, dict):
        for name, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor):
                if not torch.isfinite(tensor).all():
                    nan_c = torch.isnan(tensor).sum().item()
                    inf_c = torch.isinf(tensor).sum().item()
                    errors.append(f"Model tensor '{name}' contains non-finite values (NaN: {nan_c}, Inf: {inf_c})")

    opt_state = state.get("optimizer_state_dict")
    if isinstance(opt_state, dict):
        state_dict = opt_state.get("state", {})
        for p_id, p_state in state_dict.items():
            if isinstance(p_state, dict):
                for s_name, s_val in p_state.items():
                    if isinstance(s_val, torch.Tensor):
                        if not torch.isfinite(s_val).all():
                            nan_c = torch.isnan(s_val).sum().item()
                            inf_c = torch.isinf(s_val).sum().item()
                            errors.append(f"Optimizer param {p_id} state '{s_name}' contains non-finite values (NaN: {nan_c}, Inf: {inf_c})")

    return (len(errors) == 0, errors)


def save_checkpoint_atomic(state: dict, filepath: str, allow_corrupted: bool = False) -> bool:
    """
    Saves a training checkpoint atomically to prevent corruption on process interruption.
    Guarantees that a numerically corrupted checkpoint never overwrites a healthy checkpoint.
    """
    is_valid, errors = validate_checkpoint_state(state)
    if not is_valid and not allow_corrupted:
        err_summary = "; ".join(errors[:5])
        if len(errors) > 5:
            err_summary += f" ... ({len(errors)-5} more errors)"
        logger.error(
            f"REFUSING TO SAVE CORRUPTED CHECKPOINT to '{filepath}'! "
            f"Numerical corruption detected ({len(errors)} issues): {err_summary}"
        )
        diag_filepath = filepath.replace(".pth", "_corrupted_diagnostic.pth")
        try:
            os.makedirs(os.path.dirname(diag_filepath), exist_ok=True)
            tmp_diag = diag_filepath + ".tmp"
            torch.save(state, tmp_diag)
            os.replace(tmp_diag, diag_filepath)
            logger.warning(f"Corrupted state preserved separately for post-mortem diagnostics at: '{diag_filepath}'")
        except Exception as e:
            logger.warning(f"Could not save corrupted diagnostic file: {e}")
        return False

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_filepath = filepath + ".tmp"
    torch.save(state, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    return True


def restore_rng_states(rs: dict):
    """Restores PyTorch, CUDA, NumPy, and Python RNG states from checkpoint dictionary."""
    if not isinstance(rs, dict):
        return

    torch_state_raw = rs.get("torch")
    if torch_state_raw is not None:
        try:
            if isinstance(torch_state_raw, torch.Tensor):
                rng_state = torch_state_raw.to(dtype=torch.uint8, device="cpu")
            elif isinstance(torch_state_raw, np.ndarray):
                rng_state = torch.from_numpy(torch_state_raw).to(dtype=torch.uint8, device="cpu")
            else:
                rng_state = torch.tensor(torch_state_raw, dtype=torch.uint8, device="cpu")
            if isinstance(rng_state, torch.Tensor) and rng_state.dtype == torch.uint8 and not rng_state.is_cuda:
                torch.set_rng_state(rng_state)
                logger.info("Restored Torch RNG state successfully.")
        except Exception as e:
            logger.warning(f"Failed to restore Torch RNG state: {e}")

    cuda_state_raw = rs.get("cuda")
    if cuda_state_raw is not None and torch.cuda.is_available():
        try:
            raw_items = cuda_state_raw if isinstance(cuda_state_raw, (list, tuple)) else [cuda_state_raw]
            converted_cuda_states = []
            for item in raw_items:
                if isinstance(item, torch.Tensor):
                    c_state = item.to(dtype=torch.uint8, device="cpu")
                else:
                    c_state = torch.tensor(item, dtype=torch.uint8, device="cpu")
                converted_cuda_states.append(c_state)
            torch.cuda.set_rng_state_all(converted_cuda_states)
            logger.info("Restored CUDA RNG state successfully.")
        except Exception as e:
            logger.warning(f"Failed to restore CUDA RNG state: {e}")

    numpy_state = rs.get("numpy")
    if numpy_state is not None:
        try:
            np.random.set_state(numpy_state)
            logger.info("Restored NumPy RNG state successfully.")
        except Exception as e:
            logger.warning(f"Failed to restore NumPy RNG state: {e}")

    python_state = rs.get("python")
    if python_state is not None:
        try:
            random.setstate(python_state)
            logger.info("Restored Python random state successfully.")
        except Exception as e:
            logger.warning(f"Failed to restore Python random state: {e}")


# ==============================================================================
# 4. Data Leakage Prevention Check
# ==============================================================================

def check_data_leakage(train_ds, val_ds, test_ds):
    """
    Checks for leakage between train, validation, and test datasets at the identity/source-video level.
    Halts training immediately if leakage is detected.
    """
    logger.info("Running automated data leakage checks at source-video and identity levels...")
    
    def get_source_id(sample):
        return os.path.basename(sample["file_path"])
        
    train_sources = set(get_source_id(s) for s in train_ds.samples)
    val_sources = set(get_source_id(s) for s in val_ds.samples)
    test_sources = set(get_source_id(s) for s in test_ds.samples) if test_ds is not None else set()
    
    train_val_leak = train_sources.intersection(val_sources)
    train_test_leak = train_sources.intersection(test_sources)
    val_test_leak = val_sources.intersection(test_sources)
    
    if train_val_leak:
        raise ValueError(f"CRITICAL DATA LEAKAGE DETECTED: {len(train_val_leak)} source files overlap between Train and Val!")
    if train_test_leak:
        raise ValueError(f"CRITICAL DATA LEAKAGE DETECTED: {len(train_test_leak)} source files overlap between Train and Test!")
    if val_test_leak:
        raise ValueError(f"CRITICAL DATA LEAKAGE DETECTED: {len(val_test_leak)} source files overlap between Val and Test!")
        
    # Extract identity
    def extract_identity(file_path):
        filename = os.path.basename(file_path)
        import re
        match = re.match(r'^(id\d+)_', filename)
        if match:
            return match.group(1)
        return None
        
    train_ids = set(extract_identity(s["file_path"]) for s in train_ds.samples if extract_identity(s["file_path"]) is not None)
    val_ids = set(extract_identity(s["file_path"]) for s in val_ds.samples if extract_identity(s["file_path"]) is not None)
    test_ids = set(extract_identity(s["file_path"]) for s in test_ds.samples if extract_identity(s["file_path"]) is not None) if test_ds is not None else set()
    
    id_train_val_leak = train_ids.intersection(val_ids)
    id_train_test_leak = train_ids.intersection(test_ids)
    id_val_test_leak = val_ids.intersection(test_ids)
    
    if id_train_val_leak:
        raise ValueError(f"CRITICAL IDENTITY LEAKAGE DETECTED: {len(id_train_val_leak)} identities overlap between Train and Val! overlapping: {id_train_val_leak}")
    if id_train_test_leak:
        raise ValueError(f"CRITICAL IDENTITY LEAKAGE DETECTED: {len(id_train_test_leak)} identities overlap between Train and Test! overlapping: {id_train_test_leak}")
    if id_val_test_leak:
        raise ValueError(f"CRITICAL IDENTITY LEAKAGE DETECTED: {len(id_val_test_leak)} identities overlap between Val and Test! overlapping: {id_val_test_leak}")
        
    logger.info("Data leakage and identity checks passed. No overlapping source files or identities found between splits.")


def generate_dummy_data(num_samples: int = 4):
    """Generates dummy real and fake video files using OpenCV for quick tests."""
    logger.warning("[DUMMY DATA — TEST ONLY] Generating dummy video samples...")
    os.makedirs(PipelineConfig.VIDEO_REAL_DIR, exist_ok=True)
    os.makedirs(PipelineConfig.VIDEO_FAKE_DIR, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for i in range(num_samples):
        real_path = os.path.join(PipelineConfig.VIDEO_REAL_DIR, f"dummy_real_{i}.mp4")
        fake_path = os.path.join(PipelineConfig.VIDEO_FAKE_DIR, f"dummy_fake_{i}.mp4")
        
        out_real = cv2.VideoWriter(real_path, fourcc, 10.0, (224, 224))
        for _ in range(16):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.circle(frame, (112, 112), 40, (0, 255, 0), -1)
            out_real.write(frame)
        out_real.release()
        
        out_fake = cv2.VideoWriter(fake_path, fourcc, 10.0, (224, 224))
        for _ in range(16):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.circle(frame, (112, 112), 40, (0, 0, 255), -1)
            out_fake.write(frame)
        out_fake.release()
        
    logger.warning("[DUMMY DATA — TEST ONLY] Dummy video dataset generation complete.")


def subset_dataset_deterministically(dataset, target_size: int):
    """Subsets the dataset deterministically grouping by source file to minimize I/O while preserving class ratio."""
    by_file = {}
    for s in dataset.samples:
        by_file.setdefault(s["file_path"], []).append(s)
        
    files_by_label = {0: [], 1: []}
    for file_path, samples in by_file.items():
        label = samples[0]["label"]
        files_by_label[label].append((file_path, samples))
        
    num_files_0 = len(files_by_label[0])
    num_files_1 = len(files_by_label[1])
    
    if num_files_0 == 0 or num_files_1 == 0:
        raise ValueError(f"Cannot subset dataset: class 0 files = {num_files_0}, class 1 files = {num_files_1}")
        
    total_samples = len(dataset.samples)
    samples_0 = sum(len(s) for _, s in files_by_label[0])
    ratio_0 = samples_0 / total_samples
    target_samples_0 = max(1, int(target_size * ratio_0))
    target_samples_1 = max(1, target_size - target_samples_0)
    
    selected_samples = []
    
    accumulated_0 = 0
    files_by_label[0].sort(key=lambda x: x[0])
    for file_path, samples in files_by_label[0]:
        if accumulated_0 >= target_samples_0:
            break
        selected_samples.extend(samples)
        accumulated_0 += len(samples)
        
    accumulated_1 = 0
    files_by_label[1].sort(key=lambda x: x[0])
    for file_path, samples in files_by_label[1]:
        if accumulated_1 >= target_samples_1:
            break
        selected_samples.extend(samples)
        accumulated_1 += len(samples)
        
    dataset.samples = selected_samples
    rng = random.Random(42)
    rng.shuffle(dataset.samples)
    logger.info(f"Subset dataset deterministically to {len(dataset.samples)} samples (target: {target_size}).")


def run_promotion_gate(media_type: str, run_metrics: dict, run_weights_path: str, is_baseline_run: bool, skip_promotion_overwrite: bool = False):
    """Applies the production promotion gate by comparing validation metrics against baseline."""
    results_dir = os.path.join(PipelineConfig.RESULTS_DIR, media_type)
    os.makedirs(results_dir, exist_ok=True)
    baseline_path = os.path.join(results_dir, "baseline_metrics.json")
    
    val_metrics = run_metrics.get("final_metrics", {})
    real_pct = val_metrics.get("real_pred_pct", 0.0)
    fake_pct = val_metrics.get("fake_pred_pct", 0.0)
    class_collapse = val_metrics.get("class_collapse", False) or (real_pct > 0.98 or fake_pct > 0.98)
    
    logger.info(f"[PROMOTION GATE] Prediction class distribution - REAL: {real_pct:.2%}, FAKE: {fake_pct:.2%}")
    logger.info(f"[PROMOTION GATE] CLASS_COLLAPSE flag: {class_collapse}")
    
    if class_collapse:
        logger.error("[PROMOTION GATE] Promotion REJECTED: Model suffered from class collapse.")
        return False

    if is_baseline_run:
        logger.info("[PROMOTION GATE] Saving metrics as the baseline reference.")
        if not skip_promotion_overwrite:
            with open(baseline_path, "w") as f:
                json.dump(run_metrics, f, indent=4)
            
            prod_path = PipelineConfig.VIDEO_MODEL_PATH if media_type == "video" else PipelineConfig.IMAGE_MODEL_PATH
            os.makedirs(os.path.dirname(prod_path), exist_ok=True)
            shutil.copy2(run_weights_path, prod_path)
            logger.info(f"[PROMOTION GATE] Copied baseline weights to production path: {prod_path}")
            update_model_registry(media_type, run_metrics, prod_path, is_baseline=True)
        else:
            logger.info("[PROMOTION GATE] skip_promotion_overwrite is True. Skipping writing to production paths.")
        return True
        
    if not os.path.exists(baseline_path):
        logger.warning(f"[PROMOTION GATE] Baseline metrics file not found at {baseline_path}. Promoting current model as initial reference.")
        if not skip_promotion_overwrite:
            prod_path = PipelineConfig.VIDEO_MODEL_PATH if media_type == "video" else PipelineConfig.IMAGE_MODEL_PATH
            os.makedirs(os.path.dirname(prod_path), exist_ok=True)
            shutil.copy2(run_weights_path, prod_path)
            update_model_registry(media_type, run_metrics, prod_path, is_baseline=False)
        else:
            logger.info("[PROMOTION GATE] skip_promotion_overwrite is True. Skipping writing to production paths.")
        return True
        
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
        
    base_macro_f1 = baseline["final_metrics"]["macro_f1"]
    base_real_recall = baseline["final_metrics"]["real_recall"]
    new_macro_f1 = val_metrics.get("macro_f1", 0.0)
    new_real_recall = val_metrics.get("real_recall", 0.0)
    
    logger.info(f"[PROMOTION GATE] Comparing against baseline: Macro-F1 ({new_macro_f1:.4f} vs {base_macro_f1:.4f}), REAL Recall ({new_real_recall:.4f} vs {base_real_recall:.4f})")
    passed_gate = (new_macro_f1 > base_macro_f1 and new_real_recall >= base_real_recall)
    
    if passed_gate:
        logger.info("[PROMOTION GATE] Promotion SUCCESSFUL! Model outperformed the baseline.")
        if not skip_promotion_overwrite:
            prod_path = PipelineConfig.VIDEO_MODEL_PATH if media_type == "video" else PipelineConfig.IMAGE_MODEL_PATH
            os.makedirs(os.path.dirname(prod_path), exist_ok=True)
            shutil.copy2(run_weights_path, prod_path)
            logger.info(f"[PROMOTION GATE] Copied new weights to production path: {prod_path}")
            update_model_registry(media_type, run_metrics, prod_path, is_baseline=False)
        return True
    else:
        logger.warning("[PROMOTION GATE] Promotion REJECTED: New model did not outperform baseline in Macro-F1 and REAL Recall.")
        return False


def update_model_registry(media_type: str, run_metrics: dict, prod_path: str, is_baseline: bool):
    """Updates model_registry.json with the details of the active model."""
    registry_path = PipelineConfig.REGISTRY_PATH
    registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
        except Exception:
            registry = {}
            
    registry[media_type] = {
        "model_path": prod_path,
        "strategy": run_metrics["hyperparameters"]["strategy"],
        "pretrained": run_metrics["hyperparameters"]["pretrained"],
        "optimized_threshold": run_metrics["hyperparameters"]["optimized_threshold"],
        "promoted_at": datetime.now().isoformat(),
        "is_baseline": is_baseline,
        "metrics": run_metrics["final_metrics"]
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=4)
    logger.info(f"Model registry updated successfully at {registry_path}")


# ==============================================================================
# 5. Core Video Model Training Pipeline
# ==============================================================================

def train(
    epochs: int = PipelineConfig.EPOCHS,
    batch_size: int = PipelineConfig.BATCH_SIZE,
    lr: float = PipelineConfig.LEARNING_RATE,
    generate_dummy: bool = False,
    is_pilot: bool = False,
    strategy: str = None,
    pretrained: bool = False,
    preloaded_datasets: tuple = None,
    temporal_pooling: str = None,
    max_frames: int = None,
    output_dir: str = None,
    skip_promotion_overwrite: bool = True,  # Default True to preserve verified production weights
    load_checkpoint: str = None,
    lock_test_set: bool = False,
    evaluation_threshold: float = None,
    unfreeze_layer3: bool = False,
    num_workers: int = 0,
    no_amp: bool = False,
    allow_full_training: bool = False
):
    """
    Main training execution function for SatyaDrishti VideoModel.
    - Preserves ResNet18 spatial feature extractor and planned LSTM/pooling architecture.
    - Uniformly samples 8 frames per video.
    - Executes conservative batch size (default 4) for GTX 1650 4 GB GPU.
    - Thread-constrained CPU execution (num_workers=0) to prevent RAM pressure.
    - Automatic mixed precision (AMP) with FP32 fallback for remainder batches.
    - Resumable, atomic checkpoints preserved separately from image model weights.
    - Full evaluation: accuracy, precision, recall, F1, macro-F1, ROC-AUC, confusion matrix, per-class metrics.
    """
    # Safety guard: prevent accidental full training run before image model evaluation is complete
    if not is_pilot and not generate_dummy and not allow_full_training:
        msg = (
            "\n" + "=" * 80 + "\n"
            "SAFETY GUARD: FULL VIDEO TRAINING ON 9,925 VIDEOS IS NOT AUTHORIZED YET.\n"
            "Per project guidelines, full video training is held until image model training\n"
            "and evaluation are complete and explicitly approved by the user.\n\n"
            "To run a controlled pilot test on 300 train / 100 val videos, pass: --pilot\n"
            "To run a dummy test pass: --generate-dummy\n"
            "To confirm full dataset training when approved, pass: --allow-full-training\n"
            "=" * 80
        )
        logger.warning(msg)
        print(msg)
        return

    # Determinism
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Apply configuration overrides
    num_frames = max_frames if max_frames is not None else PipelineConfig.MAX_FRAMES_PER_VIDEO
    PipelineConfig.MAX_FRAMES_PER_VIDEO = num_frames
    logger.info(f"Target sampled frames per video: {PipelineConfig.MAX_FRAMES_PER_VIDEO}")

    if temporal_pooling is not None:
        PipelineConfig.TEMPORAL_POOLING = temporal_pooling
        logger.info(f"Temporal pooling strategy: {temporal_pooling}")

    logger.info(PipelineConfig.get_summary())
    
    is_dummy_run = generate_dummy
    
    # 1. Dataset Setup
    if preloaded_datasets is not None:
        train_dataset, val_dataset, test_dataset = preloaded_datasets
        is_dummy_run = False
        log_prefix = "[PILOT TRAINING] " if is_pilot else "[REAL TRAINING] "
    elif is_dummy_run:
        logger.warning("[DUMMY DATA — TEST ONLY] Setting up dummy video dataset...")
        paths, labels = DatasetLoader.scan_video_dataset()
        if len(paths) == 0:
            generate_dummy_data(4)
            paths, labels = DatasetLoader.scan_video_dataset()
            
        dataset = VideoDataset(paths, labels, preprocessor=VideoPreprocessor(max_frames=num_frames))
        val_size = int(len(dataset) * PipelineConfig.VAL_SPLIT)
        train_size = len(dataset) - val_size
        if val_size == 0 and len(dataset) > 1:
            val_size = 1
            train_size = len(dataset) - 1
            
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        test_dataset = None
        log_prefix = "[DUMMY DATA — TEST ONLY] "
        logger.info(f"{log_prefix}Split dataset: {len(train_dataset)} train, {len(val_dataset)} val.")
    else:
        manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file '{manifest_path}' is missing. Please run generate_manifest.py first.")
            
        logger.info(f"Loading datasets from manifest: {manifest_path}")
        video_prep = VideoPreprocessor(max_frames=num_frames)
        train_dataset = DeepfakeDataset(split="train", media_type="video", manifest_path=manifest_path, preprocessor=video_prep, use_cache=False)
        val_dataset = DeepfakeDataset(split="val", media_type="video", manifest_path=manifest_path, preprocessor=video_prep, use_cache=False)
        test_dataset = None if lock_test_set else DeepfakeDataset(split="test", media_type="video", manifest_path=manifest_path, preprocessor=video_prep, use_cache=False)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No video samples found in manifest for training or validation.")
            
        # Pilot run subset
        if is_pilot:
            pilot_manifest_path = os.path.join(PipelineConfig.RESULTS_DIR, "video", "pilot_subset_manifest.csv")
            if os.path.exists(pilot_manifest_path):
                logger.info(f"Loading pre-saved pilot subset manifest from: {pilot_manifest_path}")
                import csv
                loaded_paths = {"train": set(), "val": set(), "test": set()}
                with open(pilot_manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("split") in loaded_paths:
                            loaded_paths[row["split"]].add(row["file_path"])
                
                train_dataset.samples = [s for s in train_dataset.samples if s["file_path"] in loaded_paths["train"]]
                val_dataset.samples = [s for s in val_dataset.samples if s["file_path"] in loaded_paths["val"]]
                if test_dataset is not None:
                    test_dataset.samples = [s for s in test_dataset.samples if s["file_path"] in loaded_paths["test"]]
            else:
                logger.info("Configuring PILOT RUN subset size (target: 300 train, 100 val, 100 test)...")
                subset_dataset_deterministically(train_dataset, 300)
                subset_dataset_deterministically(val_dataset, 100)
                if test_dataset is not None:
                    subset_dataset_deterministically(test_dataset, 100)
                    
        # Verify zero leakage across splits
        check_data_leakage(train_dataset, val_dataset, test_dataset)
        
        train_labels = [s["label"] for s in train_dataset.samples]
        unique_labels = set(train_labels)
        if len(unique_labels) < 2:
            raise ValueError(f"Training dataset must contain both classes (0 and 1), found: {list(unique_labels)}")
            
        log_prefix = "[PILOT TRAINING] " if is_pilot else "[REAL TRAINING] "
        test_len = len(test_dataset) if test_dataset is not None else 0
        logger.info(f"Loaded: {len(train_dataset)} train, {len(val_dataset)} val, {test_len} test samples.")

    # 2. Imbalance Handling
    active_strategy = strategy or PipelineConfig.IMBALANCE_STRATEGY
    logger.info(f"Imbalance Strategy: '{active_strategy}'")
    
    device = torch.device(PipelineConfig.DEVICE)
    model = VideoModel(num_classes=2, pretrained=pretrained, temporal_pooling=temporal_pooling).to(device)
    
    # Checkpoint paths setup (completely separate from image model weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        run_dir = output_dir
    else:
        run_dir = os.path.join(PipelineConfig.WEIGHTS_DIR, "runs", f"video_{active_strategy}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    latest_ckpt_path = os.path.join(run_dir, "latest_checkpoint.pth")
    best_ckpt_path = os.path.join(run_dir, "best_checkpoint.pth")
    model_best_path = os.path.join(run_dir, "model_best.pth")
    
    start_epoch = 0
    total_steps = 0
    best_val_loss = float('inf')
    best_val_metrics = {}
    best_opt_thresh = 0.5
    
    # Load from existing checkpoint if requested
    resume_target = load_checkpoint if load_checkpoint is not None else (latest_ckpt_path if os.path.exists(latest_ckpt_path) else None)
    if resume_target is not None and os.path.exists(resume_target):
        logger.info(f"Loading checkpoint from: {resume_target}")
        ckpt = torch.load(resume_target, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            total_steps = ckpt.get("step", 0)
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            best_opt_thresh = ckpt.get("best_opt_thresh", 0.5)
            if "rng_state" in ckpt:
                restore_rng_states(ckpt["rng_state"])
            logger.info(f"Successfully resumed from checkpoint at epoch {start_epoch}, step {total_steps}, best_val_loss: {best_val_loss:.4f}")
        else:
            model.load_state_dict(ckpt)
            logger.info("Loaded pure state_dict into VideoModel.")

    # Class weights calculation
    if not is_dummy_run:
        train_lbls = [s["label"] for s in train_dataset.samples]
        class_counts = [train_lbls.count(0), train_lbls.count(1)]
        total = len(train_lbls)
        w_real = total / (2.0 * max(1, class_counts[0]))
        w_fake = total / (2.0 * max(1, class_counts[1]))
        class_weights = [w_real, w_fake]
        logger.info(
            f"Training split class counts: REAL (0) = {class_counts[0]}, FAKE (1) = {class_counts[1]} (ratio 1:{class_counts[1]/max(1, class_counts[0]):.2f}). "
            f"Computed weights (training split only): REAL={w_real:.4f}, FAKE={w_fake:.4f}."
        )
    else:
        class_weights = [1.0, 1.0]
        
    sampler = None
    criterion_weight = None
    
    if active_strategy == "weighted_loss":
        criterion_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.info(f"Strategy: weighted_loss. Configured CrossEntropyLoss with training-only weights: REAL={w_real:.4f}, FAKE={w_fake:.4f}")
    elif active_strategy == "sampler":
        sample_weights = [class_weights[s["label"]] for s in train_dataset.samples] if not is_dummy_run else [1.0] * len(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    elif active_strategy == "both":
        criterion_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        sample_weights = [class_weights[s["label"]] for s in train_dataset.samples] if not is_dummy_run else [1.0] * len(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    
    # 3. Optimizer Setup
    if pretrained:
        if unfreeze_layer3:
            logger.info("Pretrained backbone: Layers 1-2 frozen; Layer 3, Layer 4, and temporal head trainable.")
            for name, param in model.backbone.named_parameters():
                if "layer4" not in name and "layer3" not in name:
                    param.requires_grad = False
        else:
            logger.info("Pretrained backbone: Layers 1-3 frozen; Layer 4 and temporal head trainable.")
            for name, param in model.backbone.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False
                
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_modules = [model.fc]
        if hasattr(model, "lstm"):
            head_modules.append(model.lstm)
        head_params = []
        for hm in head_modules:
            head_params.extend([p for p in hm.parameters() if p.requires_grad])
            
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": PipelineConfig.BACKBONE_LR},
            {"params": head_params, "lr": PipelineConfig.HEAD_LR}
        ], weight_decay=PipelineConfig.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=PipelineConfig.WEIGHT_DECAY)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. Mixed Precision (AMP) & Remainder FP32 Fallback Setup
    use_amp = (device.type == "cuda") and not no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    logger.info(f"Automatic Mixed Precision (AMP): {'ENABLED (with safe FP32 remainder fallback)' if use_amp else 'DISABLED'}")
    
    # 5. DataLoaders Setup (conservative workers=0 initially)
    collate = safe_collate_fn if not is_dummy_run else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(batch_size, len(train_dataset)), 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(batch_size, len(val_dataset)), 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate
    )
    
    patience = PipelineConfig.PATIENCE
    epochs_no_improve = 0
    total_batches = len(train_loader)
    
    logger.info(f"{log_prefix}Starting video model training on device: {device} (Epochs: {epochs}, Batch Size: {batch_size}, Frames: {PipelineConfig.MAX_FRAMES_PER_VIDEO})...")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None or batch[0] is None or batch[1] is None:
                continue
                
            videos, batch_labels = batch
            batch_actual_size = videos.size(0)
            total_steps += 1
            
            # Guard: check finite inputs
            check_tensor_finite(videos, "1_input_videos", "videos", epoch + 1, batch_idx + 1, current_lr)
            check_labels_finite(batch_labels, "2_target_labels", "batch_labels", epoch + 1, batch_idx + 1, current_lr)
            
            videos = videos.to(device)
            batch_labels = batch_labels.to(device)
            
            # Automatic FP32 fallback for remainder batches:
            # Enable AMP only when batch_actual_size == batch_size.
            # Remainder batches execute in FP32 to prevent non-standard cuDNN FP16 overflow.
            batch_use_amp = use_amp and (batch_actual_size == batch_size)
            if use_amp and not batch_use_amp:
                logger.info(
                    f"Batch [{batch_idx+1}/{total_batches}]: Remainder batch ({batch_actual_size} != {batch_size}). "
                    f"Disabling AMP for this batch -> executing safe FP32 forward/backward pass."
                )
                
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=batch_use_amp):
                outputs = model(videos)
                check_tensor_finite(outputs, "3_model_logits", "outputs", epoch + 1, batch_idx + 1, current_lr)
                loss = criterion(outputs, batch_labels)
                check_tensor_finite(loss, "4_loss", "cross_entropy_loss", epoch + 1, batch_idx + 1, current_lr)
                
            if batch_use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
                
            # Gradients stability check
            for p_name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    check_tensor_finite(param.grad, "5_gradients", f"grad_{p_name}", epoch + 1, batch_idx + 1, current_lr)
                    
            # Gradient norm clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_tensor_finite(grad_norm, "6_grad_norm", "clipped_grad_norm", epoch + 1, batch_idx + 1, current_lr)
            
            # Step optimizer safely
            if batch_use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Check model parameters finite
            for p_name, param in model.named_parameters():
                check_tensor_finite(param.data, "7_model_parameter", p_name, epoch + 1, batch_idx + 1, current_lr)
                
            running_loss += loss.item() * batch_actual_size
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = (correct_train / total_train) * 100 if total_train > 0 else 0.0
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        total_val = 0
        val_labels_all = []
        val_probs_all = []
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None or batch[0] is None or batch[1] is None:
                    continue
                videos, batch_labels = batch
                videos = videos.to(device)
                batch_labels = batch_labels.to(device)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(videos)
                    loss = criterion(outputs, batch_labels)
                    
                val_loss += loss.item() * videos.size(0)
                total_val += batch_labels.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                fake_probs = probs[:, 1]
                
                val_labels_all.extend(batch_labels.cpu().numpy())
                val_probs_all.extend(fake_probs.cpu().numpy())
                
        epoch_val_loss = val_loss / max(1, total_val)
        
        val_labels_arr = np.array(val_labels_all)
        val_probs_arr = np.array(val_probs_all)
        
        opt_thresh, best_val_macro_f1 = optimize_threshold(val_labels_arr, val_probs_arr)
        val_metrics_suite = compute_all_metrics(val_labels_arr, val_probs_arr, threshold=opt_thresh)
        
        epoch_duration = time.time() - epoch_start_time
        host_mem_mb = get_process_memory_mb()
        cuda_mem = get_cuda_memory_mb(device)
        
        logger.info(
            f"{log_prefix}Epoch [{epoch+1}/{epochs}] ({epoch_duration:.1f}s) - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_metrics_suite['accuracy']:.2f}% | "
            f"Host RAM: {host_mem_mb:.1f}MB, VRAM: {cuda_mem['allocated']:.1f}MB"
        )
        logger.info(
            f"{log_prefix}Val Metrics (Thresh={opt_thresh:.2f}) - "
            f"Macro-F1: {best_val_macro_f1:.4f}, ROC-AUC: {val_metrics_suite['roc_auc']:.4f}, "
            f"REAL P/R: {val_metrics_suite['real_precision']:.4f}/{val_metrics_suite['real_recall']:.4f}, "
            f"FAKE P/R: {val_metrics_suite['fake_precision']:.4f}/{val_metrics_suite['fake_recall']:.4f}, "
            f"ECE: {val_metrics_suite['ece']:.4f}"
        )
        logger.info(f"{log_prefix}Val Confusion Matrix: {val_metrics_suite['confusion_matrix']}")
        
        # Determine if this epoch is the new best model before constructing checkpoint state
        is_best = (epoch_val_loss < best_val_loss)
        if is_best:
            best_val_loss = epoch_val_loss
            best_opt_thresh = opt_thresh
            best_val_metrics = val_metrics_suite.copy()
            best_val_metrics["train_loss"] = epoch_train_loss
            best_val_metrics["train_accuracy"] = epoch_train_acc
            best_val_metrics["val_loss"] = epoch_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save atomic checkpoint state with finite best_val_loss guaranteed
        checkpoint_state = {
            "epoch": epoch + 1,
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if use_amp and scaler else None,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "best_val_loss": best_val_loss,
            "best_opt_thresh": best_opt_thresh,
            "metrics": val_metrics_suite,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "hyperparameters": {
                "batch_size": batch_size,
                "lr": lr,
                "strategy": active_strategy,
                "pretrained": pretrained,
                "temporal_pooling": PipelineConfig.TEMPORAL_POOLING,
                "max_frames_per_video": PipelineConfig.MAX_FRAMES_PER_VIDEO
            }
        }
        
        # Save latest checkpoint atomically
        save_checkpoint_atomic(checkpoint_state, latest_ckpt_path)
        
        # Save best checkpoint atomically
        if is_best:
            saved_best = save_checkpoint_atomic(checkpoint_state, best_ckpt_path)
            if saved_best:
                # Save lightweight weights-only checkpoint
                torch.save(model.state_dict(), model_best_path)
                logger.info(f"{log_prefix}New best model saved to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
            else:
                logger.error(f"{log_prefix}Failed to save best model checkpoint to {best_ckpt_path} due to validation failure!")
            
        scheduler.step(epoch_val_loss)
        
        # Early Stopping
        if epochs_no_improve >= patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    # Load best checkpoint for final evaluation
    if os.path.exists(model_best_path):
        model.load_state_dict(torch.load(model_best_path, map_location=device))
    elif os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        
    # Evaluate on test split if available
    test_metrics = {}
    if (not is_dummy_run) and (not lock_test_set) and (test_dataset is not None) and len(test_dataset) > 0:
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False, num_workers=num_workers, collate_fn=collate)
        test_labels_all = []
        test_probs_all = []
        
        with torch.no_grad():
            for batch in test_loader:
                if batch is None or batch[0] is None or batch[1] is None:
                    continue
                videos, batch_labels = batch
                videos = videos.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(videos)
                probs = torch.softmax(outputs, dim=1)
                test_labels_all.extend(batch_labels.cpu().numpy())
                test_probs_all.extend(probs[:, 1].cpu().numpy())
                
        test_labels_arr = np.array(test_labels_all)
        test_probs_arr = np.array(test_probs_all)
        test_metrics = compute_all_metrics(test_labels_arr, test_probs_arr, threshold=best_opt_thresh)
        logger.info(
            f"TEST METRICS (Optimized Threshold={best_opt_thresh:.2f}) - "
            f"Acc: {test_metrics['accuracy']:.2f}%, Macro-F1: {test_metrics['macro_f1']:.4f}, "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f}, ECE: {test_metrics['ece']:.4f} | "
            f"REAL P/R: {test_metrics['real_precision']:.4f}/{test_metrics['real_recall']:.4f}, "
            f"FAKE P/R: {test_metrics['fake_precision']:.4f}/{test_metrics['fake_recall']:.4f}"
        )
        logger.info(f"Test Confusion Matrix: {test_metrics['confusion_matrix']}")
    else:
        test_metrics = best_val_metrics

    # Save complete run metrics
    metrics_report = {
        "dataset_type": "pilot" if is_pilot else ("dummy" if is_dummy_run else "real"),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "strategy": active_strategy,
            "pretrained": pretrained,
            "optimized_threshold": best_opt_thresh,
            "temporal_pooling": PipelineConfig.TEMPORAL_POOLING,
            "max_frames_per_video": PipelineConfig.MAX_FRAMES_PER_VIDEO,
            "amp_enabled": use_amp
        },
        "final_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoints": {
            "run_dir": run_dir,
            "latest_checkpoint": latest_ckpt_path,
            "best_checkpoint": best_ckpt_path,
            "model_best": model_best_path
        }
    }
    
    report_filename = f"video_{active_strategy}_{timestamp}_metrics.json"
    report_path = os.path.join(run_dir, report_filename)
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
    logger.info(f"Saved run metrics report to: {report_path}")
    
    # Production promotion gate
    is_baseline_run = (active_strategy == "standard" and not pretrained)
    run_promotion_gate(
        media_type="video", 
        run_metrics=metrics_report, 
        run_weights_path=model_best_path, 
        is_baseline_run=is_baseline_run, 
        skip_promotion_overwrite=skip_promotion_overwrite
    )
    
    logger.info(f"Video model training completed. Checkpoints and reports preserved in: {run_dir}")
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatyaDrishti Video Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=PipelineConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=PipelineConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate small dummy dataset if empty")
    parser.add_argument("--pilot", action="store_true", help="Perform a controlled pilot training run on a subset of real data")
    parser.add_argument("--strategy", type=str, choices=["standard", "weighted_loss", "sampler", "both"], default=None, help="Imbalance handling strategy")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for ResNet18 spatial backbone")
    parser.add_argument("--temporal-pooling", type=str, choices=["lstm", "mean", "max"], default=None, help="Temporal aggregation strategy")
    parser.add_argument("--max-frames", type=int, default=None, help="Max sampled frames per video (default 8)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory to save weights and metrics")
    parser.add_argument("--skip-promotion-overwrite", action="store_true", default=True, help="Skip overwriting the production model and registry")
    parser.add_argument("--lock-test-set", action="store_true", help="Do not load or evaluate the test set")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to load before training")
    parser.add_argument("--unfreeze-layer3", action="store_true", help="Unfreeze layer 3 of backbone")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes (default 0)")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--allow-full-training", action="store_true", help="Explicit confirmation flag to allow full dataset training")
    
    args = parser.parse_args()
    
    epochs_to_run = args.epochs
    if args.pilot and args.epochs == PipelineConfig.EPOCHS:
        epochs_to_run = 2
        
    train(
        epochs=epochs_to_run, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        generate_dummy=args.generate_dummy,
        is_pilot=args.pilot,
        strategy=args.strategy,
        pretrained=args.pretrained,
        temporal_pooling=args.temporal_pooling,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        skip_promotion_overwrite=args.skip_promotion_overwrite,
        lock_test_set=args.lock_test_set,
        load_checkpoint=args.load_checkpoint,
        unfreeze_layer3=args.unfreeze_layer3,
        num_workers=args.num_workers,
        no_amp=args.no_amp,
        allow_full_training=args.allow_full_training
    )
