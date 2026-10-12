"""
Experiment C-Generalization: Domain-Generalized Forensic Training
----------------------------------------------------------------
Phase 2 Pilot Runner

Key Requirements:
1. Fresh ResNet18 ImageNet pretrained initialization (pretrained=True).
   Zero warm-start from Epoch 12 or previous checkpoints.
2. Layer 1-2 frozen (conv1, bn1, layer1, layer2).
   Layer 3-4 and FC classification head trainable.
3. Backbone LR = 1e-5, Head LR = 1e-4, Adam optimizer, weight_decay = 1e-4.
4. Inverse-frequency class weights computed from active training split ([3.1212, 0.5954]).
5. Label smoothing = 0.05 on CrossEntropyLoss.
6. Anti-shortcut domain augmentations active during training (JPEG Q=65-95, noise, blur, color jitter).
7. FP32 standard precision.
8. Isolated output directory: backend/models/weights/runs/exp_c_generalization/
9. Strictly DOES NOT modify production model, checkpoints, manifests, or thresholds.
10. Dual evaluation: In-domain validation metrics + OOD control benchmark before and after pilot.
"""

import os

# Conservative numerical thread configuration
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import sys
import time
import json
import ctypes
import argparse
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

cv2.ocl.setUseOpenCL(False)
if hasattr(cv2, "setNumThreads"):
    cv2.setNumThreads(1)

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from backend.models.image_model import ImageModel
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.core.decision_system import SafetyDecisionSystem, DecisionState
from backend.agents.image_agent import ImageAgent
from backend.agents.metadata_agent import MetadataAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.artifact_agent import ArtifactConsistencyAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.context import AnalysisContext
from backend.utils.metrics_helper import (
    optimize_threshold,
    compute_all_metrics,
    compute_ece,
)

logger = setup_logger("ExpC-Generalization")


# Definition of Corrected OOD Control Benchmark (5 canonical items)
OOD_CONTROL_SAMPLES = [
    {"name": "profile.jpeg", "rel_path": "backend/uploads/a88b3aae-5f65-44bb-b1d4-c95f6ac47a8f_profile.jpeg", "label": 0, "type": "real portrait 1"},
    {"name": "she.jpeg", "rel_path": "backend/uploads/404e3990-dbc9-4ec5-8abe-159788170bfa_she.jpeg", "label": 0, "type": "real portrait 2"},
    {"name": "picture.jpeg", "rel_path": "backend/uploads/8a30af82-839c-4be1-afc7-2ce2c58d33b1_picture.jpeg", "label": 0, "type": "real portrait 3"},
    {"name": "deepy.jpeg", "rel_path": "backend/uploads/fd9b9f7d-62fe-4923-9eaf-d0961237fc1e_deepy.jpeg", "label": 1, "type": "in-the-wild deepfake"},
    {"name": "professional.jpeg", "rel_path": "backend/uploads/ffe1b29d-96de-45de-a2e0-e6644ca4b038_professional.jpeg", "label": 1, "type": "AI-generated synthetic face"},
]


def get_ram_usage_mb():
    """Returns (used_ram_mb, total_ram_mb) using Windows API."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_mb = stat.ullTotalPhys / (1024 * 1024)
        avail_mb = stat.ullAvailPhys / (1024 * 1024)
        return round(total_mb - avail_mb, 1), round(total_mb, 1)
    except Exception:
        return 0.0, 0.0


def get_vram_usage_mb():
    """Returns (allocated_vram_mb, max_allocated_vram_mb)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return round(allocated, 1), round(max_allocated, 1)
    return 0.0, 0.0


def save_checkpoint_atomic(state: dict, target_path: str):
    """Saves checkpoint safely using atomic rename to prevent partial/corrupted writes."""
    temp_path = f"{target_path}.tmp_{os.getpid()}_{int(time.time())}"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.save(state, temp_path)
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except OSError:
            pass
    os.replace(temp_path, target_path)


def compute_manifest_class_weights(dataset: DeepfakeDataset) -> torch.Tensor:
    """Computes exact inverse-frequency class weights from active training split."""
    real_count = sum(1 for s in dataset.samples if s["label"] == 0)
    fake_count = sum(1 for s in dataset.samples if s["label"] == 1)
    total = len(dataset.samples)

    if real_count == 0 or fake_count == 0:
        raise ValueError(f"Training split must contain both classes! Found REAL={real_count}, FAKE={fake_count}")

    w_real = total / (2.0 * real_count)
    w_fake = total / (2.0 * fake_count)
    return torch.tensor([w_real, w_fake], dtype=torch.float32)


def evaluate_ood_control_benchmark(model: nn.Module, device: torch.device) -> dict:
    """
    Evaluates model on the Out-of-Domain (OOD) control benchmark.
    Evaluates:
    - Raw model P(REAL), P(FAKE), and predicted class
    - Multi-agent final fusion verdict (LIKELY REAL, LIKELY FAKE, UNCERTAIN, NOT SUITABLE)
    - Forensic agreement/conflict analysis
    """
    model = model.to(device)
    model.eval()
    preprocessor = ImagePreprocessor()
    quality_gate = ImageQualityGate()
    forensics = ForensicEvidenceSuite()
    sds = SafetyDecisionSystem()
    meta_agent = MetadataAgent()
    conf_agent = ConfidenceEvaluationAgent()
    artifact_agent = ArtifactConsistencyAgent()
    fusion_agent = FusionAgent()
    
    results = []
    real_fakes_count = 0
    real_total = 0
    fake_correct_count = 0
    fake_total = 0
    real_probs = []
    fake_probs = []

    with torch.no_grad():
        for item in OOD_CONTROL_SAMPLES:
            full_path = os.path.join(_repo_root, item["rel_path"])
            if not os.path.exists(full_path):
                logger.warning(f"OOD sample not found: {full_path}")
                continue

            try:
                # 1. Raw model forward pass
                tensor = preprocessor.preprocess_image(full_path, is_training=False).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                p_real = float(probs[0].item())
                p_fake = float(probs[1].item())
                pred_cls = 1 if p_fake >= 0.50 else 0
                pred_class_str = "FAKE" if pred_cls == 1 else "REAL"

                # 2. Quality gate & genuine forensics
                img_bgr = cv2.imread(full_path)
                quality_res = quality_gate.evaluate(img_bgr, require_face=True)
                face_bbox = quality_res.primary_face_bbox if quality_res.face_detected else None
                forensic_res = forensics.evaluate(img_bgr, face_bbox=face_bbox) if img_bgr is not None else {}

                # 3. Decision system (uses p_fake)
                decision = sds.decide(
                    raw_fake_prob=p_fake,
                    quality_result=quality_res,
                    forensic_result=forensic_res,
                )

                # 4. Multi-agent fusion pass
                filename = os.path.basename(full_path)
                size = os.path.getsize(full_path)
                _, ext = os.path.splitext(filename)
                fmt = ext[1:].upper() if ext else "PNG"

                img_agent = ImageAgent()
                img_agent_res = img_agent._format_response(
                    decision=decision,
                    quality_res=quality_res,
                    forensic_res=forensic_res,
                    filename=filename,
                    size=size,
                    fmt=fmt,
                    processing_time=0.01,
                    img_shape=img_bgr.shape if img_bgr is not None else (0, 0, 0)
                )

                meta_res = meta_agent.analyze(full_path)
                conf_res = conf_agent.evaluate(img_agent_res)

                conf_score_val = img_agent_res.get("confidence_score")
                primary_conf_ratio = float(conf_score_val if conf_score_val is not None else 50.0) / 100.0
                primary_verdict = img_agent_res.get("verdict", "REAL")
                metadata_anomalies = meta_res.get("anomalies_detected", False)
                has_conflict = (primary_verdict == "DEEPFAKE" and not metadata_anomalies) or (primary_verdict == "REAL" and metadata_anomalies)
                should_trigger_artifact = (primary_conf_ratio < PipelineConfig.AGENTIC_CONFIDENCE_THRESHOLD) or has_conflict

                artifact_res = None
                if should_trigger_artifact:
                    artifact_res = artifact_agent.analyze(full_path, media_type="image")

                # analysis_id="" ensures zero DB writes
                context = AnalysisContext(analysis_id="", media_type="image")
                context.uploaded_file = filename
                context.validation_result = {"valid": True, "media_type": "image"}
                context.image_agent_result = img_agent_res
                context.metadata_agent_result = meta_res
                context.confidence_result = conf_res
                context.artifact_agent_result = artifact_res

                fusion_res = fusion_agent.fuse(context)
                fusion_verdict = fusion_res.get("verdict", "UNCERTAIN")

                # 5. Forensic agreement / conflict analysis
                suspicious_signals = forensic_res.get("suspicious_signal_count", 0)
                evidence_items = forensic_res.get("evidence_items", [])
                has_forensic_anomaly = (suspicious_signals > 0)

                if pred_cls == 1:
                    if has_forensic_anomaly:
                        forensic_relation = "AGREES (Both model and forensics detect manipulation)"
                    else:
                        forensic_relation = "CONFLICTS (Model predicts FAKE, but forensics show clean signals)"
                else:
                    if has_forensic_anomaly:
                        forensic_relation = f"CONFLICTS (Model predicts REAL, but forensics flagged {suspicious_signals} anomaly signal(s): {', '.join(evidence_items[:2])})"
                    else:
                        forensic_relation = "AGREES (Both model and forensics indicate genuine media)"

            except Exception as e:
                logger.warning(f"Error evaluating OOD sample {item['name']}: {e}")
                continue

            is_real = (item["label"] == 0)
            if is_real:
                real_total += 1
                real_probs.append(p_fake)
                if pred_cls == 1:
                    real_fakes_count += 1
            else:
                fake_total += 1
                fake_probs.append(p_fake)
                if pred_cls == 1:
                    fake_correct_count += 1

            results.append({
                "name": item["name"],
                "type": item["type"],
                "ground_truth": "REAL" if is_real else "FAKE",
                "pred_class": pred_class_str,
                "real_prob": round(p_real, 4),
                "fake_prob": round(p_fake, 4),
                "final_fusion_verdict": fusion_verdict,
                "forensic_relation": forensic_relation,
                "forensic_evidence": evidence_items,
                "is_correct": (pred_cls == item["label"])
            })

    real_fpr = (real_fakes_count / max(1, real_total)) * 100.0 if real_total > 0 else 0.0
    fake_detection_rate = (fake_correct_count / max(1, fake_total)) * 100.0 if fake_total > 0 else 0.0
    mean_real_fake_prob = float(np.mean(real_probs)) if real_probs else 0.0
    mean_fake_fake_prob = float(np.mean(fake_probs)) if fake_probs else 0.0

    return {
        "samples_evaluated": len(results),
        "real_count": real_total,
        "fake_count": fake_total,
        "real_false_positive_rate_pct": round(real_fpr, 2),
        "fake_detection_rate_pct": round(fake_detection_rate, 2),
        "mean_real_fake_prob": round(mean_real_fake_prob, 4),
        "mean_fake_fake_prob": round(mean_fake_fake_prob, 4),
        "class_collapse_detected": bool(real_fpr >= 85.0 and fake_detection_rate >= 85.0) or (real_fpr == 0.0 and fake_detection_rate == 0.0),
        "sample_details": results
    }


def check_promotion_gates(val_metrics: dict, ood_results: dict) -> dict:
    """
    Checks if a candidate checkpoint satisfies the strict dual-domain promotion gates (Section 7).
    IN-DOMAIN:
    - Macro-F1 >= 0.88
    - REAL F1 >= 0.82
    - ROC-AUC >= 0.80
    - ECE <= 0.15

    OOD:
    - REAL false-positive rate <= 15%
    - FAKE detection rate >= 80%
    - deepy.jpeg P(FAKE) >= 0.65
    - professional.jpeg P(FAKE) >= 0.65
    - profile.jpeg P(FAKE) <= 0.30
    - no >90% single-class collapse
    """
    sample_probs = {s["name"]: s["fake_prob"] for s in ood_results.get("sample_details", [])}
    deepy_p_fake = sample_probs.get("deepy.jpeg", 0.0)
    professional_p_fake = sample_probs.get("professional.jpeg", 0.0)
    profile_p_fake = sample_probs.get("profile.jpeg", 1.0)

    macro_f1 = float(val_metrics.get("macro_f1", 0.0))
    real_f1 = float(val_metrics.get("real_f1", 0.0))
    roc_auc = float(val_metrics.get("roc_auc", 0.0))
    ece = float(val_metrics.get("ece", 1.0))

    real_fpr = float(ood_results.get("real_false_positive_rate_pct", 100.0))
    fake_det = float(ood_results.get("fake_detection_rate_pct", 0.0))
    class_collapse = bool(ood_results.get("class_collapse_detected", False))

    gates = {
        "in_domain_macro_f1_gte_0_88": bool(macro_f1 >= 0.88),
        "in_domain_real_f1_gte_0_82": bool(real_f1 >= 0.82),
        "in_domain_roc_auc_gte_0_80": bool(roc_auc >= 0.80),
        "in_domain_ece_lte_0_15": bool(ece <= 0.15),
        "ood_real_fpr_lte_15pct": bool(real_fpr <= 15.0),
        "ood_fake_det_gte_80pct": bool(fake_det >= 80.0),
        "deepy_p_fake_gte_0_65": bool(deepy_p_fake >= 0.65),
        "professional_p_fake_gte_0_65": bool(professional_p_fake >= 0.65),
        "profile_p_fake_lte_0_30": bool(profile_p_fake <= 0.30),
        "no_class_collapse": bool(not class_collapse)
    }
    all_passed = all(gates.values())
    return {
        "all_passed": all_passed,
        "gates": gates,
        "deepy_p_fake": round(deepy_p_fake, 4),
        "professional_p_fake": round(professional_p_fake, 4),
        "profile_p_fake": round(profile_p_fake, 4)
    }


def compute_generalization_score(val_metrics: dict, ood_results: dict) -> float:
    """
    Computes a composite generalization score balancing in-domain Macro-F1
    with OOD accuracy and probability separation (Section 6):
    Separation = Mean P(FAKE) on FAKES - Mean P(FAKE) on REALS.
    Score = 0.40 * Macro-F1 + 0.30 * OOD Accuracy + 0.30 * Normalized Separation.
    """
    macro_f1 = float(val_metrics.get("macro_f1", 0.0))
    details = ood_results.get("sample_details", [])
    correct = sum(1 for s in details if s.get("is_correct", False))
    ood_acc = (correct / max(1, len(details))) if details else 0.0
    mean_real_fake_prob = float(ood_results.get("mean_real_fake_prob", 0.5))
    mean_fake_fake_prob = float(ood_results.get("mean_fake_fake_prob", 0.5))
    separation = mean_fake_fake_prob - mean_real_fake_prob
    norm_sep = float(np.clip((separation + 1.0) / 2.0, 0.0, 1.0))
    return float(round(0.40 * macro_f1 + 0.30 * ood_acc + 0.30 * norm_sep, 4))


def evaluate_three_way_comparison(
    exp_c_best_path: str,
    device: torch.device,
    val_loader: DataLoader,
    max_val_batches: int = 50
) -> dict:
    """
    Evaluates and compares:
    1. Production Model (backend/models/weights/image_model.pth)
    2. Epoch 12 (backend/models/weights/runs/full_image_I1/latest_checkpoint.pth)
    3. Experiment C Best Checkpoint (exp_c_best_path)

    Compares In-Domain (Accuracy, Macro-F1, REAL F1, FAKE F1, ROC-AUC, ECE)
    and OOD (profile.jpeg, she.jpeg, picture.jpeg, deepy.jpeg, professional.jpeg).
    """
    prod_path = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "image_model.pth")
    ep12_path = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", "full_image_I1", "latest_checkpoint.pth")

    models_to_test = [
        {"name": "Production Model", "path": prod_path, "type": "weights_file"},
        {"name": "Epoch 12", "path": ep12_path, "type": "checkpoint_dict"},
        {"name": "Experiment C Best Checkpoint", "path": exp_c_best_path, "type": "checkpoint_dict"}
    ]

    criterion = nn.CrossEntropyLoss()
    comparison_results = {}

    logger.info("\n" + "=" * 80)
    logger.info("   COMMENCING THREE-WAY MODEL EVALUATION BENCHMARK (Section 11)")
    logger.info("=" * 80)

    for m_info in models_to_test:
        name = m_info["name"]
        path = m_info["path"]
        logger.info(f"\nEvaluating: {name} (Source: {path})")
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}. Skipping {name}.")
            continue

        try:
            m = ImageModel(num_classes=2, pretrained=False).to(device)
            if m_info["type"] == "weights_file":
                m.load_weights(path)
            else:
                st = torch.load(path, map_location=device, weights_only=False)
                state_dict = st.get("model_state_dict", st)
                m.load_state_dict(state_dict)
            m = m.to(device)
            m.eval()

            # 1. In-domain validation
            val_loss_accum = 0.0
            val_targets_all = []
            val_probs_all = []

            with torch.no_grad():
                for v_idx, (images, targets) in enumerate(val_loader):
                    if max_val_batches is not None and v_idx >= max_val_batches:
                        break
                    if images is None or targets is None or len(images) == 0:
                        continue
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = m(images)
                    loss = criterion(outputs, targets)
                    val_loss_accum += loss.item() * len(targets)
                    probs = torch.softmax(outputs, dim=1)
                    val_targets_all.extend(targets.cpu().numpy())
                    val_probs_all.extend(probs[:, 1].cpu().numpy())

            val_targets_arr = np.array(val_targets_all)
            val_probs_arr = np.array(val_probs_all)
            v_metrics = compute_all_metrics(val_targets_arr, val_probs_arr, threshold=0.50)
            v_metrics["val_loss"] = float(val_loss_accum / max(1, len(val_targets_all)))

            # 2. OOD Evaluation
            ood_res = evaluate_ood_control_benchmark(m, device)

            comparison_results[name] = {
                "in_domain": {
                    "accuracy": round(float(v_metrics.get("accuracy", 0.0)), 2),
                    "macro_f1": round(float(v_metrics.get("macro_f1", 0.0)), 4),
                    "real_f1": round(float(v_metrics.get("real_f1", 0.0)), 4),
                    "fake_f1": round(float(v_metrics.get("fake_f1", 0.0)), 4),
                    "roc_auc": round(float(v_metrics.get("roc_auc", 0.0)), 4),
                    "ece": round(float(v_metrics.get("ece", 0.0)), 4),
                },
                "ood_summary": {
                    "real_fpr_pct": ood_res.get("real_false_positive_rate_pct"),
                    "fake_detection_pct": ood_res.get("fake_detection_rate_pct"),
                    "mean_real_fake_prob": ood_res.get("mean_real_fake_prob"),
                    "mean_fake_fake_prob": ood_res.get("mean_fake_fake_prob")
                },
                "ood_samples": {
                    s["name"]: {
                        "p_real": s["real_prob"],
                        "p_fake": s["fake_prob"],
                        "predicted_class": s["pred_class"],
                        "fusion_verdict": s["final_fusion_verdict"],
                        "forensic_relation": s["forensic_relation"]
                    } for s in ood_res.get("sample_details", [])
                }
            }

            logger.info(f"[{name}] In-Domain: Acc={v_metrics['accuracy']:.2f}%, Macro-F1={v_metrics['macro_f1']:.4f}, ROC-AUC={v_metrics['roc_auc']:.4f}, ECE={v_metrics['ece']:.4f}")
            logger.info(f"[{name}] OOD: Real FPR={ood_res['real_false_positive_rate_pct']}%, Fake Det={ood_res['fake_detection_rate_pct']}%")
            for s in ood_res.get("sample_details", []):
                logger.info(f"    {s['name']:<20}: P(REAL)={s['real_prob']:.4f}, P(FAKE)={s['fake_prob']:.4f} -> Pred={s['pred_class']}, Fusion={s['final_fusion_verdict']}")

        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    return comparison_results


def run_experiment_c_generalization(
    epochs: int = 8,
    batch_size: int = 32,
    num_workers: int = 2,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_train_batches: int = None,
    max_val_batches: int = None,
    is_pilot: bool = False
) -> dict:
    """
    Executes Experiment C-Generalization training and evaluation (Phase 2 Pilot or Phase 3 Full Training).
    """
    device = torch.device(device_str)
    mode_name = "PILOT VERIFICATION" if is_pilot else "FULL EXPERIMENT (Phase 3)"
    logger.info("=" * 80)
    logger.info(f"   STARTING EXPERIMENT C-GENERALIZATION ({mode_name})")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Max Train Batches: {max_train_batches}")
    logger.info(f"Max Val Batches: {max_val_batches}")
    logger.info(f"DataLoader Workers: {num_workers}")
    logger.info(f"Backbone LR: 1e-5, Head LR: 1e-4, Weight Decay: 1e-4")
    logger.info(f"Precision: FP32")
    logger.info(f"Domain Augmentations: ENABLED (USE_AUGMENTATION=True)")

    # 1. Output Directories & Progress Log
    exp_dir = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", "exp_c_generalization")
    os.makedirs(exp_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(exp_dir, "latest_checkpoint.pth")
    best_ckpt_path = os.path.join(exp_dir, "best_checkpoint.pth")
    
    log_name = "exp_c_generalization_pilot.log" if is_pilot else "exp_c_generalization_progress.log"
    progress_log_path = os.path.join(PipelineConfig.RESULTS_DIR, "image", log_name)
    os.makedirs(os.path.dirname(progress_log_path), exist_ok=True)

    logger.info(f"Isolated Checkpoints Directory: {exp_dir}")
    logger.info(f"Progress Log File: {progress_log_path}")

    # 2. Fresh Model Initialization (ImageNet-1K Pretrained, NOT Warm-Started from Epoch 12)
    logger.info("Initializing ImageModel with fresh ImageNet-1K pretrained weights (pretrained=True)...")
    logger.info("Explicitly verifying: Epoch 12 and previous checkpoints are NOT loaded.")
    model = ImageModel(num_classes=2, pretrained=True).to(device)

    # Freeze Layer 1-2 (and stem), keep Layer 3-4 and FC classification head trainable
    logger.info("Configuring parameter freezing: Layers 1-2 frozen; Layers 3-4 and classification head trainable.")
    trainable_prefixes = ("layer3", "layer4", "fc")
    trainable_count = 0
    frozen_count = 0
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(p) for p in trainable_prefixes):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    for name, module in model.backbone.named_modules():
        if not any(name.startswith(p) for p in trainable_prefixes) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    logger.info(f"Parameters: {trainable_count:,} trainable (Layer 3-4 + FC), {frozen_count:,} frozen (Stem + Layer 1-2)")

    backbone_params = [p for n, p in model.named_parameters() if "backbone.fc" not in n and p.requires_grad]
    head_params = [p for p in model.backbone.fc.parameters() if p.requires_grad]
    
    backbone_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 1e-4

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr}
    ], weight_decay=weight_decay)
    logger.info(f"Optimizer: Adam (Backbone LR={backbone_lr}, Head LR={head_lr}, Weight Decay={weight_decay})")

    # 3. Datasets and Class-Weighted Loss with Label Smoothing
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    val_dataset = DeepfakeDataset(split="val", media_type="image", manifest_path=manifest_path)

    weights = compute_manifest_class_weights(train_dataset)
    criterion_weight = weights.to(device)
    w_real = float(weights[0])
    w_fake = float(weights[1])
    logger.info(f"Class-Weighted Loss Configured: REAL (0) = {w_real:.4f}, FAKE (1) = {w_fake:.4f}")

    label_smoothing = 0.05
    criterion = nn.CrossEntropyLoss(weight=criterion_weight, label_smoothing=label_smoothing)
    logger.info(f"Loss Function: CrossEntropyLoss (Class Weights=[{w_real:.4f}, {w_fake:.4f}], label_smoothing={label_smoothing})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_collate_fn
    )

    # 4. Pre-Training OOD Control Benchmark (Before Training)
    logger.info("\n" + "=" * 80)
    logger.info("--- [PRE-TRAINING] EVALUATING CORRECTED OOD CONTROL BENCHMARK ON INITIAL WEIGHTS ---")
    logger.info("=" * 80)
    pre_ood = evaluate_ood_control_benchmark(model, device)
    logger.info(f"Pre-training OOD Real FPR: {pre_ood['real_false_positive_rate_pct']}% | Fake Detection: {pre_ood['fake_detection_rate_pct']}%")
    logger.info(f"Mean Real Fake Prob: {pre_ood['mean_real_fake_prob']:.4f} | Mean Fake Fake Prob: {pre_ood['mean_fake_fake_prob']:.4f}")
    for item in pre_ood["sample_details"]:
        logger.info(f"  [{item['ground_truth']}] {item['name']:<20} P(REAL)={item['real_prob']:.4f}, P(FAKE)={item['fake_prob']:.4f} -> Pred={item['pred_class']}, Fusion={item['final_fusion_verdict']}")

    history = []
    best_generalization_score = -1.0
    best_epoch = 0
    patience = 4
    patience_counter = 0

    # 5. Training Loop
    for ep in range(epochs):
        ep_display = ep + 1
        logger.info(f"\n" + "=" * 80)
        logger.info(f">>> Epoch {ep_display}/{epochs} Training Starting...")
        logger.info("=" * 80)
        model.train()
        # Keep frozen BatchNorm layers in eval mode
        for name, module in model.backbone.named_modules():
            if not any(name.startswith(p) for p in trainable_prefixes) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        train_loss_accum = 0.0
        train_correct = 0
        train_processed = 0
        loss_trend = []
        ep_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                logger.info(f"Reached maximum train batches limit ({max_train_batches}). Concluding epoch.")
                break
            if images is None or targets is None or len(images) == 0:
                continue

            # Verify augmentation tensor shape and absence of NaN/Inf
            if images.shape[1:] != torch.Size([3, 224, 224]):
                raise RuntimeError(f"Invalid tensor shape {images.shape} at batch {batch_idx+1}, expected (*, 3, 224, 224)")
            if torch.isnan(images).any() or torch.isinf(images).any():
                raise RuntimeError(f"NaN/Inf detected in input batch at Batch {batch_idx+1}")

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Verify loss remains finite
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at Epoch {ep_display}, Batch {batch_idx+1}: {loss.item()}")

            loss.backward()

            # Verify gradients remain finite
            for p_name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Non-finite gradient in {p_name} at Epoch {ep_display}, Batch {batch_idx+1}")

            optimizer.step()

            batch_loss = loss.item()
            train_loss_accum += batch_loss * len(targets)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()
            train_processed += len(targets)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == max_train_batches:
                curr_acc = (train_correct / max(1, train_processed)) * 100.0
                used_ram, total_ram = get_ram_usage_mb()
                vram_alloc, vram_max = get_vram_usage_mb()
                loss_trend.append({"batch": batch_idx + 1, "loss": round(batch_loss, 4), "acc": round(curr_acc, 2)})
                logger.info(f"Epoch {ep_display} | Batch {batch_idx+1:>3}/{max_train_batches or len(train_loader)} | Loss: {batch_loss:.4f} | Acc: {curr_acc:.2f}% | VRAM: {vram_alloc:.1f}MB | RAM: {used_ram:.0f}MB")

            if (batch_idx + 1) % 100 == 0:
                intra_ckpt = {
                    "epoch": ep_display,
                    "batch_idx": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "saved_at": datetime.now().isoformat()
                }
                intra_path = os.path.join(exp_dir, "intra_epoch_checkpoint.pth")
                save_checkpoint_atomic(intra_ckpt, intra_path)

        train_epoch_loss = train_loss_accum / max(1, train_processed)
        train_epoch_acc = (train_correct / max(1, train_processed)) * 100.0
        ep_duration = round(time.time() - ep_start_time, 1)
        logger.info(f"Epoch {ep_display} Training Completed in {ep_duration}s: Loss = {train_epoch_loss:.4f}, Accuracy = {train_epoch_acc:.2f}% ({train_processed} samples)")

        # 6. In-Domain Validation
        logger.info(f"\n>>> Epoch {ep_display}/{epochs} In-Domain Validation Starting...")
        model.eval()
        val_loss_accum = 0.0
        val_targets_all = []
        val_probs_all = []

        with torch.no_grad():
            for v_idx, (images, targets) in enumerate(val_loader):
                if max_val_batches is not None and v_idx >= max_val_batches:
                    logger.info(f"Reached maximum validation batches limit ({max_val_batches}). Concluding validation.")
                    break
                if images is None or targets is None or len(images) == 0:
                    continue
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss_accum += loss.item() * len(targets)

                probs = torch.softmax(outputs, dim=1)
                val_targets_all.extend(targets.cpu().numpy())
                val_probs_all.extend(probs[:, 1].cpu().numpy())

        val_epoch_loss = val_loss_accum / max(1, len(val_targets_all))
        val_targets_arr = np.array(val_targets_all)
        val_probs_arr = np.array(val_probs_all)

        opt_thresh, ep_macro_f1 = optimize_threshold(val_targets_arr, val_probs_arr)
        val_metrics = compute_all_metrics(val_targets_arr, val_probs_arr, threshold=0.50)
        val_metrics["val_loss"] = val_epoch_loss
        val_metrics["train_loss"] = train_epoch_loss
        val_metrics["train_accuracy"] = train_epoch_acc
        val_metrics["optimal_threshold"] = opt_thresh
        val_metrics["optimal_macro_f1"] = ep_macro_f1
        val_metrics["samples_evaluated"] = len(val_targets_all)

        for k, v in val_metrics.items():
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                val_metrics[k] = 0.0

        logger.info(f"In-Domain Validation Results (N={len(val_targets_all)}):")
        logger.info(f"  Accuracy:  {val_metrics['accuracy']:.2f}%")
        logger.info(f"  Macro-F1:  {val_metrics['macro_f1']:.4f} (at threshold=0.50), {ep_macro_f1:.4f} (at opt threshold={opt_thresh:.2f})")
        logger.info(f"  REAL F1:   {val_metrics['real_f1']:.4f} (Recall: {val_metrics['real_recall']:.4f}, Prec: {val_metrics['real_precision']:.4f})")
        logger.info(f"  FAKE F1:   {val_metrics['fake_f1']:.4f} (Recall: {val_metrics['fake_recall']:.4f}, Prec: {val_metrics['fake_precision']:.4f})")
        logger.info(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
        logger.info(f"  ECE:       {val_metrics['ece']:.4f}")

        # 7. Post-Epoch Corrected OOD Benchmark
        logger.info(f"\n--- [EPOCH {ep_display}] EVALUATING CORRECTED OOD CONTROL BENCHMARK ---")
        post_ood = evaluate_ood_control_benchmark(model, device)
        logger.info(f"OOD Real FPR: {post_ood['real_false_positive_rate_pct']}% | Fake Detection: {post_ood['fake_detection_rate_pct']}%")
        logger.info(f"Mean Real Fake Prob: {post_ood['mean_real_fake_prob']:.4f} | Mean Fake Fake Prob: {post_ood['mean_fake_fake_prob']:.4f}")
        for item in post_ood["sample_details"]:
            logger.info(f"  [{item['ground_truth']}] {item['name']:<20} P(REAL)={item['real_prob']:.4f}, P(FAKE)={item['fake_prob']:.4f} -> Pred={item['pred_class']}, Fusion={item['final_fusion_verdict']}")

        # 8. Check Promotion Gates and Generalization Composite Score (Sections 6 & 7)
        gate_status = check_promotion_gates(val_metrics, post_ood)
        gen_score = compute_generalization_score(val_metrics, post_ood)
        logger.info(f"Epoch {ep_display} Generalization Composite Score: {gen_score:.4f} | Promotion Gates Passed: {gate_status['all_passed']}")
        for g_k, g_v in gate_status["gates"].items():
            logger.info(f"    Gate '{g_k}': {'[PASS]' if g_v else '[FAIL]'}")

        # 9. Save Checkpoints (Atomic and Isolated)
        checkpoint_payload = {
            "epoch": ep_display,
            "batches_trained": batch_idx + 1 if 'batch_idx' in locals() else max_train_batches,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_epoch_loss,
            "val_loss": val_epoch_loss,
            "metrics": val_metrics,
            "ood_benchmark": post_ood,
            "promotion_gate_status": gate_status,
            "generalization_score": gen_score,
            "class_weights": {"real": w_real, "fake": w_fake},
            "saved_at": datetime.now().isoformat()
        }

        # Save latest checkpoint
        save_checkpoint_atomic(checkpoint_payload, latest_ckpt_path)
        logger.info(f"Saved latest isolated checkpoint to: {latest_ckpt_path}")

        # Save individual epoch checkpoint
        epoch_ckpt_path = os.path.join(exp_dir, f"checkpoint_epoch_{ep_display}.pth")
        save_checkpoint_atomic(checkpoint_payload, epoch_ckpt_path)
        logger.info(f"Saved epoch {ep_display} checkpoint to: {epoch_ckpt_path}")

        # Update best checkpoint based on composite generalization score
        if gen_score > best_generalization_score:
            best_generalization_score = gen_score
            best_epoch = ep_display
            patience_counter = 0
            save_checkpoint_atomic(checkpoint_payload, best_ckpt_path)
            logger.info(f">>> NEW BEST GENERALIZATION CHECKPOINT (Epoch {best_epoch}, Score={best_generalization_score:.4f}) saved to: {best_ckpt_path}")
        else:
            patience_counter += 1
            logger.info(f"No generalization score improvement (Best Epoch {best_epoch}: {best_generalization_score:.4f}, Current: {gen_score:.4f}). Early stopping patience: {patience_counter}/{patience}")

        history.append({
            "epoch": ep_display,
            "train_loss": round(train_epoch_loss, 4),
            "val_loss": round(val_epoch_loss, 4),
            "macro_f1": round(val_metrics["macro_f1"], 4),
            "real_f1": round(val_metrics["real_f1"], 4),
            "fake_f1": round(val_metrics["fake_f1"], 4),
            "roc_auc": round(val_metrics["roc_auc"], 4),
            "ece": round(val_metrics["ece"], 4),
            "ood_real_fpr_pct": post_ood["real_false_positive_rate_pct"],
            "ood_fake_det_pct": post_ood["fake_detection_rate_pct"],
            "mean_real_fake_prob": post_ood["mean_real_fake_prob"],
            "mean_fake_fake_prob": post_ood["mean_fake_fake_prob"],
            "generalization_score": gen_score,
            "gates_passed": gate_status["all_passed"]
        })

        # Save progress log after each epoch
        intermediate_report = {
            "experiment": "Experiment C-Generalization",
            "mode": mode_name,
            "current_epoch": ep_display,
            "best_epoch": best_epoch,
            "best_generalization_score": best_generalization_score,
            "history": history
        }
        with open(progress_log_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_report, f, indent=4)

        # 10. Early Stopping Checks (Section 10)
        # Check catastrophic class collapse
        if val_metrics.get("class_collapse", False) and ep_display >= 2:
            logger.warning(f"Catastrophic class collapse detected at Epoch {ep_display}. Triggering early stopping.")
            break

        # Check OOD deterioration
        if ep_display >= 3 and post_ood["real_false_positive_rate_pct"] >= 100.0 and post_ood["fake_detection_rate_pct"] <= 0.0:
            logger.warning(f"OOD catastrophic collapse detected at Epoch {ep_display}. Triggering early stopping.")
            break

        # Check patience
        if patience_counter >= patience and ep_display >= 4:
            logger.info(f"Early stopping patience reached ({patience} epochs without generalization improvement). Halting training.")
            break

    # 11. Final Three-Way Comparison Benchmark (Section 11)
    logger.info("\nTraining concluded. Running comprehensive three-way evaluation benchmark...")
    comparison_results = evaluate_three_way_comparison(best_ckpt_path, device, val_loader, max_val_batches=max_val_batches or 50)

    final_report = {
        "experiment": "Experiment C-Generalization",
        "completed_at": datetime.now().isoformat(),
        "configuration": {
            "initialization": "ImageNet-1K (ResNet18_Weights.DEFAULT)",
            "frozen_layers": "Layer 1-2 (Stem + conv1/bn1 + layer1 + layer2)",
            "trainable_layers": "Layer 3-4 + FC classification head",
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "class_weights": {"real": round(w_real, 4), "fake": round(w_fake, 4)},
            "label_smoothing": label_smoothing,
            "precision": "FP32",
            "batch_size": batch_size,
            "total_epochs": ep_display,
            "best_epoch": best_epoch,
            "best_generalization_score": best_generalization_score,
            "num_workers": num_workers,
            "augmentations_active": True
        },
        "history": history,
        "three_way_comparison": comparison_results,
        "checkpoint_path": best_ckpt_path
    }

    with open(progress_log_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    logger.info(f"Final Experiment C-Generalization log written to: {progress_log_path}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment C-Generalization")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs (planned 8-10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--pilot", action="store_true", help="Run 1-epoch / 200-batch pilot")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=50, help="Max val batches")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    args = parser.parse_args()

    if args.pilot:
        run_experiment_c_generalization(
            epochs=1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_batches=args.max_train_batches if args.max_train_batches is not None else 200,
            max_val_batches=args.max_val_batches if args.max_val_batches is not None else 50,
            is_pilot=True
        )
    else:
        run_experiment_c_generalization(
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
            is_pilot=False
        )

