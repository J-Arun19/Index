import os
import sys
import json
import time
import cv2
import torch
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from backend.utils.logger import setup_logger
from backend.utils.metrics_helper import compute_ece
from backend.agents.experimental.experimental_sbi_agent import ExperimentalSBIImageAgent
from backend.agents.experimental.experimental_image_agent import ExperimentalImageAgent

logger = setup_logger("FusionCalibrator")

CONFIG_OUT_PATH = "backend/models/experimental/fusion_calibration_config.json"
VAL_MANIFEST_PATH = "research/experimental_fusion/datasets/validation_manifest_v2.json"

def nll_cost(temperature: float, logits: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Negative Log-Likelihood (Binary Cross-Entropy) given temperature T."""
    t = max(1e-2, float(temperature))
    scaled_logits = logits / t
    # numerically stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    bce = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
    return float(bce)

def fit_temperature(raw_probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    """Fits optimal temperature T to minimize NLL."""
    p_clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(p_clipped / (1.0 - p_clipped))
    
    ece_before = compute_ece(raw_probs, y_true)
    
    res = minimize_scalar(
        nll_cost,
        bounds=(0.1, 10.0),
        method="bounded",
        args=(logits, y_true),
        options={"xatol": 1e-3, "maxiter": 500}
    )
    opt_t = float(res.x)
    
    # Calibrated probabilities
    cal_logits = logits / opt_t
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    ece_after = compute_ece(cal_probs, y_true)
    
    return opt_t, round(ece_before, 4), round(ece_after, 4)

def run_calibration():
    print("=" * 80)
    print("FITTING ARCHITECTURE-SPECIFIC TEMPERATURE CALIBRATION (VALIDATION SPLIT ONLY)")
    print("=" * 80)

    with open(VAL_MANIFEST_PATH) as f:
        val_data = json.load(f)

    val_images = val_data["images"]
    print(f"Loaded {len(val_images)} validation image candidates from {VAL_MANIFEST_PATH}")
    print("Locked test manifest is NOT accessed during calibration.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing experimental agents on {device}...")
    sbi_agent = ExperimentalSBIImageAgent(device=device)
    xcep_agent = ExperimentalImageAgent(backbone="xception", device=device)

    sbi_raw_scores = []
    xcep_raw_scores = []
    y_true = []

    print("\nRunning inference on validation split...")
    for idx, item in enumerate(val_images, 1):
        vid_p = item["source_video_path"]
        gt = 1 if item["ground_truth"] == "FAKE" else 0
        fn = item["source_filename"]

        # Extract middle frame
        cap = cv2.VideoCapture(vid_p)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            continue

        # Save temporary frame in scratch
        os.makedirs("scratch/temp_val", exist_ok=True)
        tmp_frame_p = os.path.join("scratch/temp_val", f"val_frame_{idx:03d}.jpg")
        cv2.imwrite(tmp_frame_p, frame)

        # 1. SBI Inference
        try:
            res_sbi = sbi_agent.analyze(tmp_frame_p, require_face=False, crop_mode="sbi_margin")
            raw_s = res_sbi.get("raw_model_score") or 0.50
        except Exception:
            raw_s = 0.50

        # 2. Xception Inference
        try:
            res_xcep = xcep_agent.analyze(tmp_frame_p, require_face=False)
            raw_x = res_xcep.get("raw_model_score") or 0.50
        except Exception:
            raw_x = 0.50

        sbi_raw_scores.append(raw_s)
        xcep_raw_scores.append(raw_x)
        y_true.append(gt)

        if idx % 10 == 0 or idx == len(val_images):
            print(f"  [{idx:02d}/{len(val_images)}] {fn[:22]} | GT={gt} | SBI={raw_s:.3f} | Xcep={raw_x:.3f}")

    sbi_raw_arr = np.array(sbi_raw_scores, dtype=np.float64)
    xcep_raw_arr = np.array(xcep_raw_scores, dtype=np.float64)
    y_true_arr = np.array(y_true, dtype=np.int32)

    # Fit SBI Temperature
    print("\n--- Fitting Temperature for SBI (EfficientNet-B4) ---")
    t_sbi, ece_sbi_before, ece_sbi_after = fit_temperature(sbi_raw_arr, y_true_arr)
    print(f"Optimal T_SBI: {t_sbi:.4f} | ECE Before: {ece_sbi_before:.4f} -> After: {ece_sbi_after:.4f}")

    # Fit Xception Temperature
    print("\n--- Fitting Temperature for Xception ---")
    t_xcep, ece_xcep_before, ece_xcep_after = fit_temperature(xcep_raw_arr, y_true_arr)
    print(f"Optimal T_Xception: {t_xcep:.4f} | ECE Before: {ece_xcep_before:.4f} -> After: {ece_xcep_after:.4f}")

    # Build Configuration Object
    config_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fitted_on": "validation_split_only",
        "validation_manifest": VAL_MANIFEST_PATH,
        "sample_count": len(y_true),
        "sbi": {
            "model": "SBI EfficientNet-B4 (FFc23.tar)",
            "temperature": round(t_sbi, 4),
            "validation_ece_before": ece_sbi_before,
            "validation_ece_after": ece_sbi_after,
        },
        "xception": {
            "model": "DeepfakeBench Xception (xception_best.pth)",
            "temperature": round(t_xcep, 4),
            "validation_ece_before": ece_xcep_before,
            "validation_ece_after": ece_xcep_after,
        },
        "decision_thresholds": {
            "tau_real": 0.35,
            "tau_fake": 0.65,
            "min_confidence": 60.0,
            "abstention_strategy": "conservative_disagreement_to_uncertain",
        },
        "provenance_notes": (
            "Temperatures T_sbi and T_xcep were independently optimized strictly on the validation split "
            "(val.json + non-test Celeb-DF) by minimizing binary negative log-likelihood. "
            "Locked test split was strictly excluded from calibration."
        )
    }

    os.makedirs(os.path.dirname(CONFIG_OUT_PATH), exist_ok=True)
    with open(CONFIG_OUT_PATH, "w") as f:
        json.dump(config_payload, f, indent=2)

    print(f"\n[+] Saved isolated fusion calibration config to {CONFIG_OUT_PATH}")

if __name__ == "__main__":
    run_calibration()
