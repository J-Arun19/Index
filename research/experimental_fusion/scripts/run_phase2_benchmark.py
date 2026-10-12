import os
import sys
import time
import json
import hashlib
import cv2
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from backend.utils.logger import setup_logger
from backend.core.decision_system import DecisionState
from backend.preprocessing.quality_gate import ImageQualityGate
from backend.preprocessing.face_detector import FaceDetector
from backend.forensics.forensic_analyzers import ForensicEvidenceSuite
from backend.models.experimental.sbi_detector import SBIDetector
from backend.models.experimental.xception_detector import XceptionDetector
from backend.agents.experimental.experimental_fusion_agent import ForensicFusionEngine

logger = setup_logger("Phase2Benchmark")

# Phase 1 Immutable Regression Artifacts & Hashes
PHASE1_EXPECTED_HASHES = {
    "scratch/image_eval_manifest.json": "48413326aa5a185ac1096594a47353ec70da96899ac6109a89049c8941b8f58a",
    "scratch/expanded_video_benchmark_results.json": "73f426942e0cd9964be2d6635c436e773719afcc50c3a0e1f51f36f986c1e11f",
    "scratch/full_video_sbi_benchmark_results.json": "da3e55f005e3c9771bfa2563b3a7a7e9d0062c4d3d7b6c1bee255733b4a1764f",
    "scratch/still_image_validation_results.json": "2bf28b201e7ce2c2149fa9cf852e074da374cd4849ba0d75de36596dee09de98",
    "research/experimental_fusion/fusion_benchmark_results.json": "f8dd080435801edbea81cfa7c2b7c0a0a7a3ff6a77dee8366c924a1da972e5d7",
    "research/experimental_fusion/FUSION_BENCHMARK_REPORT.md": "dbd44777d506b8d3fb0b31f381986d87d59e44f5404da2e64d45acce3c03d13e",
    "research/experimental_fusion/FUSION_AUDIT_REPORT.md": "90eae8d177196acef778478c2a0b417367176b7166bf20c7514212521b8e166e",
}

# Phase 2 Sealed Manifests
PHASE2_MANIFEST_HASHES = {
    "research/experimental_fusion/datasets/locked_test_manifest_v2.json": "16a1972f158d06f9009b647a2b08b29ee0aef01235175d8ce7291398b3084e55",
    "research/experimental_fusion/datasets/validation_manifest_v2.json": "baf6f6615bc176341fd08cfa5e6fdc2797ba0e2c501876f2c04d40dcdb94d21a",
}

OUTPUT_DIR = "research/experimental_fusion/phase2"
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "phase2_benchmark_results.json")

SHARED_CELEB_IDS = ["id0", "id10", "id16", "id17"]


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_all_manifest_hashes(label: str = "Pre-Execution") -> bool:
    """Verifies that all recorded hashes match 100%."""
    print(f"\n--- Verifying Integrity Hashes ({label}) ---")
    all_matched = True

    # Check Phase 1 artifacts
    for rel_path, exp_h in PHASE1_EXPECTED_HASHES.items():
        if not os.path.exists(rel_path):
            print(f"[FAIL] Missing Phase 1 artifact: {rel_path}")
            all_matched = False
            continue
        act_h = compute_sha256(rel_path)
        if act_h != exp_h:
            print(f"[FAIL] Hash mismatch for Phase 1 artifact {rel_path}!\n  Expected: {exp_h}\n  Actual:   {act_h}")
            all_matched = False
        else:
            print(f"  [OK] {rel_path} verified.")

    # Check Phase 2 manifests
    for rel_path, exp_h in PHASE2_MANIFEST_HASHES.items():
        if not os.path.exists(rel_path):
            print(f"[FAIL] Missing Phase 2 manifest: {rel_path}")
            all_matched = False
            continue
        act_h = compute_sha256(rel_path)
        if act_h != exp_h:
            print(f"[FAIL] Hash mismatch for Phase 2 manifest {rel_path}!\n  Expected: {exp_h}\n  Actual:   {act_h}")
            all_matched = False
        else:
            print(f"  [OK] {rel_path} verified.")

    if not all_matched:
        raise RuntimeError(f"Integrity check failed during {label}! Execution halted to protect regression artifacts.")
    print(f"[SUCCESS] All recorded hashes verified ({label}).")
    return True


def check_is_shared_celeb_id(filename: str, dataset: str) -> bool:
    """Checks whether a video or image filename belongs to the 4 shared Celeb-DF identities."""
    if "celeb" not in dataset.lower() and "celeb" not in filename.lower():
        return False
    import re
    tokens = re.findall(r"id\d+", filename.lower())
    return any(t in SHARED_CELEB_IDS for t in tokens)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Dict[str, Any]:
    """Calculates Wilson score 95% confidence interval for proportion k / n."""
    if n == 0:
        return {"k": k, "n": n, "proportion": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_percent": "[0.0%, 0.0%]"}
    p_hat = k / n
    z = 1.95996
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return {
        "k": k,
        "n": n,
        "proportion": round(p_hat, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_percent": f"[{round(ci_lower * 100, 2)}%, {round(ci_upper * 100, 2)}%]",
    }


def compute_binary_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes binary classification metrics from predictions."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "FAKE")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "FAKE" and yp == "REAL")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "REAL")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "REAL" and yp == "FAKE")

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_fake = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    prec_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_real = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_real = (2 * prec_real * rec_real) / (prec_real + rec_real) if (prec_real + rec_real) > 0 else 0.0
    macro_f1 = (f1_fake + f1_real) / 2.0

    # Wilson Confidence Intervals
    acc_ci = wilson_ci(tp + tn, total)
    fpr_ci = wilson_ci(fp, tn + fp)
    fnr_ci = wilson_ci(fn, tp + fn)
    prec_ci = wilson_ci(tp, tp + fp)
    rec_ci = wilson_ci(tp, tp + fn)

    return {
        "confusion_matrix": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_tpr": round(rec, 4),
        "specificity_tnr": round(spec, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "f1_score": round(f1_fake, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total,
        "confidence_intervals_95": {
            "accuracy": acc_ci,
            "false_positive_rate": fpr_ci,
            "false_negative_rate": fnr_ci,
            "precision": prec_ci,
            "recall": rec_ci,
        },
    }


def compute_3state_metrics(y_true: List[str], verdicts: List[str]) -> Dict[str, Any]:
    """Computes three-state safety metrics."""
    real_indices = [i for i, yt in enumerate(y_true) if yt == "REAL"]
    fake_indices = [i for i, yt in enumerate(y_true) if yt == "FAKE"]

    real_lr = sum(1 for i in real_indices if verdicts[i] == DecisionState.LIKELY_REAL)
    real_unc = sum(1 for i in real_indices if verdicts[i] == DecisionState.UNCERTAIN)
    real_fk = sum(1 for i in real_indices if verdicts[i] == DecisionState.FAKE)

    fake_fk = sum(1 for i in fake_indices if verdicts[i] == DecisionState.FAKE)
    fake_unc = sum(1 for i in fake_indices if verdicts[i] == DecisionState.UNCERTAIN)
    fake_lr = sum(1 for i in fake_indices if verdicts[i] == DecisionState.LIKELY_REAL)

    total = len(y_true)
    total_unc = sum(1 for v in verdicts if v == DecisionState.UNCERTAIN)
    total_not_suitable = sum(1 for v in verdicts if v == DecisionState.NOT_SUITABLE)

    abstention_rate = total_unc / total if total > 0 else 0.0
    real_abstention_rate = real_unc / len(real_indices) if real_indices else 0.0
    fake_abstention_rate = fake_unc / len(fake_indices) if fake_indices else 0.0

    fpr_3st = real_fk / len(real_indices) if real_indices else 0.0
    fnr_3st = fake_lr / len(fake_indices) if fake_indices else 0.0

    decided_total = real_lr + real_fk + fake_fk + fake_lr
    decided_correct = real_lr + fake_fk
    decided_accuracy = decided_correct / decided_total if decided_total > 0 else 0.0

    raw_3st_accuracy = (real_lr + fake_fk) / total if total > 0 else 0.0

    # Wilson Confidence Intervals
    abstention_ci = wilson_ci(total_unc, total)
    fpr_3st_ci = wilson_ci(real_fk, len(real_indices))
    fnr_3st_ci = wilson_ci(fake_lr, len(fake_indices))
    decided_acc_ci = wilson_ci(decided_correct, decided_total)

    return {
        "real_samples": {
            "total": len(real_indices),
            "likely_real_tn": real_lr,
            "uncertain_abstained": real_unc,
            "fake_fp": real_fk,
            "false_positive_rate_3state": round(fpr_3st, 4),
        },
        "fake_samples": {
            "total": len(fake_indices),
            "fake_tp": fake_fk,
            "uncertain_abstained": fake_unc,
            "likely_real_fn": fake_lr,
            "false_negative_rate_3state": round(fnr_3st, 4),
        },
        "abstention_rate_overall": round(abstention_rate, 4),
        "real_abstention_rate": round(real_abstention_rate, 4),
        "fake_abstention_rate": round(fake_abstention_rate, 4),
        "decided_cases": {
            "total_decided": decided_total,
            "correct_decided": decided_correct,
            "decided_accuracy": round(decided_accuracy, 4),
        },
        "raw_3state_accuracy": round(raw_3st_accuracy, 4),
        "not_suitable_count": total_not_suitable,
        "confidence_intervals_95": {
            "abstention_rate": abstention_ci,
            "fpr_3state": fpr_3st_ci,
            "fnr_3state": fnr_3st_ci,
            "decided_accuracy": decided_acc_ci,
        },
    }


def extract_sbi_crop(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 0.25) -> np.ndarray:
    """Extracts face crop with 25% boundary expansion (w/4, h/4) for SBI."""
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]
    wm = int(w * margin_ratio)
    hm = int(h * margin_ratio)
    x0 = max(0, x - wm)
    y0 = max(0, y - hm)
    x1 = min(W, x + w + wm)
    y1 = min(H, y + h + hm)
    return frame_bgr[y0:y1, x0:x1]


def run_benchmark():
    print("=" * 95)
    print("SATYADRISHTI PHASE 2 LOCKED BENCHMARK RUNNER")
    print("Strict Zero-Leakage & Sequential GTX-1650 Execution")
    print("=" * 95)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Pre-execution hash check
    verify_all_manifest_hashes("Pre-Execution")

    # 2. Load locked test manifest
    manifest_path = "research/experimental_fusion/datasets/locked_test_manifest_v2.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_videos = manifest["videos"]
    test_images = manifest["images"]
    print(f"Loaded {len(test_videos)} test videos and {len(test_images)} test images from locked manifest.")

    # Verify all files exist
    print("Verifying all local file paths exist...")
    missing_paths = []
    for v in test_videos:
        if not os.path.exists(v["filepath"]):
            missing_paths.append(v["filepath"])
    for img in test_images:
        p = img["source_video_path"]
        if not os.path.exists(p):
            missing_paths.append(p)

    if missing_paths:
        raise FileNotFoundError(f"Found {len(missing_paths)} missing test paths! Example: {missing_paths[0]}")
    print("All file paths verified on local disk.")

    # 3. Initialize components sequentially on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nInitializing neural detectors and forensic engines on {device}...")

    sbi_detector = SBIDetector(device=device)
    xcep_detector = XceptionDetector(device=device)
    face_detector = FaceDetector()
    quality_gate = ImageQualityGate()
    forensics = ForensicEvidenceSuite()

    # Recalibrated Fusion Engine with validation-derived temperatures
    engine = ForensicFusionEngine(
        t_sbi=2.7676,
        t_xcep=2.4210,
        tau_real=0.35,
        tau_fake=0.65,
        min_confidence=60.0,
    )
    print(f"ForensicFusionEngine initialized: T_sbi={engine.t_sbi}, T_xcep={engine.t_xcep}, tau_real={engine.tau_real}, tau_fake={engine.tau_fake}")

    # =========================================================================
    # PART 1: STILL-IMAGE BENCHMARK (207 Items)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING PART 1: STILL-IMAGE EVALUATION (207 Items)")
    print("Acknowledge: 200 items are middle frames extracted from the 200 test videos;")
    print("7 items are standalone in-the-wild and safety control images.")
    print("=" * 90)

    image_results = []
    t0_images_total = time.time()

    for idx, item in enumerate(test_images, 1):
        item_id = item["id"]
        gt = item["ground_truth"]
        dataset = item["dataset"]
        subset = item["subset"]
        frame_sel = item["frame_selection"]
        src_path = item["source_video_path"]
        src_fn = item["source_filename"]
        is_shared_celeb = check_is_shared_celeb_id(src_fn, dataset)

        t0_sample = time.time()

        # Load or extract frame
        if frame_sel == "static_image":
            frame_bgr = cv2.imread(src_path)
        else:
            cap = cv2.VideoCapture(src_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_idx = max(0, frame_count // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret or frame_bgr is None:
                raise RuntimeError(f"Could not read frame from {src_path}")

        if frame_bgr is None:
            raise RuntimeError(f"Image frame is None for {src_path}")

        # Quality Gate evaluation
        q_res = quality_gate.evaluate(frame_bgr, require_face=False)

        # Detect face
        faces = face_detector.detect_faces(frame_bgr)
        face_found = len(faces) > 0
        primary_bbox = faces[0] if face_found else None

        # Check for non-face media
        if not face_found or gt == "NON-FACE":
            # Quality Gate marks unsupported
            q_res_unsupp = type("QualityDummy", (), {
                "eligibility": "UNSUPPORTED",
                "is_suitable": False,
                "face_detected": False,
                "rejection_reason": "No human face found in image.",
                "issues": ["UNSUPPORTED_MEDIA"],
                "to_dict": lambda s: {"eligibility": "UNSUPPORTED", "face_detected": False}
            })()
            fusion_dec = engine.evaluate(
                raw_sbi=None,
                raw_xcep=None,
                quality_result=q_res_unsupp,
                forensic_result={},
            )
            raw_sbi, cal_sbi, sbi_pred_str, sbi_forced = None, None, "NOT_SUITABLE", "NOT_SUITABLE"
            raw_xcep, cal_xcep, xcep_pred_str, xcep_forced = None, None, "NOT_SUITABLE", "NOT_SUITABLE"
            fused_fake_prob = None
            fused_forced = "NOT_SUITABLE"
            fused_verdict = fusion_dec.prediction
            fused_conf = None
            fused_reason = fusion_dec.reason
        else:
            # Forensics
            forensic_res = forensics.evaluate(frame_bgr, face_bbox=primary_bbox)

            # Crop for SBI (25% margin)
            sbi_crop = extract_sbi_crop(frame_bgr, primary_bbox, margin_ratio=0.25)
            sbi_out = sbi_detector.predict_crop(sbi_crop)
            raw_sbi = float(sbi_out["fake_probability"])
            cal_sbi = engine.calibrate(raw_sbi, temperature=engine.t_sbi)
            sbi_forced = "FAKE" if raw_sbi >= 0.50 else "REAL"
            sbi_pred_str = "FAKE" if cal_sbi > 0.65 else ("LIKELY_REAL" if cal_sbi < 0.35 else "UNCERTAIN")

            # Crop for Xception (tight crop)
            xcep_crop = face_detector.crop_face(frame_bgr, primary_bbox)
            xcep_out = xcep_detector.predict_frame(xcep_crop)
            raw_xcep = float(xcep_out["fake_probability"])
            cal_xcep = engine.calibrate(raw_xcep, temperature=engine.t_xcep)
            xcep_forced = "FAKE" if raw_xcep >= 0.50 else "REAL"
            xcep_pred_str = "FAKE" if cal_xcep > 0.65 else ("LIKELY_REAL" if cal_xcep < 0.35 else "UNCERTAIN")

            # Fusion decision
            fusion_dec = engine.evaluate(
                raw_sbi=raw_sbi,
                raw_xcep=raw_xcep,
                quality_result=q_res,
                forensic_result=forensic_res,
            )
            fused_fake_prob = fusion_dec.fake_probability
            fused_forced = "FAKE" if (fused_fake_prob or 0.5) >= 0.50 else "REAL"
            fused_verdict = fusion_dec.prediction
            fused_conf = fusion_dec.confidence
            fused_reason = fusion_dec.reason

        sample_latency = round(time.time() - t0_sample, 4)

        image_entry = {
            "id": item_id,
            "filename": src_fn,
            "source_video_id": item["source_video_id"],
            "dataset": dataset,
            "subset": subset,
            "ground_truth": gt,
            "frame_selection": frame_sel,
            "is_shared_celeb_id": is_shared_celeb,
            "face_detected": face_found,
            "sbi": {
                "raw_fake_prob": round(raw_sbi, 4) if raw_sbi is not None else None,
                "calibrated_prob": round(cal_sbi, 4) if cal_sbi is not None else None,
                "forced_binary": sbi_forced,
                "verdict_3state": sbi_pred_str,
            },
            "xception": {
                "raw_fake_prob": round(raw_xcep, 4) if raw_xcep is not None else None,
                "calibrated_prob": round(cal_xcep, 4) if cal_xcep is not None else None,
                "forced_binary": xcep_forced,
                "verdict_3state": xcep_pred_str,
            },
            "fusion": {
                "fused_probability": round(fused_fake_prob, 4) if fused_fake_prob is not None else None,
                "confidence": fused_conf,
                "forced_binary": fused_forced,
                "verdict_3state": fused_verdict,
                "reason": fused_reason,
            },
            "latency_seconds": sample_latency,
        }
        image_results.append(image_entry)

        if idx % 25 == 0 or idx == len(test_images):
            print(f"  [Image {idx:03d}/{len(test_images)}] {src_fn[:25]} | GT: {gt:8s} | Fused: {fused_verdict:11s} (P={fused_fake_prob}) | {sample_latency:.3f}s")

    t_images_elapsed = round(time.time() - t0_images_total, 2)
    print(f"Finished Still-Image Evaluation in {t_images_elapsed}s.")

    # =========================================================================
    # PART 2: VIDEO BENCHMARK (200 Clips)
    # =========================================================================
    print("\n" + "=" * 90)
    print("EXECUTING PART 2: VIDEO EVALUATION (200 Clips)")
    print("8 Uniform Frames Sampled Per Clip; Sequential Shared-Frame Decoding")
    print("=" * 90)

    video_results = []
    t0_videos_total = time.time()

    for idx, item in enumerate(test_videos, 1):
        vid_id = item["id"]
        gt = item["ground_truth"]
        dataset = item["dataset"]
        subset = item["subset"]
        vid_path = item["filepath"]
        fn = item["filename"]
        is_shared_celeb = check_is_shared_celeb_id(fn, dataset)

        t0_sample = time.time()

        # Open video and extract 8 uniform frames
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            sample_indices = [0]
        else:
            num_samples = min(8, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

        frames: List[np.ndarray] = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No readable frames in video {vid_path}")

        # Process frames sequentially
        sbi_probs = []
        xcep_probs = []
        usable_faces = 0

        for frame in frames:
            faces = face_detector.detect_faces(frame)
            if faces:
                usable_faces += 1
                bbox = faces[0]
                # SBI crop (25% margin)
                s_crop = extract_sbi_crop(frame, bbox, margin_ratio=0.25)
                s_out = sbi_detector.predict_crop(s_crop)
                sbi_probs.append(float(s_out["fake_probability"]))

                # Xception crop (tight)
                x_crop = face_detector.crop_face(frame, bbox)
                x_out = xcep_detector.predict_frame(x_crop)
                xcep_probs.append(float(x_out["fake_probability"]))
            else:
                # Fallback to whole frame
                s_out = sbi_detector.predict_crop(frame)
                sbi_probs.append(float(s_out["fake_probability"]))
                x_out = xcep_detector.predict_frame(frame)
                xcep_probs.append(float(x_out["fake_probability"]))

        # Model aggregations
        mean_sbi = float(np.mean(sbi_probs))
        sbi_var = float(np.std(sbi_probs))
        sbi_conflict = bool((max(sbi_probs) > 0.70 and min(sbi_probs) < 0.25) or (sbi_var > 0.22))
        cal_sbi = engine.calibrate(mean_sbi, temperature=engine.t_sbi)
        sbi_forced = "FAKE" if mean_sbi >= 0.50 else "REAL"
        sbi_verdict = "FAKE" if cal_sbi > 0.65 else ("LIKELY_REAL" if cal_sbi < 0.35 else "UNCERTAIN")

        mean_xcep = float(np.mean(xcep_probs))
        xcep_var = float(np.std(xcep_probs))
        xcep_conflict = bool((max(xcep_probs) > 0.70 and min(xcep_probs) < 0.25) or (xcep_var > 0.22))
        cal_xcep = engine.calibrate(mean_xcep, temperature=engine.t_xcep)
        xcep_forced = "FAKE" if mean_xcep >= 0.50 else "REAL"
        xcep_verdict = "FAKE" if cal_xcep > 0.65 else ("LIKELY_REAL" if cal_xcep < 0.35 else "UNCERTAIN")

        # Temporal aggregation across models
        temp_var = max(sbi_var, xcep_var)
        temp_conflict = sbi_conflict or xcep_conflict

        # Midpoint frame forensic analysis
        mid_frame = frames[len(frames) // 2]
        forensic_res = forensics.evaluate(mid_frame)

        # Fusion evaluation
        fusion_dec = engine.evaluate(
            raw_sbi=mean_sbi,
            raw_xcep=mean_xcep,
            temporal_variance=temp_var,
            temporal_conflict=temp_conflict,
            forensic_result=forensic_res,
        )
        fused_fake_prob = fusion_dec.fake_probability
        fused_forced = "FAKE" if (fused_fake_prob or 0.5) >= 0.50 else "REAL"
        fused_verdict = fusion_dec.prediction
        fused_conf = fusion_dec.confidence
        fused_reason = fusion_dec.reason

        sample_latency = round(time.time() - t0_sample, 4)

        video_entry = {
            "id": vid_id,
            "filename": fn,
            "dataset": dataset,
            "subset": subset,
            "ground_truth": gt,
            "is_shared_celeb_id": is_shared_celeb,
            "sampled_frames_count": len(frames),
            "faces_detected_count": usable_faces,
            "temporal_metrics": {
                "sbi_variance": round(sbi_var, 4),
                "xcep_variance": round(xcep_var, 4),
                "max_temporal_variance": round(temp_var, 4),
                "temporal_conflict": temp_conflict,
            },
            "sbi": {
                "raw_fake_prob": round(mean_sbi, 4),
                "calibrated_prob": round(cal_sbi, 4),
                "forced_binary": sbi_forced,
                "verdict_3state": sbi_verdict,
            },
            "xception": {
                "raw_fake_prob": round(mean_xcep, 4),
                "calibrated_prob": round(cal_xcep, 4),
                "forced_binary": xcep_forced,
                "verdict_3state": xcep_verdict,
            },
            "fusion": {
                "fused_probability": round(fused_fake_prob, 4) if fused_fake_prob is not None else None,
                "confidence": fused_conf,
                "forced_binary": fused_forced,
                "verdict_3state": fused_verdict,
                "reason": fused_reason,
            },
            "latency_seconds": sample_latency,
        }
        video_results.append(video_entry)

        if idx % 20 == 0 or idx == len(test_videos):
            print(f"  [Video {idx:03d}/{len(test_videos)}] {fn[:22]} | GT: {gt:8s} | Fused: {fused_verdict:11s} (P={fused_fake_prob}) | {sample_latency:.2f}s")

    t_videos_elapsed = round(time.time() - t0_videos_total, 2)
    print(f"Finished Video Evaluation in {t_videos_elapsed}s.")

    # =========================================================================
    # PART 3: METRIC AGGREGATIONS & COMPARISONS
    # =========================================================================
    print("\n" + "=" * 90)
    print("COMPUTING FORMAL METRICS, 95% WILSON CIS & SUBSET BREAKDOWNS")
    print("=" * 90)

    # 1. Image Metrics (Excluding 2 Non-Face controls from facial binary evaluation)
    img_facial = [r for r in image_results if r["ground_truth"] in ["REAL", "FAKE"]]
    img_y_true = [r["ground_truth"] for r in img_facial]
    img_sbi_b = [r["sbi"]["forced_binary"] for r in img_facial]
    img_sbi_3 = [r["sbi"]["verdict_3state"] for r in img_facial]
    img_xcep_b = [r["xception"]["forced_binary"] for r in img_facial]
    img_xcep_3 = [r["xception"]["verdict_3state"] for r in img_facial]
    img_fused_b = [r["fusion"]["forced_binary"] for r in img_facial]
    img_fused_3 = [r["fusion"]["verdict_3state"] for r in img_facial]

    img_metrics = {
        "sbi": {
            "binary": compute_binary_metrics(img_y_true, img_sbi_b),
            "safety_3state": compute_3state_metrics(img_y_true, img_sbi_3),
        },
        "xception": {
            "binary": compute_binary_metrics(img_y_true, img_xcep_b),
            "safety_3state": compute_3state_metrics(img_y_true, img_xcep_3),
        },
        "fusion": {
            "binary": compute_binary_metrics(img_y_true, img_fused_b),
            "safety_3state": compute_3state_metrics(img_y_true, img_fused_3),
        },
        "latency": {
            "mean": round(float(np.mean([r["latency_seconds"] for r in image_results])), 4),
            "median": round(float(np.median([r["latency_seconds"] for r in image_results])), 4),
            "min": round(float(np.min([r["latency_seconds"] for r in image_results])), 4),
            "max": round(float(np.max([r["latency_seconds"] for r in image_results])), 4),
            "p95": round(float(np.percentile([r["latency_seconds"] for r in image_results], 95)), 4),
        },
    }

    # 2. Video Metrics
    vid_y_true = [r["ground_truth"] for r in video_results]
    vid_sbi_b = [r["sbi"]["forced_binary"] for r in video_results]
    vid_sbi_3 = [r["sbi"]["verdict_3state"] for r in video_results]
    vid_xcep_b = [r["xception"]["forced_binary"] for r in video_results]
    vid_xcep_3 = [r["xception"]["verdict_3state"] for r in video_results]
    vid_fused_b = [r["fusion"]["forced_binary"] for r in video_results]
    vid_fused_3 = [r["fusion"]["verdict_3state"] for r in video_results]

    vid_metrics = {
        "sbi": {
            "binary": compute_binary_metrics(vid_y_true, vid_sbi_b),
            "safety_3state": compute_3state_metrics(vid_y_true, vid_sbi_3),
        },
        "xception": {
            "binary": compute_binary_metrics(vid_y_true, vid_xcep_b),
            "safety_3state": compute_3state_metrics(vid_y_true, vid_xcep_3),
        },
        "fusion": {
            "binary": compute_binary_metrics(vid_y_true, vid_fused_b),
            "safety_3state": compute_3state_metrics(vid_y_true, vid_fused_3),
        },
        "latency": {
            "mean": round(float(np.mean([r["latency_seconds"] for r in video_results])), 4),
            "median": round(float(np.median([r["latency_seconds"] for r in video_results])), 4),
            "min": round(float(np.min([r["latency_seconds"] for r in video_results])), 4),
            "max": round(float(np.max([r["latency_seconds"] for r in video_results])), 4),
            "p95": round(float(np.percentile([r["latency_seconds"] for r in video_results], 95)), 4),
        },
    }

    # 3. Seven Safety Controls Analysis
    control_results = [r for r in image_results if r["dataset"] == "In-the-wild_and_Controls"]
    safety_controls_summary = []
    for c in control_results:
        passed = False
        if c["ground_truth"] == "NON-FACE":
            passed = (c["fusion"]["verdict_3state"] == DecisionState.NOT_SUITABLE and c["fusion"]["fused_probability"] is None)
        elif c["ground_truth"] == "REAL":
            passed = (c["fusion"]["verdict_3state"] in [DecisionState.LIKELY_REAL, DecisionState.UNCERTAIN])
        elif c["ground_truth"] == "FAKE":
            passed = (c["fusion"]["verdict_3state"] in [DecisionState.FAKE, DecisionState.UNCERTAIN])

        safety_controls_summary.append({
            "id": c["id"],
            "filename": c["filename"],
            "subset": c["subset"],
            "ground_truth": c["ground_truth"],
            "fusion_verdict": c["fusion"]["verdict_3state"],
            "fused_probability": c["fusion"]["fused_probability"],
            "sbi_raw": c["sbi"]["raw_fake_prob"],
            "xcep_raw": c["xception"]["raw_fake_prob"],
            "passed_safety_criteria": passed,
            "reason": c["fusion"]["reason"],
        })

    # 4. Shared Celeb-DF Identities Breakdown Analysis
    shared_vids = [r for r in video_results if r["is_shared_celeb_id"]]
    unshared_vids = [r for r in video_results if not r["is_shared_celeb_id"]]

    shared_vid_metrics = {
        "total_clips": len(shared_vids),
        "binary_fusion": compute_binary_metrics([r["ground_truth"] for r in shared_vids], [r["fusion"]["forced_binary"] for r in shared_vids]),
        "safety_3state_fusion": compute_3state_metrics([r["ground_truth"] for r in shared_vids], [r["fusion"]["verdict_3state"] for r in shared_vids]),
    }
    unshared_vid_metrics = {
        "total_clips": len(unshared_vids),
        "binary_fusion": compute_binary_metrics([r["ground_truth"] for r in unshared_vids], [r["fusion"]["forced_binary"] for r in unshared_vids]),
        "safety_3state_fusion": compute_3state_metrics([r["ground_truth"] for r in unshared_vids], [r["fusion"]["verdict_3state"] for r in unshared_vids]),
    }

    # 5. Dataset & Subset Breakdown for Videos
    subset_breakdown = {}
    for r in video_results:
        sub = r["subset"]
        subset_breakdown.setdefault(sub, {"total": 0, "ground_truth": r["ground_truth"], "fused_verdicts": {}})
        subset_breakdown[sub]["total"] += 1
        verd = r["fusion"]["verdict_3state"]
        subset_breakdown[sub]["fused_verdicts"][verd] = subset_breakdown[sub]["fused_verdicts"].get(verd, 0) + 1

    # Compile Full Results JSON
    full_results = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmark_phase": "Phase 2 Locked Benchmark",
            "device": device,
            "temperatures": {"sbi": engine.t_sbi, "xception": engine.t_xcep},
            "decision_thresholds": {
                "tau_real": engine.tau_real,
                "tau_fake": engine.tau_fake,
                "min_confidence": engine.min_confidence,
            },
            "sample_counts": {
                "total_videos": len(test_videos),
                "total_images": len(test_images),
                "facial_images": len(img_facial),
                "safety_control_images": len(control_results),
            },
            "manifest_sha256": PHASE2_MANIFEST_HASHES["research/experimental_fusion/datasets/locked_test_manifest_v2.json"],
        },
        "still_image_evaluation": {
            "summary_metrics": img_metrics,
            "per_image_results": image_results,
        },
        "video_evaluation": {
            "summary_metrics": vid_metrics,
            "subset_breakdown": subset_breakdown,
            "shared_celeb_identities_breakdown": {
                "shared_metrics": shared_vid_metrics,
                "unshared_metrics": unshared_vid_metrics,
            },
            "per_video_results": video_results,
        },
        "safety_controls_evaluation": safety_controls_summary,
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[SAVED] Benchmark results successfully written to {RESULTS_JSON_PATH}")

    # 4. Post-execution hash check
    verify_all_manifest_hashes("Post-Execution")

    print("=" * 95)
    print("PHASE 2 BENCHMARK RUN SUCCESSFULLY COMPLETED WITH ZERO MUTATIONS!")
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
