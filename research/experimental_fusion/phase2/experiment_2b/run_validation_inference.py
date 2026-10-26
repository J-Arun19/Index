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

logger = setup_logger("ValidationInferenceCPU")

VAL_MANIFEST_PATH = "research/experimental_fusion/datasets/validation_manifest_v2.json"
OUTPUT_DIR = "research/experimental_fusion/phase2/experiment_2b"
PREDICTIONS_JSON_PATH = os.path.join(OUTPUT_DIR, "validation_sample_predictions.json")

SBI_WEIGHTS_PATH = "backend/models/experimental/weights/sbi_ffc23.tar"
XCEP_WEIGHTS_PATH = "backend/models/experimental/weights/xception_best.pth"
CALIBRATION_CONFIG_PATH = "backend/models/experimental/fusion_calibration_config.json"

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

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

def run_cpu_validation():
    print("=" * 90)
    print("SATYADRISHTI EXPERIMENT 2B: VALIDATION INFERENCE (CPU-ONLY)")
    print("Generating Missing Sample-Level Validation Predictions")
    print("=" * 90)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Preflight Verification
    print("\n--- Step 1: Preflight Verification ---")
    manifest_hash = sha256_file(VAL_MANIFEST_PATH)
    sbi_hash = sha256_file(SBI_WEIGHTS_PATH)
    xcep_hash = sha256_file(XCEP_WEIGHTS_PATH)
    cal_hash = sha256_file(CALIBRATION_CONFIG_PATH)

    print(f"Validation Manifest: {VAL_MANIFEST_PATH} (SHA256: {manifest_hash})")
    print(f"SBI Weights:         {SBI_WEIGHTS_PATH} (SHA256: {sbi_hash})")
    print(f"Xception Weights:    {XCEP_WEIGHTS_PATH} (SHA256: {xcep_hash})")
    print(f"Calibration Config:  {CALIBRATION_CONFIG_PATH} (SHA256: {cal_hash})")

    with open(VAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    val_videos = manifest_data["videos"]
    val_images = manifest_data["images"]
    print(f"Loaded {len(val_videos)} validation videos and {len(val_images)} validation images.")

    # Confirm CPU Device
    device = "cpu"
    print(f"Executing STRICTLY on CPU (device='{device}')")

    # Load Models on CPU
    t0_load = time.time()
    sbi_detector = SBIDetector(weights_path=SBI_WEIGHTS_PATH, device=device)
    xcep_detector = XceptionDetector(weights_path=XCEP_WEIGHTS_PATH, device=device)
    face_detector = FaceDetector()
    quality_gate = ImageQualityGate()
    forensics = ForensicEvidenceSuite()

    # Load Calibrated Temperatures
    with open(CALIBRATION_CONFIG_PATH, "r", encoding="utf-8") as f:
        cal_cfg = json.load(f)
    t_sbi = float(cal_cfg["sbi"]["temperature"])
    t_xcep = float(cal_cfg["xception"]["temperature"])
    tau_real = float(cal_cfg["decision_thresholds"]["tau_real"])
    tau_fake = float(cal_cfg["decision_thresholds"]["tau_fake"])
    min_conf = float(cal_cfg["decision_thresholds"]["min_confidence"])

    engine = ForensicFusionEngine(
        t_sbi=t_sbi,
        t_xcep=t_xcep,
        tau_real=tau_real,
        tau_fake=tau_fake,
        min_confidence=min_conf,
    )
    print(f"ForensicFusionEngine initialized on CPU: T_sbi={engine.t_sbi}, T_xcep={engine.t_xcep}, tau_real={engine.tau_real}, tau_fake={engine.tau_fake}")
    print(f"Model initialization complete in {time.time() - t0_load:.2f}s.")

    # 2. Validation Video Inference (60 clips)
    print("\n--- Step 2A: Validation Video Inference (60 clips, 8 frames each) ---")
    video_predictions = []
    t0_vids = time.time()

    for idx, item in enumerate(val_videos, 1):
        vid_id = item["id"]
        fn = item["filename"]
        vp = item["filepath"]
        dataset = item["dataset"]
        subset = item["subset"]
        gt = item["ground_truth"]

        t_start = time.time()
        status = "SUCCESS"

        if not os.path.exists(vp):
            video_predictions.append({
                "id": vid_id, "filename": fn, "filepath": vp, "dataset": dataset, "subset": subset, "ground_truth": gt,
                "inference_status": "FAILED", "error": "File not found", "latency_seconds": round(time.time() - t_start, 4)
            })
            continue

        cap = cv2.VideoCapture(vp)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            sample_indices = [0]
        else:
            num_samples = min(8, total_frames)
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

        frames = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append(frame)
        cap.release()

        if not frames:
            video_predictions.append({
                "id": vid_id, "filename": fn, "filepath": vp, "dataset": dataset, "subset": subset, "ground_truth": gt,
                "inference_status": "FAILED", "error": "No decodable frames", "latency_seconds": round(time.time() - t_start, 4)
            })
            continue

        per_frame_scores = []
        sbi_probs = []
        xcep_probs = []
        usable_faces = 0

        for f_i, frame in enumerate(frames):
            faces = face_detector.detect_faces(frame)
            if faces:
                usable_faces += 1
                bbox = faces[0]
                s_crop = extract_sbi_crop(frame, bbox, margin_ratio=0.25)
                s_out = sbi_detector.predict_crop(s_crop)
                s_prob = float(s_out["fake_probability"])

                x_crop = face_detector.crop_face(frame, bbox)
                x_out = xcep_detector.predict_frame(x_crop)
                x_prob = float(x_out["fake_probability"])
                face_found = True
            else:
                s_out = sbi_detector.predict_crop(frame)
                s_prob = float(s_out["fake_probability"])
                x_out = xcep_detector.predict_frame(frame)
                x_prob = float(x_out["fake_probability"])
                face_found = False

            sbi_probs.append(s_prob)
            xcep_probs.append(x_prob)
            per_frame_scores.append({
                "frame_idx": f_i,
                "face_detected": face_found,
                "sbi_raw": round(s_prob, 4),
                "xcep_raw": round(x_prob, 4),
            })

        mean_sbi = float(np.mean(sbi_probs))
        sbi_var = float(np.std(sbi_probs))
        sbi_conflict = bool((max(sbi_probs) > 0.70 and min(sbi_probs) < 0.25) or (sbi_var > 0.22))
        cal_sbi = engine.calibrate(mean_sbi, temperature=engine.t_sbi)

        mean_xcep = float(np.mean(xcep_probs))
        xcep_var = float(np.std(xcep_probs))
        xcep_conflict = bool((max(xcep_probs) > 0.70 and min(xcep_probs) < 0.25) or (xcep_var > 0.22))
        cal_xcep = engine.calibrate(mean_xcep, temperature=engine.t_xcep)

        temp_var = max(sbi_var, xcep_var)
        temp_conflict = sbi_conflict or xcep_conflict

        mid_frame = frames[len(frames) // 2]
        forensic_res = forensics.evaluate(mid_frame)

        fusion_dec = engine.evaluate(
            raw_sbi=mean_sbi,
            raw_xcep=mean_xcep,
            temporal_variance=temp_var,
            temporal_conflict=temp_conflict,
            forensic_result=forensic_res,
        )

        p_fused = fusion_dec.fake_probability
        forced_b = "FAKE" if (p_fused or 0.5) >= 0.50 else "REAL"
        latency = round(time.time() - t_start, 4)

        video_record = {
            "id": vid_id,
            "filename": fn,
            "filepath": vp,
            "dataset": dataset,
            "subset": subset,
            "ground_truth": gt,
            "sampled_frames_count": len(frames),
            "faces_detected_count": usable_faces,
            "per_frame_scores": per_frame_scores,
            "sbi": {
                "raw_fake_prob": round(mean_sbi, 4),
                "calibrated_prob": round(cal_sbi, 4) if cal_sbi is not None else None,
                "forced_binary": "FAKE" if mean_sbi >= 0.50 else "REAL",
                "verdict_3state": "FAKE" if (cal_sbi or 0.5) > 0.65 else ("LIKELY_REAL" if (cal_sbi or 0.5) < 0.35 else "UNCERTAIN"),
                "variance": round(sbi_var, 4),
                "conflict": sbi_conflict,
            },
            "xception": {
                "raw_fake_prob": round(mean_xcep, 4),
                "calibrated_prob": round(cal_xcep, 4) if cal_xcep is not None else None,
                "forced_binary": "FAKE" if mean_xcep >= 0.50 else "REAL",
                "verdict_3state": "FAKE" if (cal_xcep or 0.5) > 0.65 else ("LIKELY_REAL" if (cal_xcep or 0.5) < 0.35 else "UNCERTAIN"),
                "variance": round(xcep_var, 4),
                "conflict": xcep_conflict,
            },
            "temporal_metrics": {
                "sbi_variance": round(sbi_var, 4),
                "xcep_variance": round(xcep_var, 4),
                "max_temporal_variance": round(temp_var, 4),
                "temporal_conflict": temp_conflict,
            },
            "fusion": {
                "fused_probability": round(p_fused, 4) if p_fused is not None else None,
                "confidence": fusion_dec.confidence,
                "uncertainty": fusion_dec.uncertainty,
                "baseline_verdict_3state": fusion_dec.prediction,
                "baseline_forced_binary": forced_b,
                "reason": fusion_dec.reason,
            },
            "inference_status": status,
            "latency_seconds": latency,
        }
        video_predictions.append(video_record)

        if idx % 10 == 0 or idx == len(val_videos):
            print(f"  [Video {idx:02d}/60] {fn[:22]} | GT: {gt:4s} | Fused: {fusion_dec.prediction:11s} (P={p_fused}) | {latency:.2f}s")

    t_vids_elapsed = round(time.time() - t0_vids, 2)
    print(f"Completed Validation Video Inference in {t_vids_elapsed}s.")

    # 3. Validation Image Inference (60 images)
    print("\n--- Step 2B: Validation Image Inference (60 middle frames) ---")
    image_predictions = []
    t0_imgs = time.time()

    for idx, item in enumerate(val_images, 1):
        img_id = item["id"]
        src_vid_id = item["source_video_id"]
        src_path = item["source_video_path"]
        src_fn = item["source_filename"]
        dataset = item["dataset"]
        subset = item["subset"]
        gt = item["ground_truth"]

        t_start = time.time()
        status = "SUCCESS"

        if not os.path.exists(src_path):
            image_predictions.append({
                "id": img_id, "source_video_id": src_vid_id, "source_video_path": src_path, "source_filename": src_fn,
                "dataset": dataset, "subset": subset, "ground_truth": gt, "inference_status": "FAILED", "error": "File not found",
                "latency_seconds": round(time.time() - t_start, 4)
            })
            continue

        cap = cv2.VideoCapture(src_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_idx = max(0, frame_count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            image_predictions.append({
                "id": img_id, "source_video_id": src_vid_id, "source_video_path": src_path, "source_filename": src_fn,
                "dataset": dataset, "subset": subset, "ground_truth": gt, "inference_status": "FAILED", "error": "Could not extract middle frame",
                "latency_seconds": round(time.time() - t_start, 4)
            })
            continue

        q_res = quality_gate.evaluate(frame_bgr, require_face=False)
        faces = face_detector.detect_faces(frame_bgr)
        face_found = len(faces) > 0

        if not face_found:
            q_res_unsupp = type("QualityDummy", (), {
                "eligibility": "UNSUPPORTED", "is_suitable": False, "face_detected": False,
                "rejection_reason": "No human face found in image.", "issues": ["UNSUPPORTED_MEDIA"],
                "to_dict": lambda s: {"eligibility": "UNSUPPORTED", "face_detected": False}
            })()
            fusion_dec = engine.evaluate(raw_sbi=None, raw_xcep=None, quality_result=q_res_unsupp, forensic_result={})
            raw_sbi, cal_sbi = None, None
            raw_xcep, cal_xcep = None, None
            p_fused = None
            forced_b = "NOT_SUITABLE"
            status = "NOT_SUITABLE"
        else:
            primary_bbox = faces[0]
            forensic_res = forensics.evaluate(frame_bgr, face_bbox=primary_bbox)

            s_crop = extract_sbi_crop(frame_bgr, primary_bbox, margin_ratio=0.25)
            s_out = sbi_detector.predict_crop(s_crop)
            raw_sbi = float(s_out["fake_probability"])
            cal_sbi = engine.calibrate(raw_sbi, temperature=engine.t_sbi)

            x_crop = face_detector.crop_face(frame_bgr, primary_bbox)
            x_out = xcep_detector.predict_frame(x_crop)
            raw_xcep = float(x_out["fake_probability"])
            cal_xcep = engine.calibrate(raw_xcep, temperature=engine.t_xcep)

            fusion_dec = engine.evaluate(raw_sbi=raw_sbi, raw_xcep=raw_xcep, quality_result=q_res, forensic_result=forensic_res)
            p_fused = fusion_dec.fake_probability
            forced_b = "FAKE" if (p_fused or 0.5) >= 0.50 else "REAL"

        latency = round(time.time() - t_start, 4)

        image_record = {
            "id": img_id,
            "source_video_id": src_vid_id,
            "source_video_path": src_path,
            "source_filename": src_fn,
            "dataset": dataset,
            "subset": subset,
            "ground_truth": gt,
            "face_detected": face_found,
            "sbi": {
                "raw_fake_prob": round(raw_sbi, 4) if raw_sbi is not None else None,
                "calibrated_prob": round(cal_sbi, 4) if cal_sbi is not None else None,
                "forced_binary": ("FAKE" if raw_sbi >= 0.50 else "REAL") if raw_sbi is not None else "NOT_SUITABLE",
                "verdict_3state": ("FAKE" if cal_sbi > 0.65 else ("LIKELY_REAL" if cal_sbi < 0.35 else "UNCERTAIN")) if cal_sbi is not None else "NOT_SUITABLE",
            },
            "xception": {
                "raw_fake_prob": round(raw_xcep, 4) if raw_xcep is not None else None,
                "calibrated_prob": round(cal_xcep, 4) if cal_xcep is not None else None,
                "forced_binary": ("FAKE" if raw_xcep >= 0.50 else "REAL") if raw_xcep is not None else "NOT_SUITABLE",
                "verdict_3state": ("FAKE" if cal_xcep > 0.65 else ("LIKELY_REAL" if cal_xcep < 0.35 else "UNCERTAIN")) if cal_xcep is not None else "NOT_SUITABLE",
            },
            "fusion": {
                "fused_probability": round(p_fused, 4) if p_fused is not None else None,
                "confidence": fusion_dec.confidence,
                "uncertainty": fusion_dec.uncertainty,
                "baseline_verdict_3state": fusion_dec.prediction,
                "baseline_forced_binary": forced_b,
                "reason": fusion_dec.reason,
            },
            "inference_status": status,
            "latency_seconds": latency,
        }
        image_predictions.append(image_record)

        if idx % 10 == 0 or idx == len(val_images):
            print(f"  [Image {idx:02d}/60] {src_fn[:22]} | GT: {gt:4s} | Fused: {fusion_dec.prediction:11s} (P={p_fused}) | {latency:.2f}s")

    t_imgs_elapsed = round(time.time() - t0_imgs, 2)
    print(f"Completed Validation Image Inference in {t_imgs_elapsed}s.")

    # 4. Save Validation Sample Predictions JSON
    output_payload = {
        "validation_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "manifest_path": VAL_MANIFEST_PATH,
            "manifest_sha256": manifest_hash,
            "model_weights": {
                "sbi": {"path": SBI_WEIGHTS_PATH, "sha256": sbi_hash},
                "xception": {"path": XCEP_WEIGHTS_PATH, "sha256": xcep_hash},
            },
            "calibration_config": {
                "path": CALIBRATION_CONFIG_PATH,
                "sha256": cal_hash,
                "temperatures": {"sbi": t_sbi, "xception": t_xcep},
                "thresholds": {"tau_real": tau_real, "tau_fake": tau_fake, "min_confidence": min_conf},
            },
            "sample_counts": {
                "total_videos": len(val_videos),
                "total_images": len(val_images),
                "video_failures": sum(1 for v in video_predictions if v["inference_status"] == "FAILED"),
                "image_failures": sum(1 for img in image_predictions if img["inference_status"] == "FAILED"),
                "image_not_suitable": sum(1 for img in image_predictions if img["inference_status"] == "NOT_SUITABLE"),
            },
            "runtime_seconds": {
                "video_inference_total": t_vids_elapsed,
                "image_inference_total": t_imgs_elapsed,
                "total_elapsed": round(t_vids_elapsed + t_imgs_elapsed, 2),
            }
        },
        "video_predictions": video_predictions,
        "image_predictions": image_predictions,
    }

    with open(PREDICTIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[+] Successfully saved sample-level validation predictions to {PREDICTIONS_JSON_PATH}")

if __name__ == "__main__":
    run_cpu_validation()
