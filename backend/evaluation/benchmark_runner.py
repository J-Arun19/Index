"""
SatyaDrishti -- Phase 4 Benchmark Runner
"""
import os, sys, json, time, random, argparse, datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.utils.logger import setup_logger
logger = setup_logger("BenchmarkRunner")

DATASET_ROOT = os.environ.get("SATYADRISHTI_DATASET_ROOT", r"D:\Download\SatyaDrishti_Datasets")

IMAGE_SOURCES = [
    (os.path.join(DATASET_ROOT, "Faceforensic++", "extracted_frames", "real"), 0, "FF++_real"),
    (os.path.join(DATASET_ROOT, "extracted_frames", "fake"), 1, "Diverse_fake_frames"),
]


VIDEO_SOURCES = [
    (os.path.join(DATASET_ROOT, "Celeb DF", "Celeb-real"), 0, "CelebDF_real"),
    (os.path.join(DATASET_ROOT, "Celeb DF", "YouTube-real"), 0, "YouTube_real"),
    (os.path.join(DATASET_ROOT, "Celeb DF", "Celeb-synthesis"), 1, "CelebDF_fake"),
    (os.path.join(DATASET_ROOT, "FaceForensics++_C23", "original"), 0, "FF++_C23_original"),
    (os.path.join(DATASET_ROOT, "FaceForensics++_C23", "Deepfakes"), 1, "FF++_C23_Deepfakes"),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_files(directory, extensions, label, source_name):
    if not os.path.isdir(directory):
        return []
    records = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in extensions:
                records.append({
                    "file_path": os.path.join(root, f),
                    "ground_truth": label,
                    "source": source_name,
                    "filename": f,
                })
    return records


def stratified_sample(records, n, seed=42):
    rng = random.Random(seed)
    real = [r for r in records if r["ground_truth"] == 0]
    fake = [r for r in records if r["ground_truth"] == 1]
    n_each = n // 2
    n_real = min(n_each, len(real))
    n_fake = min(n - n_real, len(fake))
    n_real = min(n - n_fake, len(real))
    sampled = rng.sample(real, n_real) + rng.sample(fake, n_fake)
    rng.shuffle(sampled)
    return sampled


from backend.core.decision_system import PresentationDecisionPolicy


def decision_to_canonical(prediction):
    return PresentationDecisionPolicy.normalize_verdict(prediction)


def compute_metrics(ledger):
    not_suitable = [r for r in ledger if r["error_category"] == "NOT_SUITABLE"]
    facial = [r for r in ledger if r["error_category"] not in ("NOT_SUITABLE",)]
    decided = [r for r in facial if r["canonical_decision"] in ("REAL", "FAKE")]
    uncertain = [r for r in facial if r["canonical_decision"] == "UNCERTAIN" and r["error_category"] != "INFERENCE_ERROR"]
    errors = [r for r in facial if r["error_category"] == "INFERENCE_ERROR"]

    tp = sum(1 for r in decided if r["error_category"] == "TP")
    tn = sum(1 for r in decided if r["error_category"] == "TN")
    fp = sum(1 for r in decided if r["error_category"] == "FP")
    fn = sum(1 for r in decided if r["error_category"] == "FN")

    n_decided = len(decided)
    n_facial = len(facial)
    n_uncertain = len(uncertain)
    n_not_suitable = len(not_suitable)
    n_errors = len(errors)

    decided_acc = round((tp + tn) / n_decided * 100, 2) if n_decided > 0 else 0.0
    prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    spec = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
    f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
    cov = round(n_decided / n_facial * 100, 2) if n_facial > 0 else 0.0
    abst = round(n_uncertain / n_facial * 100, 2) if n_facial > 0 else 0.0

    latencies = [r["processing_time"] for r in ledger if r.get("processing_time")]
    avg_lat = round(float(np.mean(latencies)), 3) if latencies else 0.0

    return {
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "decisions": {"decided": n_decided, "uncertain": n_uncertain, "inference_errors": n_errors, "not_suitable": n_not_suitable, "facial_samples": n_facial},
        "metrics": {
            "decided_accuracy_pct": decided_acc,
            "coverage_pct": cov,
            "abstention_rate_pct": abst,
            "precision": prec,
            "recall_fake_tpr": rec,
            "specificity_real_tnr": spec,
            "f1_score": f1,
            "false_negatives_count": fn,
            "false_positives_count": fp,
            "avg_latency_seconds": avg_lat,
        },
        "false_negatives": [r for r in ledger if r["error_category"] == "FN"],
    }


def run_image_benchmark(n_samples=55, seed=42, output_dir=None):
    from backend.agents.image_agent import ImageAgent
    logger.info(f"=== IMAGE BENCHMARK (n={n_samples}) ===")
    agent = ImageAgent()

    all_records = []
    for directory, label, source in IMAGE_SOURCES:
        recs = collect_files(directory, IMAGE_EXTENSIONS, label, source)
        all_records.extend(recs)
        logger.info(f"  {source}: {len(recs)} files (label={label})")

    n_real_avail = sum(1 for r in all_records if r["ground_truth"] == 0)
    n_fake_avail = sum(1 for r in all_records if r["ground_truth"] == 1)
    logger.info(f"Total available: {len(all_records)} (REAL={n_real_avail}, FAKE={n_fake_avail})")

    sample = stratified_sample(all_records, n_samples, seed=seed)
    logger.info(f"Running inference on {len(sample)} samples...")

    ledger = []
    t0 = time.time()

    for i, rec in enumerate(sample):
        fp = rec["file_path"]
        gt = rec["ground_truth"]
        row = {
            "sample_id": i + 1, "file_path": fp, "filename": rec["filename"],
            "ground_truth": gt, "gt_label": "FAKE" if gt == 1 else "REAL",
            "source": rec["source"], "canonical_decision": None, "raw_prediction": None,
            "fake_probability": None, "confidence": None, "is_correct": None,
            "error_category": None, "error_reason": None, "processing_time": None,
        }
        try:
            t1 = time.time()
            result = agent.analyze(fp, require_face=True, selected_providers=[])
            row["processing_time"] = round(time.time() - t1, 3)
            raw = result.get("prediction", "UNCERTAIN")
            can = decision_to_canonical(raw)
            row.update({"raw_prediction": raw, "canonical_decision": can,
                        "fake_probability": result.get("fake_probability"),
                        "confidence": result.get("confidence"),
                        "reliability_warning": result.get("reliability_warning")})
            if can == "NOT_SUITABLE":
                row["error_category"] = "NOT_SUITABLE"
            elif can == "UNCERTAIN":
                row["error_category"] = "ABSTENTION"
                row["error_reason"] = result.get("reason", "Abstained")
            elif can == "REAL" and gt == 0:
                row.update({"is_correct": True, "error_category": "TN"})
            elif can == "FAKE" and gt == 1:
                row.update({"is_correct": True, "error_category": "TP"})
            elif can == "REAL" and gt == 1:
                row.update({"is_correct": False, "error_category": "FN",
                            "error_reason": f"FN: fake_prob={row['fake_probability']}"})
            elif can == "FAKE" and gt == 0:
                row.update({"is_correct": False, "error_category": "FP",
                            "error_reason": f"FP: fake_prob={row['fake_probability']}"})
        except Exception as e:
            row.update({"error_category": "INFERENCE_ERROR", "error_reason": str(e)})
            logger.error(f"Sample {i+1} error: {e}")

        ledger.append(row)
        if (i + 1) % 10 == 0:
            logger.info(f"  [{i+1}/{len(sample)}] done")

    total_time = round(time.time() - t0, 2)
    result_m = compute_metrics(ledger)
    n_real_s = sum(1 for r in ledger if r["ground_truth"] == 0)
    n_fake_s = sum(1 for r in ledger if r["ground_truth"] == 1)

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "image", "seed": seed,
        "dataset_available": {"total": len(all_records), "real": n_real_avail, "fake": n_fake_avail},
        "sample": {"total_sampled": len(sample), "real_sampled": n_real_s, "fake_sampled": n_fake_s},
        **result_m,
        "metrics": {**result_m["metrics"], "total_runtime_seconds": total_time},
        "ledger": ledger,
    }

    out = output_dir or os.path.join(ROOT, "backend", "reports")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "benchmark_image_ledger.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    _print_report("IMAGE", metrics)
    logger.info(f"Ledger saved: {path}")
    return metrics


def run_video_benchmark(n_samples=12, seed=42, output_dir=None):
    from backend.agents.video_agent import VideoAgent
    logger.info(f"=== VIDEO BENCHMARK (n={n_samples}) ===")
    agent = VideoAgent()

    if not os.path.exists(agent.weights_path):
        logger.error(f"Video weights missing: {agent.weights_path}")
        return {"error": "Video weights missing"}

    all_records = []
    for directory, label, source in VIDEO_SOURCES:
        recs = collect_files(directory, VIDEO_EXTENSIONS, label, source)
        all_records.extend(recs)
        logger.info(f"  {source}: {len(recs)} files (label={label})")

    n_real_avail = sum(1 for r in all_records if r["ground_truth"] == 0)
    n_fake_avail = sum(1 for r in all_records if r["ground_truth"] == 1)
    logger.info(f"Total available: {len(all_records)} (REAL={n_real_avail}, FAKE={n_fake_avail})")

    sample = stratified_sample(all_records, n_samples, seed=seed)
    logger.info(f"Running inference on {len(sample)} video samples...")

    ledger = []
    t0 = time.time()

    for i, rec in enumerate(sample):
        fp = rec["file_path"]
        gt = rec["ground_truth"]
        row = {
            "sample_id": i + 1, "file_path": fp, "filename": rec["filename"],
            "ground_truth": gt, "gt_label": "FAKE" if gt == 1 else "REAL",
            "source": rec["source"], "canonical_decision": None, "raw_prediction": None,
            "fake_probability": None, "mean_frame_fake_prob": None, "temporal_variance": None,
            "confidence": None, "is_correct": None,
            "error_category": None, "error_reason": None, "processing_time": None,
        }
        try:
            t1 = time.time()
            result = agent.analyze(fp, max_sample_frames=10, selected_providers=[])
            row["processing_time"] = round(time.time() - t1, 3)
            raw = result.get("prediction", "UNCERTAIN")
            can = decision_to_canonical(raw)
            tm = result.get("temporal_metrics", {})
            row.update({"raw_prediction": raw, "canonical_decision": can,
                        "fake_probability": result.get("fake_probability"),
                        "confidence": result.get("confidence"),
                        "reliability_warning": result.get("reliability_warning"),
                        "mean_frame_fake_prob": tm.get("mean_frame_fake_prob"),
                        "temporal_variance": tm.get("temporal_variance")})
            if can == "NOT_SUITABLE":
                row["error_category"] = "NOT_SUITABLE"
            elif can == "UNCERTAIN":
                row["error_category"] = "ABSTENTION"
                row["error_reason"] = result.get("reason", "Abstained")
            elif can == "REAL" and gt == 0:
                row.update({"is_correct": True, "error_category": "TN"})
            elif can == "FAKE" and gt == 1:
                row.update({"is_correct": True, "error_category": "TP"})
            elif can == "REAL" and gt == 1:
                row.update({"is_correct": False, "error_category": "FN",
                            "error_reason": f"FN: fake_prob={row['fake_probability']}"})
            elif can == "FAKE" and gt == 0:
                row.update({"is_correct": False, "error_category": "FP",
                            "error_reason": f"FP: fake_prob={row['fake_probability']}"})
        except Exception as e:
            row.update({"error_category": "INFERENCE_ERROR", "error_reason": str(e)})
            logger.error(f"Video {i+1} error: {e}")

        ledger.append(row)
        logger.info(f"  [{i+1}/{len(sample)}] {rec['filename']} GT={row['gt_label']} -> {row['canonical_decision']} correct={row['is_correct']}")

    total_time = round(time.time() - t0, 2)
    result_m = compute_metrics(ledger)
    n_real_s = sum(1 for r in ledger if r["ground_truth"] == 0)
    n_fake_s = sum(1 for r in ledger if r["ground_truth"] == 1)

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": "video", "seed": seed,
        "dataset_available": {"total": len(all_records), "real": n_real_avail, "fake": n_fake_avail},
        "sample": {"total_sampled": len(sample), "real_sampled": n_real_s, "fake_sampled": n_fake_s},
        **result_m,
        "metrics": {**result_m["metrics"], "total_runtime_seconds": total_time},
        "ledger": ledger,
    }

    out = output_dir or os.path.join(ROOT, "backend", "reports")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "benchmark_video_ledger.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    _print_report("VIDEO", metrics)
    logger.info(f"Ledger saved: {path}")
    return metrics


def _print_report(label, m):
    cm = m["confusion_matrix"]
    met = m["metrics"]
    dec = m["decisions"]
    print(f"\n{'='*60}")
    print(f"{label} BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Dataset available: {m['dataset_available']['total']} (REAL={m['dataset_available']['real']}, FAKE={m['dataset_available']['fake']})")
    print(f"Sampled:           {m['sample']['total_sampled']} (REAL={m['sample']['real_sampled']}, FAKE={m['sample']['fake_sampled']})")
    print(f"Facial samples:    {dec.get('facial_samples', '?')}")
    print(f"Non-face controls: {dec.get('not_suitable', 0)}")
    print()
    print(f"Decided:           {dec['decided']}")
    print(f"Uncertain:         {dec['uncertain']}")
    print(f"Inference errors:  {dec['inference_errors']}")
    print()
    print(f"Confusion Matrix (decided):  TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")
    print()
    print(f"Decided accuracy:  {met['decided_accuracy_pct']}%  (n={dec['decided']})")
    print(f"Coverage:          {met['coverage_pct']}%")
    print(f"Abstention rate:   {met['abstention_rate_pct']}%")
    print(f"Precision:         {met['precision']:.4f}")
    print(f"Recall (TPR):      {met['recall_fake_tpr']:.4f}")
    print(f"Specificity (TNR): {met['specificity_real_tnr']:.4f}")
    print(f"F1 Score:          {met['f1_score']:.4f}")
    print(f"False Negatives:   {met['false_negatives_count']}")
    print(f"False Positives:   {met['false_positives_count']}")
    print(f"Avg latency:       {met['avg_latency_seconds']}s/sample")
    print(f"Total runtime:     {met['total_runtime_seconds']}s")
    if m.get("false_negatives"):
        print(f"\nFalse Negatives (FAKE classified as REAL):")
        for fn in m["false_negatives"]:
            print(f"  Sample {fn['sample_id']}: {fn['filename']} (src={fn['source']}, fake_prob={fn['fake_probability']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "video", "all"], default="all")
    parser.add_argument("--n_img", type=int, default=55)
    parser.add_argument("--n_vid", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    if args.mode in ("image", "all"):
        run_image_benchmark(n_samples=args.n_img, seed=args.seed, output_dir=args.output_dir)
    if args.mode in ("video", "all"):
        run_video_benchmark(n_samples=args.n_vid, seed=args.seed, output_dir=args.output_dir)
