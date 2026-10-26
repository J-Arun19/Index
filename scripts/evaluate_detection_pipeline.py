import os
import sys
import json
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.image_agent import ImageAgent
from backend.core.decision_system import DecisionState

def evaluate_dataset(manifest_path: str = "scratch/image_eval_manifest.json") -> Dict[str, Any]:
    """
    Evaluates the current SatyaDrishti multi-model detection pipeline
    against the labeled evaluation set with separated task categories:
      - ai_generated
      - face_swap_or_manipulation
      - genuine
      - unsupported / no-face
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("=" * 110)
    print(f"SATYADRISHTI INDEPENDENT PIPELINE EVALUATION ({len(samples)} samples)")
    print("=" * 110)

    agent = ImageAgent()

    results = []
    category_metrics = {}

    for idx, sample in enumerate(samples, 1):
        rel_path = sample.get("path")
        gt = sample.get("ground_truth", "UNKNOWN")
        cat = sample.get("category", "unspecified")
        fn = sample.get("filename", os.path.basename(rel_path))

        # Categorize task
        if "synth" in cat or "synth" in fn or fn in ("inwild_picture.jpeg", "inwild_professional.jpeg", "fake2.png"):
            task_domain = "ai_generated"
        elif "face" in cat or "deepfake" in cat or "swap" in cat or "texture" in cat or fn in ("inwild_deepy.jpeg",):
            task_domain = "face_swap_or_manipulation"
        elif "document" in fn or "logo" in fn or "unsupported" in cat:
            task_domain = "unsupported_media"
        else:
            task_domain = "genuine_media"

        full_path = os.path.abspath(rel_path)
        if not os.path.exists(full_path):
            print(f"[{idx}/{len(samples)}] File missing: {rel_path}")
            continue

        start_t = time.time()
        try:
            res = agent.analyze(full_path, selected_providers=[])
            latency = time.time() - start_t
        except Exception as e:
            print(f"[{idx}/{len(samples)}] Execution error on {fn}: {e}")
            res = {
                "verdict": DecisionState.UNCERTAIN,
                "canonical_verdict": DecisionState.UNCERTAIN,
                "confidence_level": "Failed",
                "fake_probability": None,
                "error": str(e)
            }
            latency = time.time() - start_t

        canonical_v = res.get("canonical_verdict") or res.get("verdict", DecisionState.UNCERTAIN)
        primary_v = res.get("primary_verdict") or res.get("prediction")
        fake_p = res.get("fake_probability")
        features = res.get("features", {})
        raw_xcep = features.get("raw_xcep_score")
        raw_vit = features.get("raw_vit_score")
        raw_res = features.get("raw_resnet_score")

        results.append({
            "filename": fn,
            "path": rel_path,
            "ground_truth": gt,
            "task_domain": task_domain,
            "category": cat,
            "canonical_verdict": canonical_v,
            "primary_verdict": primary_v,
            "fake_probability": fake_p,
            "raw_xcep": raw_xcep,
            "raw_vit": raw_vit,
            "raw_resnet": raw_res,
            "is_provisional": res.get("is_binary_fallback", False),
            "latency_s": round(latency, 3),
        })

        gt_str = f"GT: {gt}"
        v_str = f"Verdict: {canonical_v}"
        print(f"[{idx:02d}/{len(samples):02d}] {fn:<32} | {gt_str:<10} | {v_str:<22} | Xcep: {str(raw_xcep):<5} | ViT: {str(raw_vit):<5} | FProb: {str(fake_p):<5}")

    # Compute comprehensive evaluation metrics
    def compute_group_metrics(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(subset)
        if total == 0:
            return {}

        tp = sum(1 for s in subset if s["ground_truth"] == "FAKE" and s["canonical_verdict"] == "FAKE")
        tn = sum(1 for s in subset if s["ground_truth"] == "REAL" and s["canonical_verdict"] == "REAL")
        fp = sum(1 for s in subset if s["ground_truth"] == "REAL" and s["canonical_verdict"] == "FAKE")
        fn = sum(1 for s in subset if s["ground_truth"] == "FAKE" and s["canonical_verdict"] == "REAL")
        abstentions = sum(1 for s in subset if s["canonical_verdict"] in ("UNCERTAIN", "NOT_SUITABLE"))

        decided = tp + tn + fp + fn
        decided_acc = round((tp + tn) / decided, 4) if decided > 0 else 0.0
        coverage = round(decided / total, 4)
        abstention_rate = round(abstentions / total, 4)

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        specificity = round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

        return {
            "total_samples": total,
            "decided_samples": decided,
            "abstentions": abstentions,
            "coverage_rate": coverage,
            "abstention_rate": abstention_rate,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "decided_accuracy": decided_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "fpr": fpr,
            "fnr": fnr,
            "f1_score": f1,
        }

    overall_metrics = compute_group_metrics(results)
    ai_gen_metrics = compute_group_metrics([r for r in results if r["task_domain"] == "ai_generated"])
    manip_metrics = compute_group_metrics([r for r in results if r["task_domain"] == "face_swap_or_manipulation"])
    genuine_metrics = compute_group_metrics([r for r in results if r["task_domain"] == "genuine_media"])

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": len(results),
        "overall_metrics": overall_metrics,
        "task_breakdown": {
            "ai_generated": ai_gen_metrics,
            "face_swap_or_manipulation": manip_metrics,
            "genuine_media": genuine_metrics,
        },
        "individual_results": results,
    }

    os.makedirs("research/results", exist_ok=True)
    report_file = "research/results/evaluation_metrics_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 110)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 110)
    print(f"Total Evaluated: {len(results)}")
    print(f"Overall Decided Accuracy: {overall_metrics.get('decided_accuracy', 0):.1%} (on decided samples)")
    print(f"Abstention / Uncertain Rate: {overall_metrics.get('abstention_rate', 0):.1%} ({overall_metrics.get('abstentions', 0)} samples held out as UNCERTAIN / NOT_SUITABLE)")
    print(f"Decided Precision (FAKE): {overall_metrics.get('precision', 0):.1%}")
    print(f"Decided Recall (FAKE):    {overall_metrics.get('recall', 0):.1%}")
    print(f"Decided F1 Score:         {overall_metrics.get('f1_score', 0):.4f}")
    print(f"Decided False Positive Rate (FPR): {overall_metrics.get('fpr', 0):.1%}")
    print(f"Decided False Negative Rate (FNR): {overall_metrics.get('fnr', 0):.1%}")
    print("-" * 110)
    print(f"Saved full report to {report_file}")

    return report

if __name__ == "__main__":
    evaluate_dataset()
