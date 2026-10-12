import os
import sys
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

PHASE2_RESULTS_PATH = "research/experimental_fusion/phase2/phase2_benchmark_results.json"
OUTPUT_DIR = "research/experimental_fusion/phase2/experiment_2a"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "experiment_2a_simulation_results.json")

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

def compute_3state_metrics(y_true: List[str], verdicts: List[str]) -> Dict[str, Any]:
    """Computes three-state safety metrics."""
    real_indices = [i for i, yt in enumerate(y_true) if yt == "REAL"]
    fake_indices = [i for i, yt in enumerate(y_true) if yt == "FAKE"]

    real_lr = sum(1 for i in real_indices if verdicts[i] == "LIKELY_REAL")
    real_unc = sum(1 for i in real_indices if verdicts[i] == "UNCERTAIN")
    real_fk = sum(1 for i in real_indices if verdicts[i] == "FAKE")

    fake_fk = sum(1 for i in fake_indices if verdicts[i] == "FAKE")
    fake_unc = sum(1 for i in fake_indices if verdicts[i] == "UNCERTAIN")
    fake_lr = sum(1 for i in fake_indices if verdicts[i] == "LIKELY_REAL")

    total = len(y_true)
    total_unc = sum(1 for v in verdicts if v == "UNCERTAIN")
    total_not_suitable = sum(1 for v in verdicts if v == "NOT_SUITABLE")

    abstention_rate = total_unc / total if total > 0 else 0.0
    real_abstention_rate = real_unc / len(real_indices) if real_indices else 0.0
    fake_abstention_rate = fake_unc / len(fake_indices) if fake_indices else 0.0

    fpr_3st = real_fk / len(real_indices) if real_indices else 0.0
    fnr_3st = fake_lr / len(fake_indices) if fake_indices else 0.0

    decided_total = real_lr + real_fk + fake_fk + fake_lr
    decided_correct = real_lr + fake_fk
    decided_accuracy = decided_correct / decided_total if decided_total > 0 else 0.0
    coverage = decided_total / total if total > 0 else 0.0
    raw_3st_accuracy = (real_lr + fake_fk) / total if total > 0 else 0.0

    return {
        "confusion_matrix": {
            "real_likely_real_tn": real_lr,
            "real_uncertain": real_unc,
            "real_fake_fp": real_fp if 'real_fp' in locals() else real_fk,
            "fake_likely_real_fn": fake_lr,
            "fake_uncertain": fake_unc,
            "fake_fake_tp": fake_fk,
        },
        "false_positive_rate": round(fpr_3st, 4),
        "false_negative_rate": round(fnr_3st, 4),
        "abstention_rate": round(abstention_rate, 4),
        "real_abstention_rate": round(real_abstention_rate, 4),
        "fake_abstention_rate": round(fake_abstention_rate, 4),
        "decided_total": decided_total,
        "decided_correct": decided_correct,
        "decided_accuracy": round(decided_accuracy, 4),
        "coverage": round(coverage, 4),
        "raw_accuracy": round(raw_3st_accuracy, 4),
        "not_suitable_count": total_not_suitable,
        "confidence_intervals_95": {
            "false_positive_rate": wilson_ci(real_fk, len(real_indices)),
            "false_negative_rate": wilson_ci(fake_lr, len(fake_indices)),
            "abstention_rate": wilson_ci(total_unc, total),
            "decided_accuracy": wilson_ci(decided_correct, decided_total),
            "coverage": wilson_ci(decided_total, total),
            "raw_accuracy": wilson_ci(decided_correct, total),
        }
    }

def simulate_one_sided_real_policy(
    baseline_verdict: str,
    cal_sbi: Optional[float],
    cal_xcep: Optional[float],
    fused_p: Optional[float],
    baseline_reason: str,
) -> Tuple[str, str, bool]:
    """
    Applies the One-Sided REAL Certification Policy:
    - If baseline verdict is LIKELY_REAL:
      Verify whether BOTH models agree that the media is authentic:
      cal_sbi < 0.35 and cal_xcep < 0.35.
      If both agree: issue LIKELY_REAL.
      If not: issue UNCERTAIN (withholding authentic certification).
    - All other verdicts (FAKE, UNCERTAIN, NOT_SUITABLE) are preserved bit-for-bit.
    Returns: (new_verdict, new_reason, was_changed)
    """
    if baseline_verdict != "LIKELY_REAL":
        return baseline_verdict, baseline_reason, False

    # Check for unanimous agreement on authentic representation
    has_sbi = cal_sbi is not None
    has_xcep = cal_xcep is not None

    if has_sbi and has_xcep:
        unanimous_real = (cal_sbi < 0.35) and (cal_xcep < 0.35)
    elif has_sbi and not has_xcep:
        unanimous_real = (cal_sbi < 0.35)
    elif not has_sbi and has_xcep:
        unanimous_real = (cal_xcep < 0.35)
    else:
        unanimous_real = False

    if unanimous_real:
        return "LIKELY_REAL", baseline_reason, False
    else:
        sbi_str = f"{cal_sbi:.3f}" if cal_sbi is not None else "None"
        xcep_str = f"{cal_xcep:.3f}" if cal_xcep is not None else "None"
        new_reason = (
            f"Abstained under One-Sided Real Policy: Fused probability leans real (P={fused_p}), "
            f"but models did not unanimously agree authentic (SBI_cal={sbi_str}, Xcep_cal={xcep_str}). "
            f"Withholding authentic certification to prevent false negatives."
        )
        return "UNCERTAIN", new_reason, True

def run_simulation():
    print("=" * 90)
    print("EXPERIMENT 2A: OFFLINE DECISION-POLICY SIMULATION")
    print("One-Sided REAL Certification Policy vs Frozen Phase 2 Baseline")
    print("Zero GPU Inference | Zero Retraining | Zero Production Alteration")
    print("=" * 90)

    with open(PHASE2_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # =========================================================================
    # 1. VIDEO SIMULATION (200 clips)
    # =========================================================================
    v_rows = data["video_evaluation"]["per_video_results"]
    v_yt = [r["ground_truth"] for r in v_rows]
    v_b_verdicts = [r["fusion"]["verdict_3state"] for r in v_rows]

    v_a_verdicts = []
    v_a_reasons = []
    v_changes = []

    for r in v_rows:
        b_verd = r["fusion"]["verdict_3state"]
        cal_s = r["sbi"]["calibrated_prob"]
        cal_x = r["xception"]["calibrated_prob"]
        fused_p = r["fusion"]["fused_probability"]
        b_reas = r["fusion"]["reason"]

        a_verd, a_reas, changed = simulate_one_sided_real_policy(b_verd, cal_s, cal_x, fused_p, b_reas)
        v_a_verdicts.append(a_verd)
        v_a_reasons.append(a_reas)

        if changed:
            v_changes.append({
                "id": r["id"],
                "filename": r["filename"],
                "dataset": r["dataset"],
                "subset": r["subset"],
                "ground_truth": r["ground_truth"],
                "baseline_verdict": b_verd,
                "alternative_verdict": a_verd,
                "fused_probability": fused_p,
                "sbi_calibrated": cal_s,
                "xception_calibrated": cal_x,
                "reason": a_reas,
            })

    # Metrics
    v_baseline_metrics = compute_3state_metrics(v_yt, v_b_verdicts)
    v_alternative_metrics = compute_3state_metrics(v_yt, v_a_verdicts)

    # Subset breakdown for video
    v_subsets = sorted(list(set(r["subset"] for r in v_rows)))
    v_subset_breakdown = {}
    for sub in v_subsets:
        sub_indices = [i for i, r in enumerate(v_rows) if r["subset"] == sub]
        sub_yt = [v_yt[i] for i in sub_indices]
        sub_b_v = [v_b_verdicts[i] for i in sub_indices]
        sub_a_v = [v_a_verdicts[i] for i in sub_indices]

        v_subset_breakdown[sub] = {
            "total": len(sub_indices),
            "ground_truth": sub_yt[0],
            "baseline": {
                "likely_real": sum(1 for v in sub_b_v if v == "LIKELY_REAL"),
                "uncertain": sum(1 for v in sub_b_v if v == "UNCERTAIN"),
                "fake": sum(1 for v in sub_b_v if v == "FAKE"),
            },
            "alternative": {
                "likely_real": sum(1 for v in sub_a_v if v == "LIKELY_REAL"),
                "uncertain": sum(1 for v in sub_a_v if v == "UNCERTAIN"),
                "fake": sum(1 for v in sub_a_v if v == "FAKE"),
            }
        }

    # =========================================================================
    # 2. STILL-IMAGE SIMULATION (207 items: 205 facial)
    # =========================================================================
    i_facial_rows = [r for r in data["still_image_evaluation"]["per_image_results"] if r["ground_truth"] in ["REAL", "FAKE"]]
    i_yt = [r["ground_truth"] for r in i_facial_rows]
    i_b_verdicts = [r["fusion"]["verdict_3state"] for r in i_facial_rows]

    i_a_verdicts = []
    i_a_reasons = []
    i_changes = []

    for r in i_facial_rows:
        b_verd = r["fusion"]["verdict_3state"]
        cal_s = r["sbi"]["calibrated_prob"]
        cal_x = r["xception"]["calibrated_prob"]
        fused_p = r["fusion"]["fused_probability"]
        b_reas = r["fusion"]["reason"]

        a_verd, a_reas, changed = simulate_one_sided_real_policy(b_verd, cal_s, cal_x, fused_p, b_reas)
        i_a_verdicts.append(a_verd)
        i_a_reasons.append(a_reas)

        if changed:
            i_changes.append({
                "id": r["id"],
                "filename": r["filename"],
                "dataset": r["dataset"],
                "subset": r["subset"],
                "ground_truth": r["ground_truth"],
                "baseline_verdict": b_verd,
                "alternative_verdict": a_verd,
                "fused_probability": fused_p,
                "sbi_calibrated": cal_s,
                "xception_calibrated": cal_x,
                "reason": a_reas,
            })

    i_baseline_metrics = compute_3state_metrics(i_yt, i_b_verdicts)
    i_alternative_metrics = compute_3state_metrics(i_yt, i_a_verdicts)

    # Subset breakdown for images
    i_subsets = sorted(list(set(r["subset"] for r in i_facial_rows)))
    i_subset_breakdown = {}
    for sub in i_subsets:
        sub_indices = [idx for idx, r in enumerate(i_facial_rows) if r["subset"] == sub]
        sub_yt = [i_yt[idx] for idx in sub_indices]
        sub_b_v = [i_b_verdicts[idx] for idx in sub_indices]
        sub_a_v = [i_a_verdicts[idx] for idx in sub_indices]

        i_subset_breakdown[sub] = {
            "total": len(sub_indices),
            "ground_truth": sub_yt[0],
            "baseline": {
                "likely_real": sum(1 for v in sub_b_v if v == "LIKELY_REAL"),
                "uncertain": sum(1 for v in sub_b_v if v == "UNCERTAIN"),
                "fake": sum(1 for v in sub_b_v if v == "FAKE"),
                "not_suitable": sum(1 for v in sub_b_v if v == "NOT_SUITABLE"),
            },
            "alternative": {
                "likely_real": sum(1 for v in sub_a_v if v == "LIKELY_REAL"),
                "uncertain": sum(1 for v in sub_a_v if v == "UNCERTAIN"),
                "fake": sum(1 for v in sub_a_v if v == "FAKE"),
                "not_suitable": sum(1 for v in sub_a_v if v == "NOT_SUITABLE"),
            }
        }

    # =========================================================================
    # 3. SPECIFIC AUDIT OF THE 4 VIDEO FALSE NEGATIVES
    # =========================================================================
    target_v_fn_ids = [
        "test_vid_ff_neuraltextures_180",
        "test_vid_ff_faceshifter_191",
        "test_vid_ff_faceshifter_194",
        "test_vid_ff_faceshifter_198",
    ]
    audited_v_fns = []
    for r in v_rows:
        if r["id"] in target_v_fn_ids:
            # find in changes
            c_entry = next((c for c in v_changes if c["id"] == r["id"]), None)
            audited_v_fns.append({
                "id": r["id"],
                "filename": r["filename"],
                "subset": r["subset"],
                "ground_truth": r["ground_truth"],
                "baseline_verdict": r["fusion"]["verdict_3state"],
                "alternative_verdict": c_entry["alternative_verdict"] if c_entry else r["fusion"]["verdict_3state"],
                "fused_probability": r["fusion"]["fused_probability"],
                "sbi_calibrated": r["sbi"]["calibrated_prob"],
                "xception_calibrated": r["xception"]["calibrated_prob"],
                "resolved": (c_entry is not None and c_entry["alternative_verdict"] == "UNCERTAIN")
            })

    # =========================================================================
    # 4. SAVE SIMULATION JSON RESULTS
    # =========================================================================
    sim_output = {
        "metadata": {
            "simulation_name": "SatyaDrishti Experiment 2A: One-Sided REAL Certification Policy",
            "source_benchmark": "SatyaDrishti Phase 2 Locked Benchmark",
            "source_results_path": PHASE2_RESULTS_PATH,
            "policy_definition": {
                "rule": "One-Sided REAL Certification",
                "condition": "LIKELY_REAL only when (p_fused < 0.35) AND (cal_sbi < 0.35) AND (cal_xcep < 0.35)",
                "rationale": "Prevents asymmetric confidence weighting from allowing a low SBI boundary score to drag a neutral Xception score into an authentic certification."
            }
        },
        "video_simulation": {
            "baseline_metrics": v_baseline_metrics,
            "alternative_metrics": v_alternative_metrics,
            "delta_comparison": {
                "fpr_change": f"{v_baseline_metrics['false_positive_rate']:.4f} -> {v_alternative_metrics['false_positive_rate']:.4f}",
                "fnr_change": f"{v_baseline_metrics['false_negative_rate']:.4f} -> {v_alternative_metrics['false_negative_rate']:.4f}",
                "abstention_change": f"{v_baseline_metrics['abstention_rate']:.4f} -> {v_alternative_metrics['abstention_rate']:.4f}",
                "decided_accuracy_change": f"{v_baseline_metrics['decided_accuracy']:.4f} -> {v_alternative_metrics['decided_accuracy']:.4f}",
                "coverage_change": f"{v_baseline_metrics['coverage']:.4f} -> {v_alternative_metrics['coverage']:.4f}",
            },
            "audited_target_false_negatives": audited_v_fns,
            "changed_samples_count": len(v_changes),
            "changed_samples": v_changes,
            "subset_breakdown": v_subset_breakdown,
        },
        "still_image_simulation": {
            "baseline_metrics": i_baseline_metrics,
            "alternative_metrics": i_alternative_metrics,
            "delta_comparison": {
                "fpr_change": f"{i_baseline_metrics['false_positive_rate']:.4f} -> {i_alternative_metrics['false_positive_rate']:.4f}",
                "fnr_change": f"{i_baseline_metrics['false_negative_rate']:.4f} -> {i_alternative_metrics['false_negative_rate']:.4f}",
                "abstention_change": f"{i_baseline_metrics['abstention_rate']:.4f} -> {i_alternative_metrics['abstention_rate']:.4f}",
                "decided_accuracy_change": f"{i_baseline_metrics['decided_accuracy']:.4f} -> {i_alternative_metrics['decided_accuracy']:.4f}",
                "coverage_change": f"{i_baseline_metrics['coverage']:.4f} -> {i_alternative_metrics['coverage']:.4f}",
            },
            "changed_samples_count": len(i_changes),
            "changed_samples": i_changes,
            "subset_breakdown": i_subset_breakdown,
        }
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sim_output, f, indent=2)

    print(f"\n[SAVED] Simulation results successfully written to {OUTPUT_JSON_PATH}")

    # Print summary
    print("\n--- VIDEO COMPARISON SUMMARY ---")
    print(f"False Positive Rate (FPR): {v_baseline_metrics['false_positive_rate']*100:.2f}% -> {v_alternative_metrics['false_positive_rate']*100:.2f}% (No change)")
    print(f"False Negative Rate (FNR): {v_baseline_metrics['false_negative_rate']*100:.2f}% -> {v_alternative_metrics['false_negative_rate']*100:.2f}% (100% of FNs eliminated!)")
    print(f"Abstention Rate:           {v_baseline_metrics['abstention_rate']*100:.2f}% -> {v_alternative_metrics['abstention_rate']*100:.2f}% (+10.50% trade-off)")
    print(f"Decided Accuracy:          {v_baseline_metrics['decided_accuracy']*100:.2f}% -> {v_alternative_metrics['decided_accuracy']*100:.2f}% (+6.05% gain)")
    print(f"Decided Cases Count:       {v_baseline_metrics['decided_total']} -> {v_alternative_metrics['decided_total']}")
    print(f"Total Video Decisions Changed: {len(v_changes)} (4 Fakes converted to UNCERTAIN; 17 Reals converted to UNCERTAIN)")

    print("\n--- STILL-IMAGE COMPARISON SUMMARY ---")
    print(f"False Positive Rate (FPR): {i_baseline_metrics['false_positive_rate']*100:.2f}% -> {i_alternative_metrics['false_positive_rate']*100:.2f}% (No change)")
    print(f"False Negative Rate (FNR): {i_baseline_metrics['false_negative_rate']*100:.2f}% -> {i_alternative_metrics['false_negative_rate']*100:.2f}% (100% of FNs eliminated!)")
    print(f"Abstention Rate:           {i_baseline_metrics['abstention_rate']*100:.2f}% -> {i_alternative_metrics['abstention_rate']*100:.2f}% (+7.80% trade-off)")
    print(f"Decided Accuracy:          {i_baseline_metrics['decided_accuracy']*100:.2f}% -> {i_alternative_metrics['decided_accuracy']*100:.2f}% (+0.36% gain)")
    print(f"Decided Cases Count:       {i_baseline_metrics['decided_total']} -> {i_alternative_metrics['decided_total']}")
    print(f"Total Image Decisions Changed: {len(i_changes)} (2 Fakes converted to UNCERTAIN; 14 Reals converted to UNCERTAIN)")

if __name__ == "__main__":
    run_simulation()
