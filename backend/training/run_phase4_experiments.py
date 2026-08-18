"""
Phase 4 Experiment Runner — SatyaDrishti Video Model Architecture Correction

Runs controlled experiments comparing:
  A. Existing baseline (standard pilot, reference only — NOT retrained)
  B. Pretrained ResNet-18 + LSTM + 10 frames
  C. Pretrained ResNet-18 + MEAN POOLING + 10 frames

All experiments:
- Use the frozen 300-train / 100-val pilot subsets
- Lock the test set (100 videos) — never loaded or evaluated
- Do NOT overwrite production model, registry, or baseline metrics
- Save outputs to backend/training/results/video/phase4/
"""

import os
import sys
import json
import time
from datetime import datetime

# Ensure the project root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.training.train_video_model import train

logger = setup_logger("Phase4Runner")

PHASE4_DIR = os.path.join(PipelineConfig.RESULTS_DIR, "video", "phase4")
os.makedirs(PHASE4_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────
EXPERIMENTS = {
    "B_lstm_10frames": {
        "label": "Experiment B: Pretrained ResNet-18 + LSTM + 10 Frames",
        "temporal_pooling": "lstm",
        "max_frames": 10,
        "pretrained": True,
        "strategy": "standard",
        "epochs": 5,
    },
    "C_mean_10frames": {
        "label": "Experiment C: Pretrained ResNet-18 + Mean Pooling + 10 Frames",
        "temporal_pooling": "mean",
        "max_frames": 10,
        "pretrained": True,
        "strategy": "standard",
        "epochs": 5,
    },
}


def load_baseline_metrics():
    """Load the existing baseline pilot metrics (Experiment A) from disk."""
    baseline_path = os.path.join(PipelineConfig.RESULTS_DIR, "video", "baseline_metrics.json")
    if not os.path.exists(baseline_path):
        logger.error(f"Baseline metrics file not found at: {baseline_path}")
        return None
    with open(baseline_path, "r") as f:
        return json.load(f)


def run_experiment(name: str, config: dict) -> dict:
    """Execute a single Phase 4 experiment and return the saved metrics path."""
    exp_dir = os.path.join(PHASE4_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"  STARTING: {config['label']}")
    logger.info(f"  Output:   {exp_dir}")
    logger.info("=" * 60)

    start_time = time.time()

    train(
        epochs=config["epochs"],
        batch_size=PipelineConfig.BATCH_SIZE,
        lr=PipelineConfig.LEARNING_RATE,
        generate_dummy=False,
        is_pilot=True,
        strategy=config["strategy"],
        pretrained=config["pretrained"],
        temporal_pooling=config["temporal_pooling"],
        max_frames=config["max_frames"],
        output_dir=exp_dir,
        skip_promotion_overwrite=True,
        lock_test_set=True,
    )

    elapsed = time.time() - start_time
    logger.info(f"  COMPLETED: {config['label']} in {elapsed:.1f}s")

    # Find the generated metrics JSON
    metrics_files = [f for f in os.listdir(exp_dir) if f.endswith("_metrics.json")]
    if not metrics_files:
        logger.error(f"No metrics file found in {exp_dir}")
        return None

    metrics_path = os.path.join(exp_dir, sorted(metrics_files)[-1])
    with open(metrics_path, "r") as f:
        return json.load(f)


def generate_comparison_table(baseline: dict, results: dict) -> str:
    """Generate a markdown table comparing all experiments."""

    def fmt(v, pct=False):
        if v is None:
            return "N/A"
        if isinstance(v, bool):
            return "YES" if v else "NO"
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:.4f}"

    # Extract metrics from each experiment
    def extract(metrics_dict, source="final_metrics"):
        m = metrics_dict.get(source, {})
        return {
            "macro_f1": m.get("macro_f1"),
            "roc_auc": m.get("roc_auc"),
            "real_recall": m.get("real_recall"),
            "fake_recall": m.get("fake_recall"),
            "real_precision": m.get("real_precision"),
            "fake_precision": m.get("fake_precision"),
            "accuracy": m.get("accuracy"),
            "ece": m.get("ece"),
            "class_collapse": m.get("class_collapse", False),
            "real_pred_pct": m.get("real_pred_pct"),
            "fake_pred_pct": m.get("fake_pred_pct"),
        }

    all_experiments = {"A_baseline": extract(baseline)}
    for name, metrics in results.items():
        if metrics:
            all_experiments[name] = extract(metrics)

    # Build table
    metrics_labels = [
        ("macro_f1", "Macro-F1", False),
        ("roc_auc", "ROC-AUC", False),
        ("real_recall", "REAL Recall", False),
        ("fake_recall", "FAKE Recall", False),
        ("real_precision", "REAL Precision", False),
        ("fake_precision", "FAKE Precision", False),
        ("accuracy", "Accuracy", False),
        ("ece", "ECE", False),
        ("class_collapse", "Class Collapse", False),
        ("real_pred_pct", "REAL Pred %", True),
        ("fake_pred_pct", "FAKE Pred %", True),
    ]

    exp_names = list(all_experiments.keys())
    header = "| Metric | " + " | ".join(exp_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(exp_names)) + "|"

    rows = [header, sep]
    for key, label, pct in metrics_labels:
        row = f"| {label} | "
        for exp_name in exp_names:
            val = all_experiments[exp_name].get(key)
            row += fmt(val, pct=pct) + " | "
        rows.append(row)

    return "\n".join(rows)


def determine_verdict(baseline: dict, results: dict) -> tuple:
    """Determine which experiment (if any) passes the promotion gate."""
    base_macro_f1 = baseline["final_metrics"]["macro_f1"]
    base_real_recall = baseline["final_metrics"]["real_recall"]

    best_candidate = None
    best_macro_f1 = base_macro_f1

    for name, metrics in results.items():
        if metrics is None:
            continue
        m = metrics["final_metrics"]
        collapse = m.get("class_collapse", False)
        macro_f1 = m.get("macro_f1", 0.0)
        real_recall = m.get("real_recall", 0.0)

        if collapse:
            logger.warning(f"  {name}: REJECTED (class collapse)")
            continue
        if macro_f1 <= base_macro_f1:
            logger.info(f"  {name}: REJECTED (Macro-F1 {macro_f1:.4f} <= baseline {base_macro_f1:.4f})")
            continue
        if real_recall < base_real_recall:
            logger.info(f"  {name}: REJECTED (REAL recall {real_recall:.4f} < baseline {base_real_recall:.4f})")
            continue

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_candidate = name

    if best_candidate:
        return best_candidate, True
    return None, False


def generate_report(baseline: dict, results: dict, candidate: str, passed: bool):
    """Generate the Phase 4 report markdown file."""
    table = generate_comparison_table(baseline, results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hyp_sections = []
    for name, metrics in results.items():
        if metrics is None:
            continue
        hyp = metrics.get("hyperparameters", {})
        hyp_sections.append(
            f"### {name}\n"
            f"- Temporal pooling: `{hyp.get('temporal_pooling', 'N/A')}`\n"
            f"- Max frames: `{hyp.get('max_frames_per_video', 'N/A')}`\n"
            f"- Strategy: `{hyp.get('strategy', 'N/A')}`\n"
            f"- Pretrained: `{hyp.get('pretrained', 'N/A')}`\n"
            f"- Epochs: `{hyp.get('epochs', 'N/A')}`\n"
            f"- Optimized Threshold: `{hyp.get('optimized_threshold', 'N/A')}`\n"
        )

    verdict_block = ""
    if passed:
        verdict_block = (
            f"> [!IMPORTANT]\n"
            f"> **PROMOTION GATE PASSED**: `{candidate}` outperformed the baseline.\n"
            f"> This candidate is eligible for one-time test-set evaluation.\n"
            f"> **Flag: `PHASE4_READY_FOR_FULL_TRAINING = True`**\n"
        )
    else:
        verdict_block = (
            f"> [!WARNING]\n"
            f"> **PROMOTION GATE FAILED**: No candidate outperformed the baseline.\n"
            f"> Further architectural investigation is required before full-scale training.\n"
            f"> **Flag: `PHASE4_READY_FOR_FULL_TRAINING = False`**\n"
        )

    confusion_sections = []
    for name, metrics in [("A_baseline", baseline)] + list(results.items()):
        if metrics is None:
            continue
        cm = metrics.get("final_metrics", {}).get("confusion_matrix", [[0, 0], [0, 0]])
        confusion_sections.append(
            f"### {name}\n"
            f"```\n"
            f"              Pred REAL   Pred FAKE\n"
            f"True REAL     {cm[0][0]:>8}   {cm[0][1]:>8}\n"
            f"True FAKE     {cm[1][0]:>8}   {cm[1][1]:>8}\n"
            f"```\n"
        )

    report = f"""# Phase 4 Report — Video Model Architecture Correction

**Generated**: {timestamp}

## Summary

Phase 4 evaluated whether replacing the unidirectional LSTM temporal aggregation with Mean Temporal Pooling and increasing frame density from 5 to 10 frames resolves the representation collapse identified in Phase 3 diagnostics.

**Class Mapping**: Class 0 = REAL, Class 1 = FAKE

---

## Experiment Configurations

{chr(10).join(hyp_sections)}

---

## Validation Metrics Comparison

{table}

---

## Confusion Matrices (Validation)

{chr(10).join(confusion_sections)}

---

## Verdict

{verdict_block}

---

## Safety Checks

- Production model (`video_model.pth`): **NOT modified**
- Model registry (`model_registry.json`): **NOT modified**
- Existing baseline metrics: **NOT modified**
- Test set (100 videos): **NOT loaded or evaluated**
- All Phase 4 outputs saved to: `backend/training/results/video/phase4/`
"""

    report_path = os.path.join(PHASE4_DIR, "phase4_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Phase 4 report saved to: {report_path}")
    return report_path


def main():
    logger.info("=" * 60)
    logger.info("  PHASE 4: VIDEO MODEL ARCHITECTURE CORRECTION")
    logger.info("  Controlled Validation Experiment Runner")
    logger.info("=" * 60)

    # Safety: Verify production files exist and record their timestamps
    prod_model_path = PipelineConfig.VIDEO_MODEL_PATH
    registry_path = PipelineConfig.REGISTRY_PATH

    prod_mtime = os.path.getmtime(prod_model_path) if os.path.exists(prod_model_path) else None
    reg_mtime = os.path.getmtime(registry_path) if os.path.exists(registry_path) else None

    logger.info(f"Production model mtime: {prod_mtime}")
    logger.info(f"Registry mtime: {reg_mtime}")

    # 1. Load baseline
    baseline = load_baseline_metrics()
    if baseline is None:
        logger.error("Cannot proceed without baseline metrics. Aborting.")
        sys.exit(1)

    logger.info(f"Baseline Macro-F1: {baseline['final_metrics']['macro_f1']:.4f}")
    logger.info(f"Baseline REAL Recall: {baseline['final_metrics']['real_recall']:.4f}")

    # 2. Run experiments
    results = {}
    for name, config in EXPERIMENTS.items():
        metrics = run_experiment(name, config)
        results[name] = metrics

    # 3. Determine verdict
    candidate, passed = determine_verdict(baseline, results)

    # 4. Generate report
    report_path = generate_report(baseline, results, candidate, passed)

    # 5. Safety: Verify production files are untouched
    if prod_mtime is not None:
        current_mtime = os.path.getmtime(prod_model_path) if os.path.exists(prod_model_path) else None
        if current_mtime != prod_mtime:
            logger.error("SAFETY VIOLATION: Production model file was modified during Phase 4!")
        else:
            logger.info("SAFETY CHECK PASSED: Production model file unchanged.")

    if reg_mtime is not None:
        current_reg_mtime = os.path.getmtime(registry_path) if os.path.exists(registry_path) else None
        if current_reg_mtime != reg_mtime:
            logger.error("SAFETY VIOLATION: Model registry file was modified during Phase 4!")
        else:
            logger.info("SAFETY CHECK PASSED: Model registry file unchanged.")

    # 6. Summary
    logger.info("=" * 60)
    if passed:
        logger.info(f"  RESULT: {candidate} PASSED the promotion gate.")
        logger.info(f"  Next step: Run one-time test-set evaluation on {candidate}.")
    else:
        logger.info("  RESULT: No candidate passed the promotion gate.")
        logger.info("  Next step: Further architecture investigation required.")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
