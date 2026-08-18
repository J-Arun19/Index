import os
import sys
import torch
import json
import csv
import random
import time
from datetime import datetime

# Add workspace root to pythonpath
workspace_dir = r"d:\Download\satya-vision-guard-main\satya-vision-guard-main"
sys.path.insert(0, workspace_dir)

from backend.utils.config import PipelineConfig
from backend.training.dataset_loader import DeepfakeDataset
from backend.training.train_image_model import train as train_image, subset_dataset_deterministically as subset_image, check_data_leakage as leak_image
from backend.training.train_video_model import train as train_video, subset_dataset_deterministically as subset_video, check_data_leakage as leak_video

# Optimize PyTorch CPU execution threads (physical cores)
torch.set_num_threads(6)
print(f"[RUNNER] Optimized PyTorch CPU threads: {torch.get_num_threads()}")

def get_comparison_table(media_type):
    results_dir = os.path.join(workspace_dir, "backend", "training", "results", media_type)
    if not os.path.exists(results_dir):
        return "No results found."
    
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "baseline_metrics.json"]
    if not json_files:
        return "No experiment result files found."
        
    runs = []
    for file in json_files:
        path = os.path.join(results_dir, file)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            runs.append({
                "filename": file,
                "strategy": data["hyperparameters"]["strategy"],
                "pretrained": data["hyperparameters"]["pretrained"],
                "epochs": data["hyperparameters"]["epochs"],
                "opt_threshold": data["hyperparameters"]["optimized_threshold"],
                "val_macro_f1": data["final_metrics"]["macro_f1"],
                "val_real_recall": data["final_metrics"]["real_recall"],
                "test_acc": data["test_metrics"]["accuracy"],
                "test_precision": data["test_metrics"]["precision"],
                "test_recall": data["test_metrics"]["recall"],
                "test_f1": data["test_metrics"]["f1_score"],
                "test_macro_f1": data["test_metrics"]["macro_f1"],
                "test_auc": data["test_metrics"]["roc_auc"],
                "test_ece": data["test_metrics"]["ece"],
                "test_real_recall": data["test_metrics"]["real_recall"],
                "test_fake_recall": data["test_metrics"]["fake_recall"]
            })
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    # Sort runs by test macro f1
    runs.sort(key=lambda x: x["test_macro_f1"], reverse=True)
    
    table = []
    table.append(f"{'Strategy':<15} | {'Pretrained':<10} | {'Thresh':<6} | {'Val F1':<7} | {'Test Acc':<8} | {'Test F1':<7} | {'Macro F1':<8} | {'ROC-AUC':<7} | {'ECE':<6} | {'Real Rec':<8}")
    table.append("-" * 115)
    for r in runs:
        strat = r["strategy"]
        pre = "Yes" if r["pretrained"] else "No"
        table.append(
            f"{strat:<15} | {pre:<10} | {r['opt_threshold']:<6.2f} | {r['val_macro_f1']:<7.4f} | {r['test_acc']:<8.2f} | "
            f"{r['test_f1']:<7.4f} | {r['test_macro_f1']:<8.4f} | {r['test_auc']:<7.4f} | {r['test_ece']:<6.4f} | {r['test_real_recall']:<8.4f}"
        )
    return "\n".join(table)

def run_image_matrix():
    print("\n==================================================")
    print("      IMAGE MODEL CONTROLLED PILOT MATRIX")
    print("==================================================")
    
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    
    # 1. Load and subset datasets
    print("[RUNNER] Initializing image datasets...")
    train_ds = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    val_ds = DeepfakeDataset(split="val", media_type="image", manifest_path=manifest_path)
    test_ds = DeepfakeDataset(split="test", media_type="image", manifest_path=manifest_path)
    
    # Subsetting
    pilot_manifest_path = os.path.join(PipelineConfig.RESULTS_DIR, "image", "pilot_subset_manifest.csv")
    if os.path.exists(pilot_manifest_path):
        print(f"[RUNNER] Loading pre-saved pilot subset manifest from: {pilot_manifest_path}")
        loaded_paths = {"train": set(), "val": set(), "test": set()}
        with open(pilot_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded_paths[row["split"]].add(row["file_path"])
        
        train_ds.samples = [s for s in train_ds.samples if s["file_path"] in loaded_paths["train"]]
        val_ds.samples = [s for s in val_ds.samples if s["file_path"] in loaded_paths["val"]]
        test_ds.samples = [s for s in test_ds.samples if s["file_path"] in loaded_paths["test"]]
        print(f"[RUNNER] Loaded exact pilot manifest. Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    else:
        print("[RUNNER] Subsetting deterministically...")
        subset_image(train_ds, 3000)
        subset_image(val_ds, 1000)
        subset_image(test_ds, 1000)
        # Save manifest
        os.makedirs(os.path.dirname(pilot_manifest_path), exist_ok=True)
        all_samples = []
        for s in train_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "train"
            all_samples.append(s_copy)
        for s in val_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "val"
            all_samples.append(s_copy)
        for s in test_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "test"
            all_samples.append(s_copy)
        
        if all_samples:
            with open(pilot_manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_samples[0].keys())
                writer.writeheader()
                writer.writerows(all_samples)
            print(f"[RUNNER] Saved pilot subset manifest to {pilot_manifest_path}")

    # Leakage check
    leak_image(train_ds, val_ds, test_ds)
    
    # Pre-cache
    print("[RUNNER] Pre-caching image datasets to RAM...")
    train_ds.pre_cache_samples()
    val_ds.pre_cache_samples()
    test_ds.pre_cache_samples()
    print("[RUNNER] Pre-caching finished successfully.")
    
    datasets = (train_ds, val_ds, test_ds)
    
    # 2. Sequential matrix runs
    configs = [
        # Baseline (standard, no-aug, pretrained=False)
        {"name": "Baseline", "strategy": "standard", "pretrained": False, "use_aug": False},
        # Augmented (standard, with aug, pretrained=False)
        {"name": "Augmented", "strategy": "standard", "pretrained": False, "use_aug": True},
        # Pretrained (standard, with aug, pretrained=True)
        {"name": "Pretrained", "strategy": "standard", "pretrained": True, "use_aug": True},
        # Weighted Loss (weighted_loss, with aug, pretrained=True)
        {"name": "Weighted Loss", "strategy": "weighted_loss", "pretrained": True, "use_aug": True},
        # Sampler (sampler, with aug, pretrained=True)
        {"name": "Sampler", "strategy": "sampler", "pretrained": True, "use_aug": True},
        # Both (both, with aug, pretrained=True)
        {"name": "Both", "strategy": "both", "pretrained": True, "use_aug": True},
    ]
    
    for config in configs:
        print(f"\n[RUNNER] >>> Starting Experiment: {config['name']} <<<")
        PipelineConfig.USE_AUGMENTATION = config["use_aug"]
        print(f"[RUNNER] Set USE_AUGMENTATION to {config['use_aug']}")
        
        # Check if baseline run already exists in results folder to avoid rerunning
        # but wait, let's rerun all to ensure completely consistent output if needed, 
        # since we already ran Baseline let's skip Baseline if its json exists.
        results_dir = os.path.join(workspace_dir, "backend", "training", "results", "image")
        json_pattern = os.path.join(results_dir, "image_standard_*_metrics.json")
        import glob
        matching_files = [f for f in glob.glob(json_pattern) if "standard" in f and not "pretrained" in f]
        
        # If we have baseline results, skip baseline run
        if config["name"] == "Baseline" and len(matching_files) > 0:
            print("[RUNNER] Baseline run metrics already exist. Skipping rerun.")
            continue
            
        train_image(
            epochs=2,
            batch_size=128,
            lr=PipelineConfig.LEARNING_RATE,
            generate_dummy=False,
            is_pilot=True,
            strategy=config["strategy"],
            pretrained=config["pretrained"],
            preloaded_datasets=datasets
        )
        
    print("\n[RUNNER] Image Model matrix run finished.")
    print(get_comparison_table("image"))

def run_video_matrix():
    print("\n==================================================")
    print("      VIDEO MODEL CONTROLLED PILOT MATRIX")
    print("==================================================")
    
    # PILOT RUN optimization: Override frame sampling and length configuration to speed up runs
    PipelineConfig.MAX_FRAMES_PER_VIDEO = 5
    PipelineConfig.FRAME_INTERVAL = 15
    print(f"[RUNNER] PILOT RUN optimization: Overriding MAX_FRAMES_PER_VIDEO to {PipelineConfig.MAX_FRAMES_PER_VIDEO} and FRAME_INTERVAL to {PipelineConfig.FRAME_INTERVAL}")
    
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    
    # 1. Load and subset datasets
    print("[RUNNER] Initializing video datasets...")
    train_ds = DeepfakeDataset(split="train", media_type="video", manifest_path=manifest_path, use_cache=False)
    val_ds = DeepfakeDataset(split="val", media_type="video", manifest_path=manifest_path, use_cache=False)
    test_ds = DeepfakeDataset(split="test", media_type="video", manifest_path=manifest_path, use_cache=False)
    
    # Subsetting
    pilot_manifest_path = os.path.join(PipelineConfig.RESULTS_DIR, "video", "pilot_subset_manifest.csv")
    if os.path.exists(pilot_manifest_path):
        print(f"[RUNNER] Loading pre-saved pilot subset manifest from: {pilot_manifest_path}")
        loaded_paths = {"train": set(), "val": set(), "test": set()}
        with open(pilot_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loaded_paths[row["split"]].add(row["file_path"])
        
        train_ds.samples = [s for s in train_ds.samples if s["file_path"] in loaded_paths["train"]]
        val_ds.samples = [s for s in val_ds.samples if s["file_path"] in loaded_paths["val"]]
        test_ds.samples = [s for s in test_ds.samples if s["file_path"] in loaded_paths["test"]]
        print(f"[RUNNER] Loaded exact pilot manifest. Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    else:
        print("[RUNNER] Subsetting deterministically...")
        subset_video(train_ds, 300)
        subset_video(val_ds, 100)
        subset_video(test_ds, 100)
        # Save manifest
        os.makedirs(os.path.dirname(pilot_manifest_path), exist_ok=True)
        all_samples = []
        for s in train_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "train"
            all_samples.append(s_copy)
        for s in val_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "val"
            all_samples.append(s_copy)
        for s in test_ds.samples:
            s_copy = s.copy()
            s_copy["split"] = "test"
            all_samples.append(s_copy)
        
        if all_samples:
            with open(pilot_manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_samples[0].keys())
                writer.writeheader()
                writer.writerows(all_samples)
            print(f"[RUNNER] Saved pilot subset manifest to {pilot_manifest_path}")

    # Leakage check
    leak_video(train_ds, val_ds, test_ds)
    
    # Pre-cache skipped for video due to high RAM overhead on CPU environment
    print("[RUNNER] Skipping video dataset pre-caching to RAM to avoid OOM.")
    
    datasets = (train_ds, val_ds, test_ds)
    
    # 2. Sequential matrix runs
    configs = [
        # Baseline (standard, no-aug, pretrained=False)
        {"name": "Baseline", "strategy": "standard", "pretrained": False, "use_aug": False},
        # Augmented (standard, with aug, pretrained=False)
        {"name": "Augmented", "strategy": "standard", "pretrained": False, "use_aug": True},
        # Pretrained (standard, with aug, pretrained=True)
        {"name": "Pretrained", "strategy": "standard", "pretrained": True, "use_aug": True},
        # Weighted Loss (weighted_loss, with aug, pretrained=True)
        {"name": "Weighted Loss", "strategy": "weighted_loss", "pretrained": True, "use_aug": True},
        # Sampler (sampler, with aug, pretrained=True)
        {"name": "Sampler", "strategy": "sampler", "pretrained": True, "use_aug": True},
        # Both (both, with aug, pretrained=True)
        {"name": "Both", "strategy": "both", "pretrained": True, "use_aug": True},
    ]
    
    for config in configs:
        print(f"\n[RUNNER] >>> Starting Experiment: {config['name']} <<<")
        PipelineConfig.USE_AUGMENTATION = config["use_aug"]
        print(f"[RUNNER] Set USE_AUGMENTATION to {config['use_aug']}")
        
        # Check if baseline run already exists
        results_dir = os.path.join(workspace_dir, "backend", "training", "results", "video")
        json_pattern = os.path.join(results_dir, "video_standard_*_metrics.json")
        import glob
        matching_files = [f for f in glob.glob(json_pattern) if "standard" in f and not "pretrained" in f]
        
        if config["name"] == "Baseline" and len(matching_files) > 0:
            print("[RUNNER] Baseline run metrics already exist. Skipping rerun.")
            continue
            
        train_video(
            epochs=2,
            batch_size=16,
            lr=PipelineConfig.LEARNING_RATE,
            generate_dummy=False,
            is_pilot=True,
            strategy=config["strategy"],
            pretrained=config["pretrained"],
            preloaded_datasets=datasets
        )
        
    print("\n[RUNNER] Video Model matrix run finished.")
    print(get_comparison_table("video"))

if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if mode in ("image", "all"):
        run_image_matrix()
    if mode in ("video", "all"):
        run_video_matrix()
