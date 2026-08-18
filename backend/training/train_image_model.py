import os
import argparse
import random
import json
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np

from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.dataset_loader import DatasetLoader
from backend.training.image_dataset import ImageDataset
from backend.models.image_model import ImageModel
from backend.utils.metrics_helper import compute_ece, optimize_threshold

logger = setup_logger("TrainImageModel")

def check_data_leakage(train_ds, val_ds, test_ds):
    """
    Checks for leakage between train, validation, and test datasets at the identity/source-video level.
    Halts training immediately if leakage is detected.
    """
    logger.info("Running automated data leakage checks...")
    
    def get_source_id(sample):
        return os.path.basename(sample["file_path"])
        
    train_sources = set(get_source_id(s) for s in train_ds.samples)
    val_sources = set(get_source_id(s) for s in val_ds.samples)
    test_sources = set(get_source_id(s) for s in test_ds.samples)
    
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
    test_ids = set(extract_identity(s["file_path"]) for s in test_ds.samples if extract_identity(s["file_path"]) is not None)
    
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

def generate_dummy_data(num_samples: int = 10):
    """Generates dummy real and fake images to test the training pipeline."""
    import cv2
    logger.warning("[DUMMY DATA — TEST ONLY] Generating dummy training samples...")
    os.makedirs(PipelineConfig.IMAGE_REAL_DIR, exist_ok=True)
    os.makedirs(PipelineConfig.IMAGE_FAKE_DIR, exist_ok=True)
    
    for i in range(num_samples):
        # Create random noise image (simulating a face frame)
        img_real = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.circle(img_real, (150, 150), 80, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(PipelineConfig.IMAGE_REAL_DIR, f"dummy_real_{i}.jpg"), img_real)
        
        img_fake = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.circle(img_fake, (150, 150), 80, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(PipelineConfig.IMAGE_FAKE_DIR, f"dummy_fake_{i}.jpg"), img_fake)
        
    logger.warning("[DUMMY DATA — TEST ONLY] Dummy image dataset generation complete.")

def subset_dataset_deterministically(dataset, target_size: int):
    """Subsets the dataset deterministically to exactly target_size, grouping by source file (e.g. video)
    to minimize the number of unique video files that need to be opened and decoded,
    while preserving class ratio and sequence identity splits."""
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
    samples_1 = sum(len(s) for _, s in files_by_label[1])
    
    ratio_0 = samples_0 / total_samples
    target_samples_0 = max(1, int(target_size * ratio_0))
    target_samples_1 = max(1, target_size - target_samples_0)
    
    selected_samples = []
    
    # Class 0 selection
    accumulated_0 = 0
    files_by_label[0].sort(key=lambda x: x[0])
    for file_path, samples in files_by_label[0]:
        if accumulated_0 >= target_samples_0:
            break
        selected_samples.extend(samples)
        accumulated_0 += len(samples)
        
    # Class 1 selection
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
    
    logger.info(f"Subset dataset deterministically to {len(dataset.samples)} samples (target: {target_size}) using grouped files to optimize video I/O.")

def run_promotion_gate(media_type: str, run_metrics: dict, run_weights_path: str, is_baseline_run: bool):
    """
    Applies the production promotion gate by comparing validation metrics against baseline.
    
    If the run is marked as baseline, saves it directly as the reference baseline.
    If it is an improved run, compares validation Macro-F1 and REAL Recall.
    If successful, promotes the model to the primary production path.
    """
    results_dir = os.path.join(PipelineConfig.RESULTS_DIR, media_type)
    os.makedirs(results_dir, exist_ok=True)
    baseline_path = os.path.join(results_dir, "baseline_metrics.json")
    
    if is_baseline_run:
        logger.info("[PROMOTION GATE] Saving metrics as the baseline reference.")
        with open(baseline_path, "w") as f:
            json.dump(run_metrics, f, indent=4)
        
        # Copy to primary production path as base
        prod_path = PipelineConfig.IMAGE_MODEL_PATH if media_type == "image" else PipelineConfig.VIDEO_MODEL_PATH
        os.makedirs(os.path.dirname(prod_path), exist_ok=True)
        shutil.copy2(run_weights_path, prod_path)
        logger.info(f"[PROMOTION GATE] Copied baseline weights to production path: {prod_path}")
        update_model_registry(media_type, run_metrics, prod_path, is_baseline=True)
        return True
        
    if not os.path.exists(baseline_path):
        logger.warning(f"[PROMOTION GATE] Baseline metrics file not found at {baseline_path}. Promoting current model as initial reference.")
        prod_path = PipelineConfig.IMAGE_MODEL_PATH if media_type == "image" else PipelineConfig.VIDEO_MODEL_PATH
        os.makedirs(os.path.dirname(prod_path), exist_ok=True)
        shutil.copy2(run_weights_path, prod_path)
        update_model_registry(media_type, run_metrics, prod_path, is_baseline=False)
        return True
        
    with open(baseline_path, "r") as f:
        baseline = json.load(f)
        
    base_macro_f1 = baseline["final_metrics"]["macro_f1"]
    base_real_recall = baseline["final_metrics"]["real_recall"]
    
    new_macro_f1 = run_metrics["final_metrics"]["macro_f1"]
    new_real_recall = run_metrics["final_metrics"]["real_recall"]
    
    logger.info(f"[PROMOTION GATE] Comparing against baseline:")
    logger.info(f"- Macro-F1: Baseline={base_macro_f1:.4f}, New={new_macro_f1:.4f}")
    logger.info(f"- REAL Recall: Baseline={base_real_recall:.4f}, New={new_real_recall:.4f}")
    
    # Class collapse safeguard: check if both classes are predicted in validation predictions
    cm = run_metrics["final_metrics"]["confusion_matrix"]
    pred_0_total = cm[0][0] + cm[1][0]
    pred_1_total = cm[0][1] + cm[1][1]
    
    if pred_0_total == 0 or pred_1_total == 0:
        logger.error("[PROMOTION GATE] Promotion REJECTED: Model suffered from complete class collapse.")
        return False
        
    # Promotion conditions: New Macro-F1 > baseline, and New REAL Recall >= baseline
    if new_macro_f1 > base_macro_f1 and new_real_recall >= base_real_recall:
        logger.info("[PROMOTION GATE] Promotion SUCCESSFUL! Model outperformed the baseline.")
        prod_path = PipelineConfig.IMAGE_MODEL_PATH if media_type == "image" else PipelineConfig.VIDEO_MODEL_PATH
        os.makedirs(os.path.dirname(prod_path), exist_ok=True)
        shutil.copy2(run_weights_path, prod_path)
        logger.info(f"[PROMOTION GATE] Copied new weights to production path: {prod_path}")
        update_model_registry(media_type, run_metrics, prod_path, is_baseline=False)
        return True
    else:
        logger.warning("[PROMOTION GATE] Promotion REJECTED: New model did not outperform baseline in Macro-F1 and REAL Recall.")
        return False

def update_model_registry(media_type: str, run_metrics: dict, prod_path: str, is_baseline: bool):
    """Updates model_registry.json with the details of the promoted active model."""
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

def train(epochs: int, batch_size: int, lr: float, generate_dummy: bool, is_pilot: bool = False, strategy: str = None, pretrained: bool = False, preloaded_datasets: tuple = None):
    logger.info(PipelineConfig.get_summary())
    
    is_dummy_run = generate_dummy
    
    # 1. Dataset Initialization
    if preloaded_datasets is not None:
        train_dataset, val_dataset, test_dataset = preloaded_datasets
        is_dummy_run = False
        log_prefix = "[PILOT TRAINING] " if is_pilot else "[REAL TRAINING] "
    elif is_dummy_run:
        logger.warning("[DUMMY DATA — TEST ONLY] --generate-dummy flag was provided. Setting up dummy dataset...")
        paths, labels = DatasetLoader.scan_image_dataset()
        if len(paths) == 0:
            logger.warning("[DUMMY DATA — TEST ONLY] Generating dummy dataset...")
            generate_dummy_data(10)
            paths, labels = DatasetLoader.scan_image_dataset()
            
        dataset = ImageDataset(paths, labels)
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
        log_prefix = "[DUMMY DATA — TEST ONLY] "
        logger.info(f"{log_prefix}Split dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    else:
        logger.info("========================================")
        logger.info("  [REAL DATASET TRAINING - IMAGE MODEL]  ")
        logger.info("========================================")
        
        manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file '{manifest_path}' is missing. Please run generate_manifest first.")
            
        logger.info(f"Loading real datasets from manifest: {manifest_path}")
        from backend.training.dataset_loader import DeepfakeDataset
        
        train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
        val_dataset = DeepfakeDataset(split="val", media_type="image", manifest_path=manifest_path)
        test_dataset = DeepfakeDataset(split="test", media_type="image", manifest_path=manifest_path)
        
        if len(train_dataset) == 0:
            raise ValueError("No training samples found in the image dataset manifest.")
        if len(val_dataset) == 0:
            raise ValueError("No validation samples found in the image dataset manifest.")
        if len(test_dataset) == 0:
            raise ValueError("No test samples found in the image dataset manifest.")
            
        # Check if saved pilot manifest exists to guarantee exact same subset
        pilot_manifest_path = os.path.join(PipelineConfig.RESULTS_DIR, "image", "pilot_subset_manifest.csv")
        if is_pilot and os.path.exists(pilot_manifest_path):
            logger.info(f"Loading pre-saved pilot subset manifest from: {pilot_manifest_path}")
            import csv
            loaded_paths = {"train": set(), "val": set(), "test": set()}
            with open(pilot_manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loaded_paths[row["split"]].add(row["file_path"])
            
            def filter_by_subset(dataset, split_name):
                subset_paths = loaded_paths[split_name]
                dataset.samples = [s for s in dataset.samples if s["file_path"] in subset_paths]
                logger.info(f"Filtered {split_name} dataset to match pre-saved pilot manifest: {len(dataset.samples)} samples.")
                
            filter_by_subset(train_dataset, "train")
            filter_by_subset(val_dataset, "val")
            filter_by_subset(test_dataset, "test")
        else:
            if is_pilot:
                logger.info("Configuring PILOT RUN subset size (target: 3000 train, 1000 val, 1000 test)...")
                subset_dataset_deterministically(train_dataset, 3000)
                subset_dataset_deterministically(val_dataset, 1000)
                subset_dataset_deterministically(test_dataset, 1000)
                
                # Save the pilot manifest
                os.makedirs(os.path.dirname(pilot_manifest_path), exist_ok=True)
                import csv
                all_samples = []
                for s in train_dataset.samples:
                    s_copy = s.copy()
                    s_copy["split"] = "train"
                    all_samples.append(s_copy)
                for s in val_dataset.samples:
                    s_copy = s.copy()
                    s_copy["split"] = "val"
                    all_samples.append(s_copy)
                for s in test_dataset.samples:
                    s_copy = s.copy()
                    s_copy["split"] = "test"
                    all_samples.append(s_copy)
                
                if all_samples:
                    keys = all_samples[0].keys()
                    with open(pilot_manifest_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(all_samples)
                logger.info(f"Saved pilot subset manifest to: {pilot_manifest_path}")
                
        # Run automated data leakage checks
        check_data_leakage(train_dataset, val_dataset, test_dataset)
            
        if len(train_dataset) <= 5000:
            logger.info("Pre-caching train, validation and test datasets to RAM...")
            train_dataset.pre_cache_samples()
            val_dataset.pre_cache_samples()
            test_dataset.pre_cache_samples()
            
        # Two-class training safeguard
        train_labels = [s["label"] for s in train_dataset.samples]
        unique_labels = set(train_labels)
        if len(unique_labels) < 2:
            raise ValueError(f"ImageModel training stopped: Training dataset must contain both classes (0 and 1), but found: {list(unique_labels)}")
            
        log_prefix = "[PILOT TRAINING] " if is_pilot else "[REAL TRAINING] "
        logger.info(f"Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples, {len(test_dataset)} test samples.")

    # 2. Imbalance Handling Configurations
    active_strategy = strategy or PipelineConfig.IMBALANCE_STRATEGY
    logger.info(f"Selected Imbalance Strategy: '{active_strategy}'")
    
    device = torch.device(PipelineConfig.DEVICE)
    model = ImageModel(num_classes=2, pretrained=pretrained).to(device)
    
    # Calculate training frequencies
    if not is_dummy_run:
        train_lbls = [s["label"] for s in train_dataset.samples]
        class_counts = [train_lbls.count(0), train_lbls.count(1)]
        total = len(train_lbls)
        # Class weights: inverse frequency
        w_real = total / (2.0 * max(1, class_counts[0]))
        w_fake = total / (2.0 * max(1, class_counts[1]))
        class_weights = [w_real, w_fake]
    else:
        class_weights = [1.0, 1.0]
        
    sampler = None
    criterion_weight = None
    
    if active_strategy == "weighted_loss":
        criterion_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.info(f"Strategy: weighted_loss. Class weights: REAL={w_real:.4f}, FAKE={w_fake:.4f}")
    elif active_strategy == "sampler":
        if not is_dummy_run:
            sample_weights = [class_weights[s["label"]] for s in train_dataset.samples]
        else:
            sample_weights = [1.0] * len(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Strategy: sampler. WeightedRandomSampler configured.")
    elif active_strategy == "both":
        criterion_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
        if not is_dummy_run:
            sample_weights = [class_weights[s["label"]] for s in train_dataset.samples]
        else:
            sample_weights = [1.0] * len(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Strategy: both. Both WeightedRandomSampler and class-weighted loss configured.")
    else:
        logger.info("Strategy: standard baseline. No weighted sampler/loss.")
        
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    
    # 3. Optimizer Setup (Backbone vs Head Learning Rates)
    if pretrained:
        # Freeze ResNet layers 1-3. Keep layer 4 and fc classification head trainable.
        logger.info("Pretrained model selected. Freezing backbone layers 1-3 (keeping layer 4 and fc trainable).")
        for name, param in model.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
                
        backbone_params = [p for n, p in model.named_parameters() if "backbone.fc" not in n and p.requires_grad]
        head_params = [p for p in model.backbone.fc.parameters() if p.requires_grad]
        logger.info(f"Setting split learning rates. Backbone LR: {PipelineConfig.BACKBONE_LR}, Head LR: {PipelineConfig.HEAD_LR}")
        optimizer = optim.Adam([
            {"params": backbone_params, "lr": PipelineConfig.BACKBONE_LR},
            {"params": head_params, "lr": PipelineConfig.HEAD_LR}
        ], weight_decay=PipelineConfig.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=PipelineConfig.WEIGHT_DECAY)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 4. DataLoaders Setup
    # shuffle MUST be False when using sampler
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(batch_size, len(train_dataset)), 
        sampler=sampler, 
        shuffle=(sampler is None)
    )
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    
    best_val_loss = float('inf')
    best_val_metrics = {}
    best_opt_thresh = 0.5
    patience = PipelineConfig.PATIENCE
    epochs_no_improve = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_weights_path = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "runs", f"image_{active_strategy}_{timestamp}.pth")
    os.makedirs(os.path.dirname(run_weights_path), exist_ok=True)
    
    logger.info(f"{log_prefix}Starting training on device: {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, batch_labels in train_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100 if total_train > 0 else 0.0
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        total_val = 0
        
        val_labels_all = []
        val_probs_all = []
        
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * images.size(0)
                total_val += batch_labels.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                fake_probs = probs[:, 1]
                
                val_labels_all.extend(batch_labels.cpu().numpy())
                val_probs_all.extend(fake_probs.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        
        # 5. Threshold Optimization & Validation Metrics Evaluation
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
        
        val_labels_arr = np.array(val_labels_all)
        val_probs_arr = np.array(val_probs_all)
        
        # Find threshold that maximizes validation Macro-F1 only
        opt_thresh, best_val_macro_f1 = optimize_threshold(val_labels_arr, val_probs_arr)
        
        # Compute predicted classes based on optimized threshold
        val_preds_arr = (val_probs_arr >= opt_thresh).astype(int)
        
        epoch_val_acc = float(np.mean(val_preds_arr == val_labels_arr)) * 100
        val_precision = float(precision_score(val_labels_arr, val_preds_arr, zero_division=0))
        val_recall = float(recall_score(val_labels_arr, val_preds_arr, zero_division=0))
        val_f1 = float(f1_score(val_labels_arr, val_preds_arr, zero_division=0))
        
        try:
            val_auc = float(roc_auc_score(val_labels_arr, val_probs_arr))
        except ValueError:
            val_auc = 0.0
            
        val_cm = confusion_matrix(val_labels_arr, val_preds_arr).tolist()
        class_prec, class_rec, _, _ = precision_recall_fscore_support(val_labels_arr, val_preds_arr, labels=[0, 1], zero_division=0)
        real_precision = float(class_prec[0])
        real_recall = float(class_rec[0])
        fake_precision = float(class_prec[1])
        fake_recall = float(class_rec[1])
        
        val_ece = compute_ece(val_labels_arr, val_probs_arr)
        
        logger.info(
            f"{log_prefix}Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )
        logger.info(
            f"{log_prefix}Metrics (Threshold={opt_thresh:.2f}) - "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
            f"F1-Score: {val_f1:.4f}, Macro-F1: {best_val_macro_f1:.4f}, ROC-AUC: {val_auc:.4f}, ECE: {val_ece:.4f} | "
            f"REAL P/R: {real_precision:.4f}/{real_recall:.4f}, FAKE P/R: {fake_precision:.4f}/{fake_recall:.4f}"
        )
        logger.info(f"{log_prefix}Confusion Matrix: {val_cm}")
        
        # Save best weights for this run
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_opt_thresh = opt_thresh
            best_val_metrics = {
                "train_loss": epoch_train_loss,
                "train_accuracy": epoch_train_acc,
                "val_loss": epoch_val_loss,
                "accuracy": epoch_val_acc,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1,
                "macro_f1": best_val_macro_f1,
                "roc_auc": val_auc,
                "ece": val_ece,
                "confusion_matrix": val_cm,
                "real_precision": real_precision,
                "real_recall": real_recall,
                "fake_precision": fake_precision,
                "fake_recall": fake_recall
            }
            model.save_weights(run_weights_path)
            logger.info(f"{log_prefix}New best run model saved to {run_weights_path} with Val Loss: {best_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        scheduler.step(epoch_val_loss)
        
        # Early Stopping
        if epochs_no_improve >= patience:
            logger.warning(f"Early stopping triggered after {epoch+1} epochs due to no validation improvement.")
            break
            
    # Load best weights of this run for final evaluation & promotion check
    if os.path.exists(run_weights_path):
        model.load_state_dict(torch.load(run_weights_path))
        
    # Evaluate once on the test split using the optimized threshold
    test_labels_all = []
    test_probs_all = []
    
    if not is_dummy_run:
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False)
        with torch.no_grad():
            for images, batch_labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                fake_probs = probs[:, 1]
                
                test_labels_all.extend(batch_labels.cpu().numpy())
                test_probs_all.extend(fake_probs.cpu().numpy())
                
        test_labels_arr = np.array(test_labels_all)
        test_probs_arr = np.array(test_probs_all)
        
        test_preds_arr = (test_probs_arr >= best_opt_thresh).astype(int)
        
        test_acc = float(np.mean(test_preds_arr == test_labels_arr)) * 100
        test_precision = float(precision_score(test_labels_arr, test_preds_arr, zero_division=0))
        test_recall = float(recall_score(test_labels_arr, test_preds_arr, zero_division=0))
        test_f1 = float(f1_score(test_labels_arr, test_preds_arr, zero_division=0))
        test_macro_f1 = float(f1_score(test_labels_arr, test_preds_arr, average="macro", zero_division=0))
        
        try:
            test_auc = float(roc_auc_score(test_labels_arr, test_probs_arr))
        except ValueError:
            test_auc = 0.0
            
        test_cm = confusion_matrix(test_labels_arr, test_preds_arr).tolist()
        test_class_prec, test_class_rec, _, _ = precision_recall_fscore_support(test_labels_arr, test_preds_arr, labels=[0, 1], zero_division=0)
        test_real_precision = float(test_class_prec[0])
        test_real_recall = float(test_class_rec[0])
        test_fake_precision = float(test_class_prec[1])
        test_fake_recall = float(test_class_rec[1])
        
        test_ece = compute_ece(test_labels_arr, test_probs_arr)
        
        logger.info(
            f"TEST SPLIT METRICS (Using Optimized Threshold={best_opt_thresh:.2f}) - "
            f"Accuracy: {test_acc:.2f}%, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, "
            f"F1-Score: {test_f1:.4f}, Macro-F1: {test_macro_f1:.4f}, ROC-AUC: {test_auc:.4f}, ECE: {test_ece:.4f} | "
            f"REAL P/R: {test_real_precision:.4f}/{test_real_recall:.4f}, FAKE P/R: {test_fake_precision:.4f}/{test_fake_recall:.4f}"
        )
        logger.info(f"Test Confusion Matrix: {test_cm}")
    else:
        test_acc = best_val_metrics.get("accuracy", epoch_val_acc)
        test_precision = best_val_metrics.get("precision", val_precision)
        test_recall = best_val_metrics.get("recall", val_recall)
        test_f1 = best_val_metrics.get("f1_score", val_f1)
        test_macro_f1 = best_val_metrics.get("macro_f1", best_val_macro_f1)
        test_auc = best_val_metrics.get("roc_auc", val_auc)
        test_ece = best_val_metrics.get("ece", val_ece)
        test_cm = best_val_metrics.get("confusion_matrix", val_cm)
        test_real_precision = best_val_metrics.get("real_precision", real_precision)
        test_real_recall = best_val_metrics.get("real_recall", real_recall)
        test_fake_precision = best_val_metrics.get("fake_precision", fake_precision)
        test_fake_recall = best_val_metrics.get("fake_recall", fake_recall)

    # Final Metrics Log Structure
    metrics_report = {
        "dataset_type": "pilot" if is_pilot else ("dummy" if is_dummy_run else "real"),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "strategy": active_strategy,
            "pretrained": pretrained,
            "optimized_threshold": best_opt_thresh
        },
        "final_metrics": best_val_metrics,
        "test_metrics": {
            "accuracy": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1,
            "macro_f1": test_macro_f1,
            "roc_auc": test_auc,
            "ece": test_ece,
            "confusion_matrix": test_cm,
            "real_precision": test_real_precision,
            "real_recall": test_real_recall,
            "fake_precision": test_fake_precision,
            "fake_recall": test_fake_recall
        }
    }
    
    # Save the run metrics file
    report_filename = f"image_{active_strategy}_{timestamp}_metrics.json"
    report_dir = os.path.join(PipelineConfig.RESULTS_DIR, "image")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_filename)
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
    logger.info(f"Saved run metrics report to: {report_path}")
    
    # 6. Apply Production Promotion Gate
    is_baseline_run = (active_strategy == "standard" and not pretrained)
    run_promotion_gate(media_type="image", run_metrics=metrics_report, run_weights_path=run_weights_path, is_baseline_run=is_baseline_run)
    
    logger.info("Image model training pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatyaDrishti Image Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=PipelineConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=PipelineConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate small dummy dataset if empty")
    parser.add_argument("--pilot", action="store_true", help="Perform a controlled pilot training run on a subset of real data")
    parser.add_argument("--strategy", type=str, choices=["standard", "weighted_loss", "sampler", "both"], default=None, help="Imbalance handling strategy")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for ResNet18")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-time augmentations")
    
    args = parser.parse_args()
    
    if args.no_augmentation:
        logger.info("Overriding USE_AUGMENTATION to False.")
        PipelineConfig.USE_AUGMENTATION = False
        
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
        pretrained=args.pretrained
    )
