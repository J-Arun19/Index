import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.dataset_loader import DatasetLoader
from backend.training.image_dataset import ImageDataset
from backend.models.image_model import ImageModel

logger = setup_logger("TrainImageModel")

def generate_dummy_data(num_samples: int = 10):
    """Generates dummy real and fake images to test the training pipeline."""
    import cv2
    import numpy as np
    
    logger.warning("[DUMMY DATA — TEST ONLY] Generating dummy training samples...")
    os.makedirs(PipelineConfig.IMAGE_REAL_DIR, exist_ok=True)
    os.makedirs(PipelineConfig.IMAGE_FAKE_DIR, exist_ok=True)
    
    for i in range(num_samples):
        # Create random noise image (simulating a face frame)
        img_real = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        # Draw a simple circle to simulate face boundary
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
    # Group samples by file_path
    by_file = {}
    for s in dataset.samples:
        by_file.setdefault(s["file_path"], []).append(s)
        
    # Classify files by their label (assuming all frames of a video have the same label)
    files_by_label = {0: [], 1: []}
    for file_path, samples in by_file.items():
        label = samples[0]["label"]
        files_by_label[label].append((file_path, samples))
        
    num_files_0 = len(files_by_label[0])
    num_files_1 = len(files_by_label[1])
    
    if num_files_0 == 0 or num_files_1 == 0:
        raise ValueError(f"Cannot subset dataset: class 0 files = {num_files_0}, class 1 files = {num_files_1}")
        
    # We want target_size samples in total. Let's calculate the ratio of samples we need.
    total_samples = len(dataset.samples)
    samples_0 = sum(len(s) for _, s in files_by_label[0])
    samples_1 = sum(len(s) for _, s in files_by_label[1])
    
    ratio_0 = samples_0 / total_samples
    target_samples_0 = max(1, int(target_size * ratio_0))
    target_samples_1 = max(1, target_size - target_samples_0)
    
    # Select files deterministically until we reach the target number of samples for each class
    selected_samples = []
    
    # Class 0 selection
    accumulated_0 = 0
    # Sort files by path to make it deterministic
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
    
    # Shuffle deterministically
    rng = random.Random(42)
    rng.shuffle(dataset.samples)
    
    logger.info(f"Subset dataset deterministically to {len(dataset.samples)} samples (target: {target_size}) using grouped files to optimize video I/O.")

def train(epochs: int, batch_size: int, lr: float, generate_dummy: bool, is_pilot: bool = False):
    # Print configuration summary
    logger.info(PipelineConfig.get_summary())
    
    is_dummy_run = generate_dummy
    
    if is_dummy_run:
        logger.warning("[DUMMY DATA — TEST ONLY] --generate-dummy flag was provided. Setting up dummy dataset...")
        paths, labels = DatasetLoader.scan_image_dataset()
        if len(paths) == 0:
            logger.warning("[DUMMY DATA — TEST ONLY] Generating dummy dataset because --generate-dummy was specified...")
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
            raise FileNotFoundError(
                f"Manifest file '{manifest_path}' is missing. Please run "
                f"'python -m backend.training.generate_manifest' first to index the dataset, "
                f"or run with --generate-dummy for testing."
            )
            
        logger.info(f"Loading real datasets from manifest: {manifest_path}")
        from backend.training.dataset_loader import DeepfakeDataset
        
        train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
        val_dataset = DeepfakeDataset(split="val", media_type="image", manifest_path=manifest_path)
        
        if len(train_dataset) == 0:
            raise ValueError("No training samples found in the image dataset manifest. Run generate_manifest.py first.")
        if len(val_dataset) == 0:
            raise ValueError("No validation samples found in the image dataset manifest. Run generate_manifest.py first.")
            
        # If pilot run, subset datasets deterministically
        if is_pilot:
            logger.info("Configuring PILOT RUN subset size (target: 3000 train, 1000 val)...")
            subset_dataset_deterministically(train_dataset, 3000)
            subset_dataset_deterministically(val_dataset, 1000)
            
        # Pre-cache samples in RAM if subset is small (e.g. pilot run) to avoid OpenCV seeking bottlenecks
        if len(train_dataset) <= 5000:
            logger.info("Pre-caching train and validation datasets to RAM for rapid execution...")
            train_dataset.pre_cache_samples()
            val_dataset.pre_cache_samples()
            
        # Two-class training safeguard
        train_labels = [s["label"] for s in train_dataset.samples]
        unique_labels = set(train_labels)
        if len(unique_labels) < 2:
            raise ValueError(f"ImageModel training stopped: Training dataset must contain both classes (0 and 1), but only found: {list(unique_labels)}")
            
        # Detailed reports before training
        logger.info("Training Dataset Distribution Summary:")
        logger.info(f"- Class Labels: REAL (0): {train_labels.count(0)}, FAKE (1): {train_labels.count(1)}")
        
        sources = [s["dataset_source"] for s in train_dataset.samples]
        source_counts = {src: sources.count(src) for src in set(sources)}
        logger.info(f"- Source Datasets: {source_counts}")
        
        origins = [s["sample_origin"] for s in train_dataset.samples]
        origin_counts = {orig: origins.count(orig) for orig in set(origins)}
        logger.info(f"- Sample Origins: {origin_counts}")
        
        log_prefix = "[PILOT TRAINING] " if is_pilot else "[REAL TRAINING] "
        
        # Log split strategy details
        logger.info("Dataset Split Strategy:")
        logger.info("- Celeb-DF: Subject identity-based partition (preventing identity leakage)")
        logger.info("- FaceForensics++: Sequence-based official partition")
        logger.info("- DeepFakeDetection: Disjoint actor components partition")
        logger.info(f"Loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    # Create DataLoaders
    # set shuffle=True for train, drop_last=False to process all samples
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    
    # Initialize model, loss, optimizer
    device = torch.device(PipelineConfig.DEVICE)
    model = ImageModel(num_classes=2).to(device)
    
    # Calculate class weights for real/pilot training, otherwise use None
    class_weights = None
    if not is_dummy_run:
        class_weights = train_dataset.get_class_weights().to(device)
        logger.info(f"Applying class-weighted CrossEntropyLoss. Weights: REAL={class_weights[0].item():.4f}, FAKE={class_weights[1].item():.4f}")
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    logger.info(f"{log_prefix}Starting training on device: {device}...")
    
    # We will compute and log all validation metrics at the end of each epoch
    for epoch in range(epochs):
        # Training Phase
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
        correct_val = 0
        total_val = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * images.size(0)
                total_val += batch_labels.size(0)
                
                # Get softmax probabilities for metric aggregation
                probs = torch.softmax(outputs, dim=1)
                fake_probs = probs[:, 1]
                preds = (fake_probs >= PipelineConfig.CLASSIFICATION_THRESHOLD).long()
                
                correct_val += (preds == batch_labels).sum().item()
                
                all_labels.extend(batch_labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(fake_probs.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        epoch_val_acc = (correct_val / total_val) * 100 if total_val > 0 else 0.0
        
        # Calculate sklearn validation metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
        
        val_precision = float(precision_score(all_labels, all_preds, zero_division=0))
        val_recall = float(recall_score(all_labels, all_preds, zero_division=0))
        val_f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        try:
            val_auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            val_auc = 0.0
            
        val_cm = confusion_matrix(all_labels, all_preds).tolist()
        
        # Class specific metrics
        class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], zero_division=0)
        real_precision = float(class_prec[0])
        real_recall = float(class_rec[0])
        fake_precision = float(class_prec[1])
        fake_recall = float(class_rec[1])
        macro_f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
        
        logger.info(
            f"{log_prefix}Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )
        logger.info(
            f"{log_prefix}Validation Metrics - "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
            f"F1-Score: {val_f1:.4f}, ROC-AUC: {val_auc:.4f} | "
            f"REAL P/R: {real_precision:.4f}/{real_recall:.4f}, "
            f"FAKE P/R: {fake_precision:.4f}/{fake_recall:.4f}, "
            f"Macro F1: {macro_f1:.4f}"
        )
        logger.info(f"{log_prefix}Confusion Matrix: {val_cm}")
        
        # Save best weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            weights_path = os.path.join(PipelineConfig.BACKEND_DIR, "models", "weights", "pilot_image_model.pth") if is_pilot else PipelineConfig.IMAGE_MODEL_PATH
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            model.save_weights(weights_path)
            logger.info(f"{log_prefix}New best model saved to {weights_path} with Val Loss: {best_val_loss:.4f}")
            
    # Save validation metrics to a machine-readable JSON report
    import json
    metrics_report = {
        "dataset_type": "pilot" if is_pilot else ("dummy" if is_dummy_run else "real"),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "classification_threshold": PipelineConfig.CLASSIFICATION_THRESHOLD
        },
        "final_metrics": {
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "accuracy": epoch_val_acc,
            "precision": val_precision,
            "recall": val_recall,
            "f1_score": val_f1,
            "macro_f1": macro_f1,
            "roc_auc": val_auc,
            "confusion_matrix": val_cm,
            "real_precision": real_precision,
            "real_recall": real_recall,
            "fake_precision": fake_precision,
            "fake_recall": fake_recall
        }
    }
    
    if is_pilot:
        report_filename = "pilot_image_metrics.json"
    else:
        report_filename = "dummy_image_training_metrics.json" if is_dummy_run else "image_training_metrics.json"
        
    report_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", report_filename)
    with open(report_path, "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    logger.info(f"{log_prefix}Saved final metrics report to: {report_path}")
    logger.info("Image model training pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatyaDrishti Image Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=PipelineConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=PipelineConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate small dummy dataset if empty")
    parser.add_argument("--pilot", action="store_true", help="Perform a controlled pilot training run on a subset of real data")
    
    args = parser.parse_args()
    
    # If pilot run, set epoch default to 2
    epochs_to_run = args.epochs
    if args.pilot and args.epochs == PipelineConfig.EPOCHS:
        epochs_to_run = 2
        
    train(
        epochs=epochs_to_run, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        generate_dummy=args.generate_dummy,
        is_pilot=args.pilot
    )
