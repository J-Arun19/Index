import os
import argparse
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
    
    logger.info(f"Generating {num_samples} dummy training samples...")
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
        
    logger.info("Dummy image dataset generation complete.")

def train(epochs: int, batch_size: int, lr: float, generate_dummy: bool):
    # Print configuration summary
    logger.info(PipelineConfig.get_summary())
    
    # Generate dummy data if requested or if directories are empty
    paths, labels = DatasetLoader.scan_image_dataset()
    if (len(paths) == 0 and generate_dummy) or (len(paths) == 0):
        logger.info("No training data found. Generating dummy dataset for testing...")
        generate_dummy_data(10)
        paths, labels = DatasetLoader.scan_image_dataset()
        
    if len(paths) == 0:
        logger.error("No dataset available. Please add images to dataset/images/real and dataset/images/fake.")
        return
        
    # Create dataset
    dataset = ImageDataset(paths, labels)
    
    # Split train and validation
    val_size = int(len(dataset) * PipelineConfig.VAL_SPLIT)
    train_size = len(dataset) - val_size
    
    # Handle small datasets (ensure at least 1 sample in validation if dataset is very small)
    if val_size == 0 and len(dataset) > 1:
        val_size = 1
        train_size = len(dataset) - 1
        
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Split dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # Create DataLoaders
    # set shuffle=True for train, drop_last=False to process all samples
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    
    # Initialize model, loss, optimizer
    device = torch.device(PipelineConfig.DEVICE)
    model = ImageModel(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    logger.info(f"Starting training on device: {device}...")
    
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
        
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images = images.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        epoch_val_acc = (correct_val / total_val) * 100 if total_val > 0 else 0.0
        
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%"
        )
        
        # Save best weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model.save_weights(PipelineConfig.IMAGE_MODEL_PATH)
            logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            
    logger.info("Image model training pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatyaDrishti Image Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=PipelineConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=PipelineConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate small dummy dataset if empty")
    
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, generate_dummy=args.generate_dummy)
