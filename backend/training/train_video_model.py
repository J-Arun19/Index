import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.dataset_loader import DatasetLoader
from backend.training.video_dataset import VideoDataset
from backend.models.video_model import VideoModel

logger = setup_logger("TrainVideoModel")

def generate_dummy_data(num_samples: int = 4):
    """Generates dummy real and fake video files using OpenCV to test the pipeline."""
    import cv2
    import numpy as np
    
    logger.info(f"Generating {num_samples} dummy video samples...")
    os.makedirs(PipelineConfig.VIDEO_REAL_DIR, exist_ok=True)
    os.makedirs(PipelineConfig.VIDEO_FAKE_DIR, exist_ok=True)
    
    # Use MP4V codec which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i in range(num_samples):
        real_path = os.path.join(PipelineConfig.VIDEO_REAL_DIR, f"dummy_real_{i}.mp4")
        fake_path = os.path.join(PipelineConfig.VIDEO_FAKE_DIR, f"dummy_fake_{i}.mp4")
        
        # Build a 20-frame real video
        out_real = cv2.VideoWriter(real_path, fourcc, 10.0, (256, 256))
        for f in range(20):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.circle(frame, (128, 128), 50, (0, 255, 0), -1)  # Real indicator (green face)
            out_real.write(frame)
        out_real.release()
        
        # Build a 20-frame fake video
        out_fake = cv2.VideoWriter(fake_path, fourcc, 10.0, (256, 256))
        for f in range(20):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.circle(frame, (128, 128), 50, (0, 0, 255), -1)  # Fake indicator (red face)
            out_fake.write(frame)
        out_fake.release()
        
    logger.info("Dummy video dataset generation complete.")

def train(epochs: int, batch_size: int, lr: float, generate_dummy: bool):
    # Print configuration summary
    logger.info(PipelineConfig.get_summary())
    
    # Load dataset
    paths, labels = DatasetLoader.scan_video_dataset()
    if (len(paths) == 0 and generate_dummy) or (len(paths) == 0):
        logger.info("No training data found. Generating dummy dataset for testing...")
        generate_dummy_data(4)
        paths, labels = DatasetLoader.scan_video_dataset()
        
    if len(paths) == 0:
        logger.error("No video dataset available. Please add videos to dataset/videos/real and dataset/videos/fake.")
        return
        
    # Create dataset
    dataset = VideoDataset(paths, labels)
    
    # Split train and validation
    val_size = int(len(dataset) * PipelineConfig.VAL_SPLIT)
    train_size = len(dataset) - val_size
    
    # Handle small datasets
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
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False)
    
    # Initialize model, loss, optimizer
    device = torch.device(PipelineConfig.DEVICE)
    model = VideoModel(num_classes=2).to(device)
    
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
        
        for videos, batch_labels in train_loader:
            videos = videos.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * videos.size(0)
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
            for videos, batch_labels in val_loader:
                videos = videos.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(videos)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * videos.size(0)
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
            model.save_weights(PipelineConfig.VIDEO_MODEL_PATH)
            logger.info(f"New best model saved with Val Loss: {best_val_loss:.4f}")
            
    logger.info("Video model training pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SatyaDrishti Video Deepfake Detection Model")
    parser.add_argument("--epochs", type=int, default=PipelineConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=PipelineConfig.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=PipelineConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--generate-dummy", action="store_true", help="Generate small dummy dataset if empty")
    
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, generate_dummy=args.generate_dummy)
