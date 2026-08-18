import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.models.image_model import ImageModel
from backend.models.video_model import VideoModel
from backend.training.dataset_loader import DeepfakeDataset

logger = setup_logger("SmokeTest")

def run_smoke_test():
    logger.info("========================================")
    logger.info("  RUNNING PIPELINE INTEGRATION SMOKE TEST  ")
    logger.info("========================================")
    
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest file missing: {manifest_path}. Please run generate_manifest.py first.")
        return False
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    
    # 1. TEST IMAGE PIPELINE
    logger.info("\n--- Testing Image Model Pipeline ---")
    try:
        image_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
        if len(image_dataset) == 0:
            logger.warning("No image samples found in dataset splits. Skipping image pipeline check.")
        else:
            # Load 5 samples
            subset_indices = list(range(min(5, len(image_dataset))))
            image_loader = DataLoader(image_dataset, batch_size=len(subset_indices), shuffle=False)
            
            # Fetch a batch
            images, labels = next(iter(image_loader))
            images = images.to(device)
            labels = labels.to(device)
            
            logger.info(f"Successfully loaded image batch. Images shape: {images.shape}, Labels shape: {labels.shape}")
            
            # Initialize model
            image_model = ImageModel(num_classes=2).to(device)
            image_model.eval()
            
            # Forward pass
            with torch.no_grad():
                outputs = image_model(images)
                loss = criterion(outputs, labels)
                
            logger.info(f"Image Forward Pass output shape: {outputs.shape}")
            logger.info(f"Image Forward Pass loss: {loss.item():.4f}")
            logger.info("Image Model Pipeline: SUCCESS")
    except Exception as e:
        logger.error(f"Image Model Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    # 2. TEST VIDEO PIPELINE
    logger.info("\n--- Testing Video Model Pipeline ---")
    try:
        video_dataset = DeepfakeDataset(split="train", media_type="video", manifest_path=manifest_path)
        if len(video_dataset) == 0:
            logger.warning("No video samples found in dataset splits. Skipping video pipeline check.")
        else:
            # Load 2 samples (video preprocessing is heavy, 2 is plenty for a quick check)
            subset_indices = list(range(min(2, len(video_dataset))))
            # We use batch size 1 or 2. Let's load them individually to inspect progress
            video_loader = DataLoader(video_dataset, batch_size=len(subset_indices), shuffle=False)
            
            # Fetch a batch
            logger.info("Extracting and preprocessing video frames (lazy loading)...")
            videos, labels = next(iter(video_loader))
            videos = videos.to(device)
            labels = labels.to(device)
            
            logger.info(f"Successfully loaded video batch. Videos shape: {videos.shape}, Labels shape: {labels.shape}")
            
            # Initialize model
            video_model = VideoModel(num_classes=2).to(device)
            video_model.eval()
            
            # Forward pass
            with torch.no_grad():
                outputs = video_model(videos)
                loss = criterion(outputs, labels)
                
            logger.info(f"Video Forward Pass output shape: {outputs.shape}")
            logger.info(f"Video Forward Pass loss: {loss.item():.4f}")
            logger.info("Video Model Pipeline: SUCCESS")
    except Exception as e:
        logger.error(f"Video Model Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    logger.info("\n========================================")
    logger.info("  [SMOKE TEST PASSED] ALL PIPELINES OK  ")
    logger.info("========================================")
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    import sys
    sys.exit(0 if success else 1)
