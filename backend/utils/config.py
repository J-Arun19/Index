import os
import torch

class PipelineConfig:
    # Directory setup
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset Directories
    DATASET_DIR = os.path.join(BACKEND_DIR, 'dataset')
    IMAGE_DATASET_DIR = os.path.join(DATASET_DIR, 'images')
    VIDEO_DATASET_DIR = os.path.join(DATASET_DIR, 'videos')
    PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
    
    IMAGE_REAL_DIR = os.path.join(IMAGE_DATASET_DIR, 'real')
    IMAGE_FAKE_DIR = os.path.join(IMAGE_DATASET_DIR, 'fake')
    
    VIDEO_REAL_DIR = os.path.join(VIDEO_DATASET_DIR, 'real')
    VIDEO_FAKE_DIR = os.path.join(VIDEO_DATASET_DIR, 'fake')
    
    # Model Weights Directory
    WEIGHTS_DIR = os.path.join(BACKEND_DIR, 'models', 'weights')
    IMAGE_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'image_model.pth')
    VIDEO_MODEL_PATH = os.path.join(WEIGHTS_DIR, 'video_model.pth')
    
    # Preprocessing Configurations
    IMAGE_SIZE = (224, 224)  # Height, Width for standard architectures like ResNet18
    ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    ALLOWED_VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    # Video specific preprocessing configuration
    FRAME_INTERVAL = 10      # Process every Nth frame (adjustable)
    MAX_FRAMES_PER_VIDEO = 30 # Cap maximum number of frames extracted per video to maintain resource constraints
    
    # Training Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    VAL_SPLIT = 0.2          # 20% validation split
    
    # Device Selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_summary(cls) -> str:
        """Returns a string summary of the configuration for logging."""
        return (
            f"--- Pipeline Configuration ---\n"
            f"Device: {cls.DEVICE}\n"
            f"Image Size: {cls.IMAGE_SIZE}\n"
            f"Batch Size: {cls.BATCH_SIZE}\n"
            f"Epochs: {cls.EPOCHS}\n"
            f"Learning Rate: {cls.LEARNING_RATE}\n"
            f"Frame Sampling Interval: {cls.FRAME_INTERVAL}\n"
            f"Max Frames per Video: {cls.MAX_FRAMES_PER_VIDEO}\n"
            f"Image Dataset path: {cls.IMAGE_DATASET_DIR}\n"
            f"Video Dataset path: {cls.VIDEO_DATASET_DIR}\n"
            f"-----------------------------"
        )
