import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_backend_dir, '.env'))

class PipelineConfig:
    # Directory setup
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dataset Root Directory
    SATYADRISHTI_DATASET_ROOT = os.environ.get("SATYADRISHTI_DATASET_ROOT", r"D:\Download\SatyaDrishti_Datasets")
    
    # Dataset Directories (Local/Legacy defaults)
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
    MAX_FRAMES_PER_VIDEO = 8 # Default 8 sampled frames per video for GTX 1650 4GB GPU
    
    # Training Hyperparameters
    BATCH_SIZE = 4           # Conservative batch size suitable for GTX 1650 4GB GPU
    LEARNING_RATE = 1e-4
    BACKBONE_LR = 1e-5
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10
    PATIENCE = 5
    CLASSIFICATION_THRESHOLD = 0.5
    VAL_SPLIT = 0.2
    
    # Class Imbalance Strategies: 'standard', 'weighted_loss', 'sampler', 'both'
    IMBALANCE_STRATEGY = 'weighted_loss'
    
    # Temporal aggregation strategy: 'lstm' or 'mean'
    TEMPORAL_POOLING = 'lstm'
    
    # Data Augmentation Toggle
    USE_AUGMENTATION = True
    
    # Registry & Results
    REGISTRY_PATH = os.path.join(BACKEND_DIR, 'models', 'model_registry.json')
    RESULTS_DIR = os.path.join(BACKEND_DIR, 'training', 'results')
    
    # Agentic AI Configurations
    AGENTIC_CONFIDENCE_THRESHOLD = 0.80
    WEIGHT_PRIMARY = 0.50
    WEIGHT_ARTIFACT = 0.35
    WEIGHT_METADATA = 0.15
    
    # Device Selection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def update_config(cls, batch_size: int = None, epochs: int = None, lr: float = None, strategy: str = None):
        """Updates runtime configuration settings with explicit values from CLI or checkpoint."""
        if batch_size is not None:
            cls.BATCH_SIZE = batch_size
        if epochs is not None:
            cls.EPOCHS = epochs
        if lr is not None:
            cls.LEARNING_RATE = lr
        if strategy is not None:
            cls.IMBALANCE_STRATEGY = strategy

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
            f"Dataset Root path: {cls.SATYADRISHTI_DATASET_ROOT}\n"
            f"-----------------------------"
        )

