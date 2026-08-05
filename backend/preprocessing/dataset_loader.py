import os
from typing import List, Tuple
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger

logger = setup_logger("DatasetLoader")

class DatasetLoader:
    """Helper class to scan local directories for real and fake images and videos."""

    @staticmethod
    def scan_image_dataset() -> Tuple[List[str], List[int]]:
        """
        Scans dataset/images/real and dataset/images/fake folders.
        
        Returns:
            Tuple[List[str], List[int]]: Tuple containing list of image paths and list of binary labels (0 for Real, 1 for Fake).
        """
        paths: List[str] = []
        labels: List[int] = []
        
        # Real images
        real_dir = PipelineConfig.IMAGE_REAL_DIR
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in PipelineConfig.ALLOWED_IMAGE_EXT:
                    paths.append(os.path.join(real_dir, filename))
                    labels.append(0)  # Label 0: Real
                    
        # Fake images
        fake_dir = PipelineConfig.IMAGE_FAKE_DIR
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in PipelineConfig.ALLOWED_IMAGE_EXT:
                    paths.append(os.path.join(fake_dir, filename))
                    labels.append(1)  # Label 1: Fake
                    
        logger.info(f"Scanned Image Dataset: {len(paths)} total samples. (Real: {labels.count(0)}, Fake: {labels.count(1)})")
        return paths, labels

    @staticmethod
    def scan_video_dataset() -> Tuple[List[str], List[int]]:
        """
        Scans dataset/videos/real and dataset/videos/fake folders.
        
        Returns:
            Tuple[List[str], List[int]]: Tuple containing list of video paths and list of binary labels (0 for Real, 1 for Fake).
        """
        paths: List[str] = []
        labels: List[int] = []
        
        # Real videos
        real_dir = PipelineConfig.VIDEO_REAL_DIR
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in PipelineConfig.ALLOWED_VIDEO_EXT:
                    paths.append(os.path.join(real_dir, filename))
                    labels.append(0)  # Label 0: Real
                    
        # Fake videos
        fake_dir = PipelineConfig.VIDEO_FAKE_DIR
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in PipelineConfig.ALLOWED_VIDEO_EXT:
                    paths.append(os.path.join(fake_dir, filename))
                    labels.append(1)  # Label 1: Fake
                    
        logger.info(f"Scanned Video Dataset: {len(paths)} total samples. (Real: {labels.count(0)}, Fake: {labels.count(1)})")
        return paths, labels
