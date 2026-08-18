import torch
from torch.utils.data import Dataset
from typing import List, Optional
from backend.preprocessing.video_preprocessor import VideoPreprocessor

class VideoDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing deepfake detection videos."""
    
    def __init__(self, file_paths: List[str], labels: List[int], preprocessor: Optional[VideoPreprocessor] = None):
        """
        Args:
            file_paths (List[str]): List of absolute video file paths.
            labels (List[int]): List of matching labels (0 for Real, 1 for Fake).
            preprocessor (VideoPreprocessor): Preprocessor instance to handle frame extraction, face detection, etc.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = preprocessor or VideoPreprocessor()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract sampled frames, crop faces, resize, normalize, and pad/truncate to fixed shape
            tensor = self.preprocessor.preprocess_video(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to process video sample '{file_path}': {str(e)}")
            
        # Return video tensor and label
        return tensor, torch.tensor(label, dtype=torch.long)
