import torch
from torch.utils.data import Dataset
from typing import List, Optional
from backend.preprocessing.image_preprocessor import ImagePreprocessor

class ImageDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing deepfake detection images."""
    
    def __init__(self, file_paths: List[str], labels: List[int], preprocessor: Optional[ImagePreprocessor] = None):
        """
        Args:
            file_paths (List[str]): List of absolute image file paths.
            labels (List[int]): List of matching labels (0 for Real, 1 for Fake).
            preprocessor (ImagePreprocessor): Preprocessor instance to handle face detection, resizing, etc.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = preprocessor or ImagePreprocessor()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract face, crop, resize, normalize, and convert to PyTorch tensor
            tensor = self.preprocessor.preprocess_image(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to process image sample '{file_path}': {str(e)}")
            
        # Return image tensor and long tensor label (useful for CrossEntropyLoss)
        return tensor, torch.tensor(label, dtype=torch.long)
