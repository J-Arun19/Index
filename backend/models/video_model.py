import torch
import torch.nn as nn
import torchvision.models as models
from backend.utils.logger import setup_logger

logger = setup_logger("VideoModel")

class VideoModel(nn.Module):
    """Frame-based ResNet18 classifier for video files. Extracted frames are pooled temporally."""
    
    def __init__(self, num_classes: int = 2):
        super(VideoModel, self).__init__()
        try:
            # Modern torchvision syntax
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            # Older torchvision syntax fallback
            self.backbone = models.resnet18(pretrained=False)
            
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace the backbone fc layer with Identity to extract raw feature vectors (512-dim)
        self.backbone.fc = nn.Identity()
        
        # Final classification head for aggregated sequence features
        self.fc = nn.Linear(self.feature_dim, num_classes)
        logger.info("Successfully initialized VideoModel with frame-based ResNet18 backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input batch of video frames, shape (batch_size, seq_len, 3, 224, 224).
            
        Returns:
            torch.Tensor: Logits tensor, shape (batch_size, num_classes).
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten batch and sequence dimensions: (batch_size * seq_len, 3, H, W)
        x_flat = x.view(batch_size * seq_len, c, h, w)
        
        # Pass through the CNN feature extractor
        features_flat = self.backbone(x_flat)  # (batch_size * seq_len, feature_dim)
        
        # Unflatten to sequence: (batch_size, seq_len, feature_dim)
        features = features_flat.view(batch_size, seq_len, self.feature_dim)
        
        # Aggregate across the sequence dimension (average pooling over frames)
        pooled_features = features.mean(dim=1)  # (batch_size, feature_dim)
        
        # Classify the pooled representation
        logits = self.fc(pooled_features)  # (batch_size, num_classes)
        return logits

    def save_weights(self, path: str) -> None:
        """Saves the weights (state dict) of the model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved VideoModel weights to: {path}")

    def load_weights(self, path: str) -> None:
        """Loads the weights (state dict) of the model."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        logger.info(f"Loaded VideoModel weights from: {path}")
