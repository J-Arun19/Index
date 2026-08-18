import torch
import torch.nn as nn
import torchvision.models as models
from backend.utils.logger import setup_logger

logger = setup_logger("ImageModel")

class ImageModel(nn.Module):
    """ResNet18-based model adapted for binary classification (Real vs Fake)."""
    
    def __init__(self, num_classes: int = 2):
        super(ImageModel, self).__init__()
        try:
            # Modern torchvision syntax
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            # Older torchvision syntax fallback
            self.backbone = models.resnet18(pretrained=False)
            
        # Replace the final linear classifier layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        logger.info("Successfully initialized ImageModel with ResNet18 backbone.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input batch of images, shape (batch_size, 3, 224, 224).
            
        Returns:
            torch.Tensor: Logits tensor, shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> dict:
        """
        Runs inference and returns structured results.
        
        Args:
            x (torch.Tensor): Input batch of images, shape (batch_size, 3, 224, 224).
            
        Returns:
            dict: Containing logits, probabilities, predicted_class, and confidence_score.
        """
        from backend.utils.config import PipelineConfig
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            # Binary classification assumed: index 0 = REAL, index 1 = FAKE
            fake_probs = probabilities[:, 1]
            predicted_class = (fake_probs >= PipelineConfig.CLASSIFICATION_THRESHOLD).long()
            confidence_score = torch.where(predicted_class == 1, fake_probs, probabilities[:, 0])
            
        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "confidence_score": confidence_score
        }

    def save_weights(self, path: str) -> None:
        """Saves the weights (state dict) of the model."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved ImageModel weights to: {path}")

    def load_weights(self, path: str) -> None:
        """Loads the weights (state dict) of the model."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        logger.info(f"Loaded ImageModel weights from: {path}")
