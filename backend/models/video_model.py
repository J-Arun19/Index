import torch
import torch.nn as nn
import torchvision.models as models
from backend.utils.logger import setup_logger

logger = setup_logger("VideoModel")

class VideoModel(nn.Module):
    """
    VideoModel classifier using a ResNet18 backbone for spatial feature extraction 
    followed by a unidirectional LSTM for temporal sequence modeling.
    
    Class mapping:
        - Class 0: REAL
        - Class 1: FAKE / DEEPFAKE
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = False, temporal_pooling: str = None):
        super(VideoModel, self).__init__()
        self.classification_threshold = None
        
        from backend.utils.config import PipelineConfig
        if temporal_pooling is None:
            raw_pooling = getattr(PipelineConfig, "TEMPORAL_POOLING", "lstm")
            if isinstance(raw_pooling, str):
                temporal_pooling = raw_pooling
            else:
                temporal_pooling = "lstm"
        self.temporal_pooling = temporal_pooling.lower()
        
        try:
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                self.backbone = models.resnet18(weights=None)
        except (TypeError, AttributeError):
            self.backbone = models.resnet18(pretrained=pretrained)
            
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace the backbone fc layer with Identity to extract raw feature vectors (512-dim)
        self.backbone.fc = nn.Identity()
        
        if self.temporal_pooling == "lstm":
            # Add standard unidirectional LSTM for temporal modeling
            self.lstm = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=256,
                num_layers=1,
                batch_first=True
            )
            # Final classification head for temporal representations (projects from LSTM hidden dim to classes)
            self.fc = nn.Linear(256, num_classes)
        elif self.temporal_pooling == "mean":
            # For mean pooling, we take the average of frame features (512-dim) across temporal dimension,
            # and map it directly to classes using a linear head.
            self.fc = nn.Linear(self.feature_dim, num_classes)
        elif self.temporal_pooling == "max":
            # For max pooling, we take the max of frame features (512-dim) across temporal dimension,
            # and map it directly to classes using a linear head.
            self.fc = nn.Linear(self.feature_dim, num_classes)
        else:
            raise ValueError(f"Unknown temporal_pooling strategy: {temporal_pooling}")
            
        logger.info(f"Successfully initialized VideoModel (pretrained={pretrained}) with ResNet18 backbone and '{self.temporal_pooling}' temporal pooling.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input batch of video frames, shape (batch_size, seq_len, 3, 224, 224).
            
        Returns:
            torch.Tensor: Logits tensor, shape (batch_size, num_classes).
        """
        # Input shape: (batch_size, sequence_length, 3, 224, 224)
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten batch and sequence dimensions: (batch_size * sequence_length, 3, 224, 224)
        x_flat = x.view(batch_size * seq_len, c, h, w)
        
        # Pass through the CNN feature extractor: (batch_size * sequence_length, 512)
        features_flat = self.backbone(x_flat)
        
        # Reshaped sequence: (batch_size, sequence_length, 512)
        features = features_flat.view(batch_size, seq_len, self.feature_dim)
        
        if self.temporal_pooling == "lstm":
            # Pass sequence through the unidirectional LSTM: (batch_size, sequence_length, 256)
            lstm_out, _ = self.lstm(features)
            # Final temporal representation: (batch_size, 256)
            temporal_out = lstm_out[:, -1, :]
        elif self.temporal_pooling == "mean":
            # Mean temporal pooling across frames: (batch_size, 512)
            temporal_out = torch.mean(features, dim=1)
        elif self.temporal_pooling == "max":
            # Max temporal pooling across frames: (batch_size, 512)
            temporal_out = torch.max(features, dim=1)[0]
            
        # Final logits: (batch_size, 2)
        logits = self.fc(temporal_out)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = None) -> dict:
        """
        Runs inference and returns structured results.
        
        Args:
            x (torch.Tensor): Input batch of video frames, shape (batch_size, seq_len, 3, 224, 224).
            threshold (float, optional): Custom classification threshold.
            
        Returns:
            dict: Containing logits, probabilities, predicted_class, and confidence_score.
        """
        from backend.utils.config import PipelineConfig
        if threshold is None:
            threshold = self.classification_threshold if self.classification_threshold is not None else PipelineConfig.CLASSIFICATION_THRESHOLD
            
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=-1)
            # Binary classification assumed: index 0 = REAL, index 1 = FAKE
            fake_probs = probabilities[:, 1]
            predicted_class = (fake_probs >= threshold).long()
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
        logger.info(f"Saved VideoModel weights to: {path}")

    def load_weights(self, path: str) -> None:
        """Loads the weights (state dict) of the model."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        logger.info(f"Loaded VideoModel weights from: {path}")
