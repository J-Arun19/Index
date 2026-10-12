import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image

from backend.models.experimental.xception_arch import Xception
from backend.utils.logger import setup_logger

logger = setup_logger("XceptionDetector")

DEFAULT_XCEPTION_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "xception_best.pth"
)

class XceptionDetector(nn.Module):
    """
    Experimental DeepfakeBench Xception detector.
    Pretrained on FaceForensics++ (c23 compression), official DeepfakeBench release v1.0.1.
    Binary classification:
      - Class 0: REAL
      - Class 1: FAKE / MANIPULATED
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        super(XceptionDetector, self).__init__()
        self.weights_path = weights_path or DEFAULT_XCEPTION_WEIGHTS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        cfg = {"num_classes": 2, "mode": "original", "inc": 3, "dropout": False}
        self.backbone = Xception(cfg)
        
        if os.path.exists(self.weights_path):
            try:
                state_dict = torch.load(self.weights_path, map_location="cpu")
                self.load_state_dict(state_dict)
                logger.info(f"Loaded Xception pretrained weights from {self.weights_path}")
            except Exception as e:
                logger.error(f"Failed loading Xception weights from {self.weights_path}: {e}")
                raise
        else:
            logger.warning(f"Xception weights file not found at {self.weights_path}")

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (B, 3, 256, 256) normalized to [-1, 1].
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        out, _ = self.backbone(x)
        return out

    def preprocess_bgr(self, img_bgr: np.ndarray, target_size: int = 256) -> torch.Tensor:
        """Resizes, converts to RGB, and normalizes to [-1, 1] range."""
        resized = cv2.resize(img_bgr, (target_size, target_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs inference on a single BGR frame or face crop."""
        tensor = self.preprocess_bgr(frame_bgr)
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            real_prob = float(probs[0].item())
            fake_prob = float(probs[1].item())

        return {
            "logits": logits.cpu().numpy(),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "raw_model_score": round(fake_prob, 4),
            "predicted_label": "FAKE" if fake_prob >= 0.5 else "REAL"
        }
