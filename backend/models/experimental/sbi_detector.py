import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from efficientnet_pytorch import EfficientNet

from backend.utils.logger import setup_logger

logger = setup_logger("SBIDetector")

DEFAULT_SBI_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "sbi_ffc23.tar"
)

class SBIDetector(nn.Module):
    """
    Experimental Self-Blended Images (SBI) detector.
    Original Paper: 'Detecting Deepfakes with Self-Blended Images' (CVPR 2022)
    Authors: Kaede Shiohara, Toshihiko Yamasaki (The University of Tokyo)
    Architecture: EfficientNet-B4 backbone (input size: 380x380)
    Trained on: FF++ c23 with Self-Blended Image synthesis
    License: Academic / Non-Commercial Research (Yamasaki Lab)

    Binary Classification:
      - Class 0: REAL / PRISTINE
      - Class 1: FAKE / MANIPULATED
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        super(SBIDetector, self).__init__()
        self.weights_path = weights_path or DEFAULT_SBI_WEIGHTS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 380

        self.backbone = EfficientNet.from_name("efficientnet-b4", num_classes=2)

        if os.path.exists(self.weights_path):
            try:
                checkpoint = torch.load(self.weights_path, map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint)
                # Strip 'net.' prefix if present in official checkpoint
                clean_state_dict = {
                    k.replace("net.", ""): v for k, v in state_dict.items()
                }
                self.backbone.load_state_dict(clean_state_dict)
                logger.info(f"Loaded SBI EfficientNet-B4 pretrained weights from {self.weights_path}")
            except Exception as e:
                logger.error(f"Failed loading SBI weights from {self.weights_path}: {e}")
                raise
        else:
            logger.warning(f"SBI weights file not found at {self.weights_path}")

        self.to(self.device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Shape (B, 3, 380, 380), float in [0, 1].
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        return self.backbone(x)

    def preprocess_bgr(self, img_bgr: np.ndarray, target_size: int = 380) -> torch.Tensor:
        """
        Converts BGR image/crop to RGB, resizes to target_size (380x380),
        and scales to [0, 1] range as defined in official SBI inference.
        """
        resized = cv2.resize(img_bgr, (target_size, target_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor.to(self.device)

    def predict_crop(self, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs inference on a single BGR face crop or image."""
        t0 = time.time()
        tensor = self.preprocess_bgr(crop_bgr, target_size=self.input_size)
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            real_prob = float(probs[0].item())
            fake_prob = float(probs[1].item())
        dt = round(time.time() - t0, 4)

        return {
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "raw_model_score": round(fake_prob, 4),
            "predicted_label": "FAKE" if fake_prob >= 0.50 else "REAL",
            "inference_time_sec": dt,
        }
