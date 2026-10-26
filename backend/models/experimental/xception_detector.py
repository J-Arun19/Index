import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image

from backend.models.experimental.xception_arch import Xception
from backend.utils.logger import setup_logger
from backend.utils.device_manager import DeviceManager

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
        self.device = device or DeviceManager.get_device_str()
        
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

    def to(self, *args, **kwargs):
        device_arg = None
        if args:
            device_arg = args[0]
        elif "device" in kwargs:
            device_arg = kwargs["device"]
            
        if device_arg is not None:
            dev_str = str(device_arg)
            self.device = "cuda" if "cuda" in dev_str else "cpu"
        return super(XceptionDetector, self).to(*args, **kwargs)

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

    def preprocess_bgr(self, img_bgr: np.ndarray, target_size: int = 256, target_device: Optional[str] = None) -> torch.Tensor:
        """Resizes, converts to RGB, and normalizes to [-1, 1] range."""
        dev = target_device or self.device
        resized = cv2.resize(img_bgr, (target_size, target_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(dev)

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs inference on a single BGR frame or face crop with safe CPU fallback on CUDA failure."""
        def _run_forward(target_dev: str):
            tensor = self.preprocess_bgr(frame_bgr, target_device=target_dev)
            try:
                with torch.no_grad():
                    logits = self.forward(tensor)
                    probs = torch.softmax(logits, dim=-1)[0]
                    real_prob = float(probs[0].item())
                    fake_prob = float(probs[1].item())
                return logits, real_prob, fake_prob
            finally:
                del tensor

        try:
            logits, real_prob, fake_prob = _run_forward(self.device)
        except Exception as e:
            is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
            if is_cuda_err and self.device != "cpu":
                logger.warning(f"XceptionDetector CUDA inference failed ({e}). Executing safe CPU fallback...")
                DeviceManager.mark_cuda_unhealthy(f"XceptionDetector inference failure: {e}")
                self.to("cpu")
                logits, real_prob, fake_prob = _run_forward("cpu")
            else:
                raise

        return {
            "logits": logits.cpu().numpy() if isinstance(logits, torch.Tensor) else logits,
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "raw_model_score": round(fake_prob, 4),
            "predicted_label": "FAKE" if fake_prob >= 0.5 else "REAL"
        }
