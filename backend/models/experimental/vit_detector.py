import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    TRANSFORMERS_AVAILABLE = False

from backend.utils.logger import setup_logger
from backend.utils.device_manager import DeviceManager

logger = setup_logger("ViTDetector")

DEFAULT_VIT_MODEL_ID = "dima806/deepfake_vs_real_image_detection"

class ViTDetector(nn.Module):
    """
    Experimental Hugging Face Vision Transformer (ViT) deepfake detector.
    Pretrained checkpoint: 'dima806/deepfake_vs_real_image_detection'
    (based on google/vit-base-patch16-224-in21k, fine-tuned for Real vs Fake classification).
    
    Class mapping (verified from model config.id2label):
      - 0: 'Real'
      - 1: 'Fake'
    """
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        super(ViTDetector, self).__init__()
        self.model_id = model_id or DEFAULT_VIT_MODEL_ID
        self.device = device or DeviceManager.get_device_str()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("HuggingFace 'transformers' package is required for ViTDetector. Install with `pip install transformers`.")

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self.model.eval()
            self.to(self.device)
            logger.info(f"Loaded ViT detector ({self.model_id}) onto {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ViT model from {self.model_id}: {e}")
            raise

    def to(self, *args, **kwargs):
        device_arg = None
        if args:
            device_arg = args[0]
        elif "device" in kwargs:
            device_arg = kwargs["device"]
            
        if device_arg is not None:
            dev_str = str(device_arg)
            self.device = "cuda" if "cuda" in dev_str else "cpu"
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to move inner ViT model to {device_arg}: {e}. Falling back inner model to CPU.")
                    self.device = "cpu"
                    self.model.to("cpu")
        return super(ViTDetector, self).to(*args, **kwargs)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            pixel_values (torch.Tensor): Processed image tensor (B, 3, 224, 224).
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def predict_image(self, img_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Runs inference on an image (path, BGR numpy array, or PIL Image).
        Returns logits, real/fake probabilities, raw_model_score, and label.
        Safely falls back to CPU if GPU inference encounters a CUDA error.
        """
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")

        def _run_forward(target_device: str):
            inputs = self.processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(target_device)
            try:
                with torch.no_grad():
                    logits = self.forward(pixel_values)
                    probs = torch.softmax(logits, dim=-1)[0]
                    real_prob = float(probs[0].item())
                    fake_prob = float(probs[1].item())
                return logits, real_prob, fake_prob
            finally:
                del inputs, pixel_values

        try:
            logits, real_prob, fake_prob = _run_forward(self.device)
        except Exception as e:
            is_cuda_err = "cuda" in str(e).lower() or "out of memory" in str(e).lower()
            if is_cuda_err and self.device != "cpu":
                logger.warning(f"ViTDetector CUDA forward pass failed ({e}). Retrying on CPU fallback...")
                DeviceManager.mark_cuda_unhealthy(f"ViTDetector forward pass failed: {e}")
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
