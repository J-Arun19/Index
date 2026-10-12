import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.utils.logger import setup_logger

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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Loaded ViT detector ({self.model_id}) onto {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ViT model from {self.model_id}: {e}")
            raise

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

        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            logits = self.forward(pixel_values)
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
