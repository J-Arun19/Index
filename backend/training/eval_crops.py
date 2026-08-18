import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.models.image_model import ImageModel
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.utils.config import PipelineConfig

def main():
    img_path = 'backend/uploads/14ed56cf-2970-4f3c-ba34-6524dc66e379_Screenshot_2026-08-18_160503.jpg'
    pre = ImagePreprocessor()
    img = cv2.imread(img_path)
    
    model = ImageModel(num_classes=2)
    model.load_weights(PipelineConfig.IMAGE_MODEL_PATH)
    model.eval()
    
    faces = pre.detector.detect_faces(img)
    print("Detected faces:", faces)
    
    for i, f in enumerate(faces):
        crop = pre.detector.crop_face(img, f)
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = pre.apply_augmentation_and_normalization(rgb).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=-1)[0]
            print(f"Face crop {i} (bbox={f}): logits={logits.numpy().tolist()[0]} FAKE_prob={prob[1].item():.4f} REAL_prob={prob[0].item():.4f}")
            
    img_full_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
    tensor_full = pre.apply_augmentation_and_normalization(img_full_rgb).unsqueeze(0)
    with torch.no_grad():
        logits_full = model(tensor_full)
        prob_full = torch.softmax(logits_full, dim=-1)[0]
        print(f"Full image (no crop): logits={logits_full.numpy().tolist()[0]} FAKE_prob={prob_full[1].item():.4f} REAL_prob={prob_full[0].item():.4f}")

if __name__ == "__main__":
    main()
