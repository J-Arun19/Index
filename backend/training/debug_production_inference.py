import os
import sys
import argparse
import hashlib
import json
import torch
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.utils.config import PipelineConfig
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.models.image_model import ImageModel
from backend.agents.image_agent import ImageAgent
from backend.agents.fusion_agent import FusionAgent
from backend.agents.metadata_agent import MetadataAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.core.context import AnalysisContext

def debug_inference(image_path: str):
    if not os.path.exists(image_path):
        print(f"ERROR: Image path not found: {image_path}")
        sys.exit(1)
        
    print("=" * 60)
    print(" SATYADRISHTI — PRODUCTION INFERENCE FAILURE DIAGNOSTIC ")
    print("=" * 60)
    
    # 1. File Metadata
    filename = os.path.basename(image_path)
    file_size = os.path.getsize(image_path)
    print(f"File: {image_path}")
    print(f"Media type: Image")
    print(f"File Size: {file_size} bytes")
    
    # 2. Checkpoint Details & SHA-256
    checkpoint_path = PipelineConfig.IMAGE_MODEL_PATH
    print(f"Model checkpoint: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        mtime = os.path.getmtime(checkpoint_path)
        import datetime
        dt_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        with open(checkpoint_path, "rb") as f:
            chk_hash = hashlib.sha256(f.read()).hexdigest()
        print(f"Checkpoint modification timestamp: {dt_str}")
        print(f"Checkpoint SHA-256 Hash: {chk_hash}")
    else:
        print("Checkpoint modification timestamp: N/A (file not found)")
        print("Checkpoint SHA-256 Hash: N/A")
        
    # 3. Model Registry Config
    registry_path = PipelineConfig.REGISTRY_PATH
    reg_threshold = None
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            reg_data = json.load(f)
            reg_threshold = reg_data.get("image", {}).get("optimized_threshold")
            print(f"Registry Entry strategy: {reg_data.get('image', {}).get('strategy')}")
            print(f"Registry Entry pretrained: {reg_data.get('image', {}).get('pretrained')}")
            print(f"Registry Entry optimized_threshold: {reg_threshold}")
            
    # 4. Model Architecture & Config
    print("Model architecture: ResNet-18 (Binary Classifier)")
    print("Model initialization parameters: num_classes=2, pretrained=False")
    print("Class mapping: Class 0 = REAL, Class 1 = FAKE")
    
    # 5. Preprocessing Audit
    preprocessor = ImagePreprocessor()
    img_bgr = cv2.imread(image_path)
    h_orig, w_orig, c_orig = img_bgr.shape if img_bgr is not None else (0,0,0)
    
    faces = preprocessor.detector.detect_faces(img_bgr) if img_bgr is not None else []
    print("\n--- PREPROCESSING DETAILS ---")
    print(f"Original Image Dimensions: {w_orig}x{h_orig} (BGR)")
    print(f"Detected Face Bounding Boxes: {faces}")
    if faces:
        crop_box = faces[0]
        print(f"Face crop dimensions: {crop_box[2]}x{crop_box[3]} at offset ({crop_box[0]}, {crop_box[1]})")
    else:
        print("Face crop dimensions: FULL_IMAGE (No face detected by Haar cascade)")
        
    print(f"Target Tensor Shape: (1, 3, 224, 224)")
    print("Channel Order: RGB")
    print("Normalization: ImageNet Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]")
    print("Value Range: float32 normalized tensor")
    
    # Preprocess image
    tensor = preprocessor.preprocess_image(image_path).unsqueeze(0)
    
    # 6. Direct ML Model Inference (Agentic Layer Bypass)
    print("\n--- DIRECT ML MODEL INFERENCE (BYPASSING AGENTS) ---")
    model = ImageModel(num_classes=2, pretrained=False)
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    model.eval()
    
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]
        real_prob = probabilities[0].item()
        fake_prob = probabilities[1].item()
        raw_argmax = torch.argmax(logits, dim=-1).item()
        
    print(f"Raw logits: {logits.numpy().tolist()[0]}")
    print(f"REAL probability: {real_prob:.4f} ({real_prob * 100:.2f}%)")
    print(f"FAKE probability: {fake_prob:.4f} ({fake_prob * 100:.2f}%)")
    print(f"Raw argmax: {raw_argmax}")
    
    pipeline_thresh = PipelineConfig.CLASSIFICATION_THRESHOLD
    print(f"Inference Threshold (Pipeline Default): {pipeline_thresh}")
    if reg_threshold is not None:
        print(f"Inference Threshold (Model Registry): {reg_threshold}")
        
    direct_verdict_default = "DEEPFAKE" if fake_prob >= pipeline_thresh else "REAL"
    direct_verdict_reg = "DEEPFAKE" if (reg_threshold is not None and fake_prob >= reg_threshold) else "REAL"
    
    print(f"Direct model verdict (at thresh={pipeline_thresh}): {direct_verdict_default}")
    print(f"Direct model verdict (at registry thresh={reg_threshold}): {direct_verdict_reg}")
    
    # 7. Agentic Layer Execution
    print("\n--- AGENTIC LAYER PIPELINE TRACE ---")
    image_agent = ImageAgent()
    img_res = image_agent.analyze(image_path)
    print(f"ImageAgent verdict: {img_res.get('verdict')}")
    print(f"ImageAgent confidence: {img_res.get('confidence_score')}%")
    
    metadata_agent = MetadataAgent()
    meta_res = metadata_agent.analyze(image_path)
    print(f"MetadataAgent anomalies: {meta_res.get('anomalies_detected')}")
    
    context = AnalysisContext(analysis_id="debug_run", user=None, media_type="image")
    context.image_agent_result = img_res
    context.metadata_agent_result = meta_res
    
    fusion_agent = FusionAgent()
    fusion_res = fusion_agent.fuse(context)
    print(f"Fusion verdict: {fusion_res.get('verdict')}")
    print(f"Final probability: {fusion_res.get('final_probability')}")
    print(f"Final confidence: {fusion_res.get('confidence')}%")
    
    print("\n--- FINAL SUMMARY ---")
    print(f"File: {filename}")
    print(f"Media type: Image")
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Model architecture: ResNet-18")
    print(f"Class mapping: 0=REAL, 1=FAKE")
    print(f"Preprocessing details: Face crop resized to 224x224 RGB ImageNet normalized")
    if faces:
        print(f"Face crop dimensions: {faces[0][2]}x{faces[0][3]}")
    else:
        print("Face crop dimensions: Full image")
    print(f"Raw logits: {logits.numpy().tolist()[0]}")
    print(f"REAL probability: {real_prob:.4f}")
    print(f"FAKE probability: {fake_prob:.4f}")
    print(f"Threshold: Default={pipeline_thresh}, Registry={reg_threshold}")
    print(f"Direct model verdict: {direct_verdict_default} (at default thresh 0.5) / {direct_verdict_reg} (at registry thresh {reg_threshold})")
    print(f"ImageAgent verdict: {img_res.get('verdict')}")
    print(f"Fusion verdict: {fusion_res.get('verdict')}")
    print(f"Final API verdict: {fusion_res.get('verdict')}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug SatyaDrishti Production Inference")
    parser.add_argument("--image-path", type=str, default="backend/uploads/14ed56cf-2970-4f3c-ba34-6524dc66e379_Screenshot_2026-08-18_160503.jpg", help="Path to image for diagnostic inference")
    args = parser.parse_args()
    
    debug_inference(args.image_path)
