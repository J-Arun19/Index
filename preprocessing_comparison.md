# Preprocessing Parity Audit Report

This report compares the preprocessing steps used during training versus production inference for the SatyaDrishti `ImageModel`.

## Comparison Matrix

| Parameter | Training Pipeline (`ImageDataset`) | Production Inference (`ImagePreprocessor`) | Parity Status |
| :--- | :--- | :--- | :--- |
| **Input Channels** | 3 (RGB) | 3 (RGB) | MATCH |
| **Resize Dimensions** | 224 x 224 | 224 x 224 | MATCH |
| **Normalization Mean** | `[0.485, 0.456, 0.406]` (ImageNet) | `[0.485, 0.456, 0.406]` (ImageNet) | MATCH |
| **Normalization Std** | `[0.229, 0.224, 0.225]` (ImageNet) | `[0.229, 0.224, 0.225]` (ImageNet) | MATCH |
| **Channel Order** | CHW (PyTorch Tensor) | CHW (PyTorch Tensor) | MATCH |
| **Value Range** | `[0.0, 1.0]` Float32 | `[0.0, 1.0]` Float32 | MATCH |
| **Batch Dimension** | `(B, 3, 224, 224)` | `(1, 3, 224, 224)` | MATCH |
| **Face Detection** | Haar Cascade Classifier | Haar Cascade Classifier | MATCH |
| **Face Cropping Logic** | Index 0 (`faces[0]`) | Index 0 (`faces[0]`) | MATCH (Flawed) |

## Key Findings

1. **Preprocessing Tensor Parity: MATCH**:
   - The numerical normalization, channel order (RGB CHW), dtype, and scale match perfectly between training and inference pipelines.

2. **Face Crop Selection Mismatch**:
   - Both training and production inference take `faces[0]` directly from OpenCV's Haar Cascade output without sorting bounding boxes by area or confidence.
   - For the failing image (`14ed56cf-2970-4f3c-ba34-6524dc66e379_Screenshot_2026-08-18_160503.jpg`), 3 face boxes were detected: `[(257, 107, 55, 55), (26, 115, 57, 57), (118, 29, 95, 95)]`.
   - Taking `faces[0]` selected a small 55x55 background box instead of the primary facial region (95x95).

3. **Face Crop Visualization**:
   - Saved face crop used during inference: `backend/training/results/failing_image_face_crop.png`.
