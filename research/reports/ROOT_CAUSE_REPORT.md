# Production Inference Failure Root-Cause Report

## Executive Summary
- **Failing File**: `14ed56cf-2970-4f3c-ba34-6524dc66e379_Screenshot_2026-08-18_160503.jpg`
- **Reported Verdict**: `REAL`
- **Reported Confidence**: `88.46%`
- **Ground Truth**: Known Face-Swap / Deepfake Image
- **Primary Failure Classification**: `H. MULTIPLE ROOT CAUSES`

---

## Breakdown of Root Causes

### 1. Threshold Disparity & Disregard Bug (Threshold Bug)
- `model_registry.json` specifies `"optimized_threshold": 0.04` for the active `ImageModel` checkpoint.
- `ImageAgent` calls `model.predict(tensor)` without passing the registered threshold, causing `ImageModel` to default to `PipelineConfig.CLASSIFICATION_THRESHOLD = 0.50`.
- At `threshold = 0.50`, the model's FAKE probability (`0.1174`) is below 0.50, producing a `REAL` prediction.
- If the registered threshold (`0.04`) had been applied, the direct model verdict would have been **`DEEPFAKE`** (`0.1174 >= 0.04`).

### 2. Machine Learning Model Representation Failure (Weak ML Model)
- The trained ResNet-18 model produced a raw logit of `[-0.2633]` for Class 1 (FAKE) versus `[1.7538]` for Class 0 (REAL).
- Even on the primary face crop (95x95), the model assigned a FAKE probability of only `0.2250` (22.5%).
- The model exhibits low calibration and weak feature representations on unaligned / compressed face-swap images.

### 3. Face Bounding Box Selection Defect (Preprocessing Parity Issue)
- OpenCV Haar Cascade detected 3 face boxes. The preprocessor picked `faces[0]` (`55x55`), which was a secondary/background ROI rather than the largest primary face (`95x95`).

---

## Diagnostic Pipeline Trace

```
INPUT (Screenshot_2026-08-18_160503.jpg)
 ↓
MEDIA VALIDATION (Valid PNG/JPG, Status: SUCCESS)
 ↓
IMAGE PREPROCESSING (Haar Cascade ROI: 55x55 -> 224x224 RGB ImageNet Normalized)
 ↓
IMAGE MODEL (ResNet-18, Weights SHA-256: cc411dafbc845e1fb65197487a87fe409bc0c7ff1520c8e4e83b4aceb4fa8aeb)
 ↓
RAW LOGITS ([1.7538, -0.2633])
 ↓
PROBABILITIES (REAL: 0.8826, FAKE: 0.1174)
 ↓
IMAGE AGENT (Applied Thresh=0.5 -> Verdict: REAL, Confidence: 88%)
 ↓
METADATA AGENT (Anomalies: False)
 ↓
CONFIDENCE AGENT (Tier: High, Score: 88%)
 ↓
ARTIFACT AGENT (Skipped: Confidence 0.88 >= 0.80)
 ↓
FUSION AGENT (Weights: Primary=0.50, Metadata=0.15 -> Final Prob Fake: 0.1154)
 ↓
FINAL VERDICT: REAL (Confidence: 88.46%)
```

---

## Controlled Benchmark Results (10 Samples: 5 REAL, 5 FAKE)

- **Accuracy**: `50.00%`
- **Macro-F1**: `0.3333`
- **REAL Recall**: `1.0000`
- **FAKE Recall**: `0.0000`
- **ROC-AUC**: `0.9600`

---

## Phase 6 Proposal for ImageModel Improvement

When full training is authorized in the future:
1. **Threshold Synchronization**: Modify `ImageAgent` to read `optimized_threshold` directly from `model_registry.json`.
2. **Pretrained ResNet-18 Backbone**: Train with ImageNet pre-trained weights with fine-tuning on layer 4 and fc head.
3. **Primary Face Alignment**: Update `FaceDetector` to sort bounding boxes by area (`w * h`) and pick the largest detected face.
4. **Data Splitting Integrity**: Enforce identity-disjoint splitting between train/val/test.
5. **Hard Negative Mining**: Treat the failing face-swap image as a hard negative example in training.
