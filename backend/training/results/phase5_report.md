# Phase 5 Deepfake Model Optimization Report

## Executive Summary

Phase 5 executed a controlled experimental cycle for both the **ImageModel** and **VideoModel** in SatyaDrishti. The objective was to address performance bottlenecks identified during Phase 4—specifically testing class-balancing techniques for the ImageModel to resolve false positive bias, and architectural/pooling refinements for the VideoModel to overcome poor temporal aggregation.

All 7 experimental runs were executed under strict safety protocol:
- **Zero modification** to the external dataset (`D:\Download\SatyaDrishti_Datasets`).
- **No changes** to production checkpoints (`image_model.pth`, `video_model.pth`) or `model_registry.json`.
- **Deterministic seeds** locked across all dataset sampling and network initialization.
- Evaluation performed on isolated pilot subsets with automated data leakage checks.

---

## Controlled Experiment Matrix

### 1. Image Model Experiments (I1 – I3)

| Metric | I1: Baseline Standard | I2: Weighted Loss | I3: WeightedRandomSampler |
| :--- | :--- | :--- | :--- |
| **Strategy** | Standard CrossEntropy | Weighted CrossEntropy (REAL=3.125, FAKE=0.595) | Class-Balanced Sampler |
| **Epochs** | 3 | 2 | 2 |
| **Optimal Threshold** | 0.35 | 0.10 | 0.18 |
| **Val Accuracy** | **82.90%** | 82.10% | 81.90% |
| **Val Macro-F1** | **0.7372** | 0.7187 | 0.7177 |
| **Val ROC-AUC** | **0.8281** | 0.7857 | 0.7772 |
| **Val REAL Recall** | **56.67%** | 51.90% | 52.38% |
| **Val FAKE Recall** | 89.87% | **90.13%** | 89.75% |
| **Val ECE** | **0.1459** | 0.4006 | 0.3258 |
| **Test Accuracy** | 81.39% | 81.49% | **82.39%** |
| **Test Macro-F1** | **0.6843** | 0.6645 | 0.6294 |
| **Test ROC-AUC** | **0.7878** | 0.7756 | 0.7776 |
| **Test REAL Recall** | **43.50%** | 36.50% | 25.00% |

#### Key Findings — ImageModel:
1. **Class-Balancing Strategies Failed to Improve REAL Recall**: Neither `weighted_loss` (I2) nor `WeightedRandomSampler` (I3) improved performance over the standard loss baseline. They pushed the threshold optimization to lower values (0.10 and 0.18), aggravating false positive rates on REAL images.
2. **Standard Loss (I1) is Superior**: Standard CrossEntropy with optimal threshold 0.35 achieves the highest Validation Macro-F1 (0.7372), highest Validation ROC-AUC (0.8281), lowest calibration error (ECE = 0.1459), and highest REAL Recall (56.67%).
3. **Classification**:
   - **I1**: **PROMISING** (Candidate for future full-dataset training)
   - **I2 & I3**: **REJECTED**

---

### 2. Video Model Experiments (V1 – V4)

| Metric | V1: Mean Pooling (Base) | V2: Max Pooling | V3: Max Pooling + Weighted Loss | V4: Mean Pooling + Weighted Loss + Layer3 Unfrozen |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal Pooling** | Mean | Max | Max | Mean |
| **Imbalance Strategy** | Standard | Standard | Weighted Loss | Weighted Loss |
| **Backbone Frozen** | Layers 1–3 | Layers 1–3 | Layers 1–3 | **Layers 1–2 (Layer 3 Unfrozen)** |
| **Max Frames** | 10 | 10 | 10 | 10 |
| **Optimal Threshold** | 0.76 | 0.86 | 0.49 | 0.57 |
| **Val Accuracy** | **82.00%** | 76.00% | 77.00% | 74.00% |
| **Val Macro-F1** | 0.5739 | 0.5935 | **0.6017** | 0.5596 (Epoch 2: **0.6169**) |
| **Val ROC-AUC** | **0.6488** | 0.5655 | 0.5565 | 0.6094 (Epoch 2: **0.6741**) |
| **Val REAL Recall** | 18.75% | **37.50%** | **37.50%** | 31.25% |
| **Val FAKE Recall** | **94.05%** | 83.33% | 84.52% | 82.14% |
| **Test Accuracy** | **84.00%** | 76.00% | 75.00% | 73.00% |
| **Test Macro-F1** | 0.5116 | 0.4695 | **0.5228** | 0.4846 |
| **Test ROC-AUC** | 0.4702 | 0.4494 | 0.4576 | **0.5007** |
| **Test REAL Recall** | 6.25% | 6.25% | **18.75%** | 12.50% |

#### Key Findings — VideoModel:
1. **Max Temporal Pooling Improves Validation REAL Recall**: Max pooling (V2 & V3) doubled Validation REAL Recall from 18.75% to 37.50% and raised Val Macro-F1 to 0.6017. However, test ROC-AUC remained near random chance (~0.45).
2. **Unfreezing Layer 3 Provides the Best Feature Representation**: Unfreezing ResNet18 Layer 3 (V4) allowed deeper spatial feature adaptation, reaching an epoch peak Val ROC-AUC of **0.6741** and Val Macro-F1 of **0.6169**.
3. **Persisting Bottleneck**: Despite architectural gains, video models suffer from limited temporal depth (10 frames) and backbone capacity when evaluated on out-of-distribution test samples.
4. **Classification**:
   - **V1, V2, V3, V4**: **REJECTED** (Do not meet production promotion criteria)

---

## Detailed Confusion Matrices & Calibration

### Image Model Confusion Matrices (Validation / Test)

- **I1 (Standard Loss)**:
  - Val CM: `[[119, 91], [80, 710]]` (REAL Recall: 56.7%, FAKE Recall: 89.9%)
  - Test CM: `[[87, 113], [74, 731]]` (REAL Recall: 43.5%, FAKE Recall: 90.8%)

- **I2 (Weighted Loss)**:
  - Val CM: `[[109, 101], [78, 712]]` (REAL Recall: 51.9%, FAKE Recall: 90.1%)
  - Test CM: `[[73, 127], [59, 746]]` (REAL Recall: 36.5%, FAKE Recall: 92.7%)

- **I3 (WeightedRandomSampler)**:
  - Val CM: `[[110, 100], [81, 709]]` (REAL Recall: 52.4%, FAKE Recall: 89.8%)
  - Test CM: `[[50, 150], [27, 778]]` (REAL Recall: 25.0%, FAKE Recall: 96.6%)

### Video Model Confusion Matrices (Validation / Test)

- **V1 (Mean Pooling Baseline)**:
  - Val CM: `[[3, 13], [5, 79]]`
  - Test CM: `[[1, 15], [1, 83]]`

- **V2 (Max Pooling Standard)**:
  - Val CM: `[[6, 10], [14, 70]]`
  - Test CM: `[[1, 15], [9, 75]]`

- **V3 (Max Pooling Weighted Loss)**:
  - Val CM: `[[6, 10], [13, 71]]`
  - Test CM: `[[3, 13], [12, 72]]`

- **V4 (Mean Pooling + Layer 3 Unfrozen)**:
  - Val CM: `[[5, 11], [15, 69]]`
  - Test CM: `[[2, 14], [13, 71]]`

---

## Production Readiness Summary

| Candidate | Status | Reason / Verdict |
| :--- | :--- | :--- |
| **ImageModel (I1 Standard)** | **PROMISING** | Achieves Val Macro-F1 = 0.7372, Val ROC-AUC = 0.8281, and Test Accuracy = 81.39%. Ready for full dataset training under optimal threshold 0.35. |
| **ImageModel (I2 Weighted Loss)** | **REJECTED** | Lower Val Macro-F1 (0.7187) and reduced REAL Recall (51.90% vs 56.67%). |
| **ImageModel (I3 Sampler)** | **REJECTED** | Severe degradation in Test REAL Recall (25.00%). |
| **VideoModel (V1 Mean Base)** | **REJECTED** | Low REAL Recall (18.75%) and poor Test ROC-AUC (0.4702). |
| **VideoModel (V2 Max Base)** | **REJECTED** | Improved Val REAL Recall (37.50%) but Test ROC-AUC degraded (0.4494). |
| **VideoModel (V3 Max Weighted)** | **REJECTED** | Improved Val Macro-F1 (0.6017) but Test ROC-AUC degraded (0.4576). |
| **VideoModel (V4 Unfreeze L3)** | **REJECTED** | Best temporal/backbone candidate (Val ROC-AUC = 0.6741 on epoch 2), but test generalization remains insufficient. |

---

## Final Recommendation & Next Steps

1. **Production Status**:
   - **DO NOT** promote any Phase 5 model candidate to production checkpoints at this time.
   - Retain current production checkpoints (`image_model.pth`, `video_model.pth`) and production configurations in `model_registry.json`.

2. **Full Training Readiness**:
   - **ImageModel**: **READY FOR FULL TRAINING** using candidate **I1** configuration (Pretrained ResNet18 backbone, standard CrossEntropy loss, learning rate split `1e-5`/`1e-3`, optimal threshold `0.35`).
   - **VideoModel**: **NOT READY FOR FULL TRAINING**. VideoModel requires further architectural enhancements (e.g. 3D Convolutional backbones such as R(2+1)D or TimeSformer, increased temporal frame count, or spatial-temporal attention mechanisms) before committing to full-dataset training.
