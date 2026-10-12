# SatyaDrishti: Experimental SBI Model Benchmark Report

## 1. Provenance, Architecture, License & Preprocessing Verification

### 1.1 Official Provenance & Checkpoint
* **Paper**: *Detecting Deepfakes with Self-Blended Images* (CVPR 2022 Oral)
* **Authors**: Kaede Shiohara and Toshihiko Yamasaki (The University of Tokyo)
* **Official Repository**: [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages)
* **Official Checkpoint**: `FFc23.tar` (134.75 MB, Google Drive ID `1X0-NYT8KPursLZZdxduRQju6E52hauV0`).
* **Checkpoint Structure**: PyTorch checkpoint containing `'model'`, `'optimizer'`, and `'epoch': 99`. The model dictionary contains 706 tensors corresponding to EfficientNet-B4 (`net._conv_stem...` through `net._fc.weight` of shape `[2, 1792]`).
* **License**: Strict **Academic or Non-Profit Organization Non-Commercial Research Use Only** ([Official License Agreement](https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE)). Commercial use requires explicit authorization from Yamasaki Lab.
* **Integrity & Storage**: Placed in isolated experimental storage at `backend/models/experimental/weights/sbi_ffc23.tar` without touching any production assets.

### 1.2 Model Architecture & Preprocessing Requirements
* **Architecture**: EfficientNet-B4 (`advprop=True`, `num_classes=2`).
  - Class 0 = REAL / PRISTINE
  - Class 1 = FAKE / MANIPULATED
* **Input Resolution**: $380 \times 380$ pixels (native EfficientNet-B4 resolution).
* **Color Space & Normalization**: RGB float tensors in range $[0, 1]$ (divided by $255.0$).
* **Critical Boundary Margin Requirement**: In the official SBI pipeline (`src/inference/preprocess.py`), face bounding boxes are expanded by **25%** ($w/4$ and $h/4$) on all sides. Because SBI operates by identifying blending boundary artifacts where the synthetic mask meets the background/hair/neck, tight bounding box crops cut off the very tampering boundaries the network is trained to detect.

---

## 2. Five-Image Canonical Smoke Test

> **IMPORTANT DISCLAIMER**: The five-image test is strictly a qualitative sanity check / smoke test on canonical reference files. It does **not** demonstrate broad statistical generalization.

### 2.1 Per-Sample Predictions & Latency Comparison

| Image File | Ground Truth | Production Baseline (ResNet-18) | Experimental Xception (Face Crop) | SBI (Official 25% Margin) | SBI (Tight Haar Crop) | SBI (Full Image Input) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `profile.jpeg` | **REAL** | Raw: 0.0698 \| Cal: 0.2019<br>**LIKELY_REAL** (0.66s) | Raw: 0.8463 \| Cal: 0.7120<br>**FAKE (FP)** (1.15s) | Raw: 0.0733 \| Cal: 0.2065<br>**LIKELY_REAL (TN)** (0.32s) | Raw: 0.6793 \| Cal: 0.5982<br>**UNCERTAIN** (0.13s) | Raw: 0.1096 \| Cal: 0.2476<br>**LIKELY_REAL** (0.35s) |
| `she.jpeg` | **REAL** | Raw: 0.5000 \| Cal: 0.5000<br>**UNCERTAIN** (0.05s) | Raw: 0.5000 \| Cal: 0.5000<br>**UNCERTAIN** (0.04s) | Raw: 0.5000 \| Cal: 0.5000<br>**UNCERTAIN** (0.04s) | Raw: 0.5000 \| Cal: 0.5000<br>**UNCERTAIN** (0.05s) | Raw: 0.5000 \| Cal: 0.5000<br>**UNCERTAIN** (0.04s) |
| `picture.jpeg` | **REAL** | Raw: 0.0466 \| Cal: 0.1677<br>**UNCERTAIN** (0.16s) | Raw: 0.5256 \| Cal: 0.5136<br>**UNCERTAIN** (0.21s) | Raw: 0.2951 \| Cal: 0.3865<br>**UNCERTAIN** (0.24s) | Raw: 0.7882 \| Cal: 0.6676<br>**FAKE (FP)** (0.19s) | Raw: 0.3588 \| Cal: 0.4236<br>**UNCERTAIN** (0.18s) |
| `deepy.jpeg` | **FAKE** | Raw: 0.1025 \| Cal: 0.2403<br>**UNCERTAIN** (0.10s) | Raw: 0.8564 \| Cal: 0.7206<br>**FAKE (TP)** (0.08s) | Raw: 0.0381 \| Cal: 0.1528<br>**UNCERTAIN** (0.10s) | Raw: 0.1960 \| Cal: 0.3211<br>**UNCERTAIN** (0.10s) | Raw: 0.0432 \| Cal: 0.1620<br>**UNCERTAIN** (0.12s) |
| `professional.jpeg`| **FAKE** | Raw: 0.0846 \| Cal: 0.2203<br>**UNCERTAIN** (0.21s) | Raw: 0.4511 \| Cal: 0.4740<br>**UNCERTAIN** (0.19s) | Raw: 0.8080 \| Cal: 0.6819<br>**FAKE (TP)** (0.20s) | Raw: 0.3375 \| Cal: 0.4115<br>**UNCERTAIN** (0.20s) | Raw: 0.7555 \| Cal: 0.6453<br>**UNCERTAIN** (0.19s) |

### 2.2 Forced Binary Smoke Test Metrics (Cutoff = 0.50)

| Metric | SBI (Official Margin) | DeepfakeBench Xception | Production ResNet-18 |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 1 (`professional.jpeg`) | 1 (`deepy.jpeg`) | 0 |
| **False Negatives (FN)** | 1 (`deepy.jpeg`) | 1 (`professional.jpeg`) | 2 |
| **True Negatives (TN)** | 2 (`profile.jpeg`, `picture.jpeg`) | 0 | 2 (`profile.jpeg`, `picture.jpeg`) |
| **False Positives (FP)** | 1 (`she.jpeg` quality-gate fallback) | 3 (`profile.jpeg`, `she.jpeg`, `picture.jpeg`) | 1 (`she.jpeg` fallback) |
| **Accuracy** | **60.0%** (3/5) | 20.0% (1/5) | 40.0% (2/5) |
| **Precision** | **50.0%** (1/2) | 25.0% (1/4) | 0.0% (0/1) |
| **Recall (TPR)** | 50.0% (1/2) | 50.0% (1/2) | 0.0% (0/2) |
| **F1 Score** | **0.5000** | 0.3333 | 0.0000 |
| **False Positive Rate (FPR)** | **33.3%** | 100.0% | 33.3% |
| **False Negative Rate (FNR)** | 50.0% | 50.0% | 100.0% |
| **Abstentions (3-State Mode)** | 3 / 5 | 3 / 5 | 4 / 5 |

#### Observations on Smoke Test:
1. **Margin Effect on Real Images**: Official 25% boundary expansion dropped `profile.jpeg` from `0.6793` (false positive) down to `0.0733` (correctly real).
2. **Margin Effect on Fake Images**: On `professional.jpeg`, tight cropping yielded `0.3375` (missed), while official 25% margin yielded `0.8080` (correctly detected as deepfake).
3. **Safety Protection**: In SatyaDrishti's 3-state system, ambiguous or low-quality cases (`she.jpeg`, `picture.jpeg`, `deepy.jpeg`) were safely routed to `UNCERTAIN` rather than emitting high-confidence erroneous verdicts.

---

## 3. Exploratory Frame-Level Benchmark on 40 Video Benchmark Samples

> **IMPORTANT DISCLAIMER**: This is an exploratory single-frame face-crop evaluation on midpoint frames from the 40 balanced benchmark video clips. It does **not** evaluate temporal consistency or video-level dynamics.

### 3.1 Dataset Sample Distribution (Leak-Free Test Splits)
* **20 Real Frames**: 5 Celeb-real, 5 YouTube-real, 10 FF++ original test split.
* **20 Fake Frames**: 10 Celeb-synthesis, 3 Deepfakes, 3 Face2Face, 2 FaceSwap, 2 NeuralTextures test split.

### 3.2 Frame-Level Confusion Matrix & Metrics (Forced Binary, 0.50 cutoff)

```
                 Predicted REAL    Predicted FAKE    Total
Actual REAL           17 (TN)            3 (FP)        20
Actual FAKE            3 (FN)           17 (TP)        20
Total                 20                20             40
```

* **Accuracy**: **85.0%** (34 / 40)
* **Precision**: **85.0%** (17 / 20)
* **Recall (TPR)**: **85.0%** (17 / 20)
* **Specificity (TNR)**: **85.0%** (17 / 20)
* **False Positive Rate (FPR)**: **15.0%** (3 / 20)
* **False Negative Rate (FNR)**: **15.0%** (3 / 20)
* **F1 Score**: **0.8500**

---

### 3.3 Comparative Domain Shift Breakdown: SBI vs. Xception

| Evaluation Dimension | Metric | DeepfakeBench Xception (Video Pipeline) | SBI EfficientNet-B4 (Frame Crops) |
| :--- | :--- | :---: | :---: |
| **In-Domain: FF++ C23 (20 clips)** | Accuracy | **85.0%** | **80.0%** |
| | False Positive Rate (FPR) | 30.0% (3/10) | **10.0% (1/10)** |
| | False Negative Rate (FNR) | **0.0% (0/10)** | 30.0% (3/10) |
| | F1 Score | **0.8696** | 0.7778 |
| **Cross-Domain: Celeb-DF v2 (20 clips)** | Accuracy | 55.0% | **90.0%** |
| | False Positive Rate (FPR) | **90.0% (9/10)** *(Severe domain collapse)* | **20.0% (2/10)** *(Resolved)* |
| | False Negative Rate (FNR) | **0.0% (0/10)** | **0.0% (0/10)** |
| | Recall (TPR) | **100.0%** | **100.0%** |
| | Precision | 52.6% | **83.3%** |
| | F1 Score | 0.6897 | **0.9091** |
| **Overall 40 Samples** | Accuracy | 70.0% | **85.0%** |
| | Precision | 62.5% | **85.0%** |
| | Recall | **100.0%** | 85.0% |
| | False Positive Rate | 60.0% (12/20) | **15.0% (3/20)** |
| | F1 Score | 0.7692 | **0.8500** |

### 3.4 Key Forensic Insights
1. **Resolution of Cross-Domain False Alarm Inflation**:
   Xception suffered a catastrophic 90% False Positive Rate on real Celeb-DF videos (scoring 9 of 10 pristine clips as fakes). SBI reduced this cross-domain False Positive Rate to **20%**, achieving **90% cross-dataset accuracy on Celeb-DF**.
2. **Complementary Strengths**:
   - **Xception**: Highly sensitive to subtle in-domain manipulations (0% FNR across FF++ Face2Face and NeuralTextures), but overfits to FF++ compression artifacts, causing severe false positives out-of-domain.
   - **SBI**: Exceptionally robust against false alarms on uncompressed/varied real faces (15% overall FPR), but exhibits false negatives on subtle facial reenactment/expression warping (e.g. FF++ Face2Face and NeuralTextures) where boundary blending seams are less prominent.
3. **Inference Latency**:
   SBI executes at **0.19 seconds per frame** on the NVIDIA GeForce GTX 1650 (~5.2 FPS), making it highly viable for real-time inference without VRAM strain.

---

## 4. Next Steps Requiring Approval

1. Do **not** modify production weights or promote SBI to production until explicit user authorization.
2. Consider an isolated ensemble experiment combining SBI's robust boundary features with Xception's reenactment sensitivity under an experimental multi-backbone fusion agent.
