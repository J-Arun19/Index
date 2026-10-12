# SatyaDrishti: SBI Pretrained Model Comprehensive Validation Report

## 1. Executive Summary & Governance Integrity

In accordance with system directives:
1. **Zero Training or Fine-Tuning**: The pretrained Self-Blended Images (SBI) EfficientNet-B4 checkpoint (`FFc23.tar`, epoch 99) was evaluated entirely zero-shot.
2. **Zero Production Modification**: All production code, production weights (`image_model.pth`, `video_model.pth`), temperature scaling calibrations, API endpoints, and decision thresholds remain 100% unmodified.
3. **Strict Experimental Isolation**: All evaluators, detector wrappers, manifests, and logs reside strictly in `backend/models/experimental/`, `backend/agents/experimental/`, and `scratch/`.
4. **Leak-Free Guarantee**: Evaluated exclusively on verified held-out test splits:
   - Celeb-DF v2 official test list (`List_of_testing_videos.txt`: 178 real, 340 fake clips) — completely unseen cross-dataset.
   - FaceForensics++ C23 official test split (`backend/training/splits/test.json`: 70 sequence pairs) — disjoint from FF++ training sequences.
   - In-the-wild real and fake test assets.
5. **Licensing Restriction Documented**: The pretrained checkpoint is subject to Yamasaki Lab's **Academic / Non-Commercial Research License** ([mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages)). Commercial use requires explicit authorization.

---

## 2. Independent Still-Image Benchmark (55 Curated Samples)

### 2.1 Dataset Composition & Leakage Prevention
The still-image evaluation set was assembled in `scratch/image_eval_set/` using `scratch/image_eval_manifest.json` across five distinct forensic categories:
* **Genuine Real Portraits (24)**:
  - 5 Celebrity interview portraits (Celeb-DF `Celeb-real` test set)
  - 5 In-the-wild YouTube interview portraits (Celeb-DF `YouTube-real` test set)
  - 10 Television broadcast portraits (FF++ C23 `original` test split)
  - 3 Personal in-the-wild photos (`profile.jpeg`, `picture.jpeg`, `HON1.jpg`)
  - 1 Degraded/low-light portrait (`she.jpeg`)
* **Face Swaps (19)**:
  - 8 High-quality deepfake face swaps (Celeb-DF `Celeb-synthesis` test set)
  - 5 Autoencoder deepfake face swaps (FF++ C23 `Deepfakes` test split)
  - 5 Graphics/blending face swaps (FF++ C23 `FaceSwap` test split)
  - 1 In-the-wild face swap (`deepy.jpeg`)
* **Facial Reenactments (9)**:
  - 5 Expression transfer reenactments (FF++ C23 `Face2Face` test split)
  - 4 Neural texture rendering reenactments (FF++ C23 `NeuralTextures` test split)
* **AI-Generated Synthetic Faces (1)**:
  - 1 Synthetic portrait (`professional.jpeg`)
* **Safety Edge Cases (2)**:
  - 1 Legal contract document (`document_contract_sample.png`)
  - 1 Application graphic logo (`deepfake_logo.png`)

---

### 2.2 Still-Image Performance Metrics: Forced Binary Mode (Cutoff = 0.50)

Evaluated across the 53 facial images (excluding non-face documents):

| Metric | Pretrained SBI (EfficientNet-B4) | Production Baseline (ResNet-18) | Experimental Xception |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | **21 / 29** | 0 / 29 | **28 / 29** |
| **False Negatives (FN)** | 8 / 29 | 29 / 29 | **1 / 29** |
| **True Negatives (TN)** | **15 / 24** | 19 / 24 | 4 / 24 |
| **False Positives (FP)** | **9 / 24** | 5 / 24 | **20 / 24 (Severe False Alarms)** |
| **Accuracy** | **67.9%** (36 / 53) | 35.8% (19 / 53) | 60.4% (32 / 53) |
| **Precision** | **70.0%** (21 / 30) | 0.0% (0 / 5) | 58.3% (28 / 48) |
| **Recall (TPR)** | **72.4%** (21 / 29) | 0.0% (0 / 29) | **96.6%** (28 / 29) |
| **Specificity (TNR)** | **62.5%** (15 / 24) | 79.2% (19 / 24) | 16.7% (4 / 24) |
| **False Positive Rate (FPR)** | **37.5%** | 20.8% | **83.3%** |
| **False Negative Rate (FNR)** | **27.6%** | 100.0% | **3.4%** |
| **F1 Score** | **0.7119** | 0.0000 | 0.7273 |
| **Mean Latency per Image** | **0.52s** | 0.94s | 0.32s |

---

### 2.3 Still-Image Safety-Gated 3-State Performance (`SafetyDecisionSystem`)

In accordance with SatyaDrishti safety policies, verdicts were passed through the calibrated three-state engine without forced binary thresholds:

| Model | Ground Truth REAL (24) | Ground Truth FAKE (29) | Overall Abstention | Non-Face Rejection (`NOT_SUITABLE`) |
| :--- | :---: | :---: | :---: | :---: |
| **SBI EfficientNet-B4** | 7 `LIKELY_REAL`<br>16 `UNCERTAIN`<br>**1 `FAKE` (FP = 4.2%)** | 17 `FAKE`<br>10 `UNCERTAIN`<br>**2 `LIKELY_REAL` (FN = 6.9%)** | **49.1%** | **2 / 2 (100.0%)** |
| **Production ResNet-18** | 7 `LIKELY_REAL`<br>17 `UNCERTAIN`<br>0 `FAKE` (FP = 0.0%) | 0 `FAKE`<br>20 `UNCERTAIN`<br>**9 `LIKELY_REAL` (FN = 31.0%)** | 69.8% | **2 / 2 (100.0%)** |
| **Experimental Xception** | 1 `LIKELY_REAL`<br>18 `UNCERTAIN`<br>**5 `FAKE` (FP = 20.8%)** | 21 `FAKE`<br>8 `UNCERTAIN`<br>0 `LIKELY_REAL` (FN = 0.0%) | 49.1% | **2 / 2 (100.0%)** |

---

### 2.4 Still-Image Manipulation Category Breakdown (SBI)

| Forensic Manipulation Category | Ground Truth | Sample Count | SBI Forced Accuracy | Mean Raw Score | 3-State Verdict Breakdown |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Celeb-DF Deepfake Face Swap** | FAKE | 8 | **100.0%** (8/8) | **0.9912** | 8 `FAKE`, 0 `UNCERTAIN`, 0 `LIKELY_REAL` |
| **FF++ FaceSwap** | FAKE | 5 | **100.0%** (5/5) | **0.8355** | 3 `FAKE`, 2 `UNCERTAIN`, 0 `LIKELY_REAL` |
| **FF++ Deepfakes** | FAKE | 5 | **80.0%** (4/5) | **0.8721** | 4 `FAKE`, 1 `UNCERTAIN`, 0 `LIKELY_REAL` |
| **AI Synthetic Portrait (`professional.jpeg`)** | FAKE | 1 | **100.0%** (1/1) | **0.8080** | 1 `FAKE`, 0 `UNCERTAIN`, 0 `LIKELY_REAL` |
| **FF++ Face2Face (Reenactment)** | FAKE | 5 | **40.0%** (2/5) | **0.4625** | 1 `FAKE`, 3 `UNCERTAIN`, 1 `LIKELY_REAL` |
| **FF++ NeuralTextures (Reenactment)** | FAKE | 4 | **25.0%** (1/4) | **0.3247** | 0 `FAKE`, 3 `UNCERTAIN`, 1 `LIKELY_REAL` |
| **In-the-wild Real Portraits** | REAL | 3 | **100.0%** (3/3) | **0.1472** | 2 `LIKELY_REAL`, 1 `UNCERTAIN`, 0 `FAKE` |
| **FF++ Broadcast Real Portraits** | REAL | 10 | **70.0%** (7/10) | **0.2317** | 3 `LIKELY_REAL`, 7 `UNCERTAIN`, 0 `FAKE` |
| **Celeb-DF Celebrity Real Portraits** | REAL | 5 | **60.0%** (3/5) | **0.3122** | 0 `LIKELY_REAL`, 5 `UNCERTAIN`, 0 `FAKE` |
| **Celeb-DF YouTube Real Portraits** | REAL | 5 | **40.0%** (2/5) | **0.4042** | 2 `LIKELY_REAL`, 2 `UNCERTAIN`, 1 `FAKE` |
| **Non-Face Media (Contract / Logo)** | NON-FACE | 2 | **100.0%** (2/2) | N/A | 2 `NOT_SUITABLE` (Safely rejected) |

---

## 3. Full Video-Level Benchmark (40 Balanced Clips)

### 3.1 Evaluation Protocol
* **Dataset**: 40 balanced clips (20 REAL, 20 FAKE) sampled uniformly across 8 frames with face detection, boundary expansion, spatial backbone inference, temporal variance estimation, and 3-state fusion.
* **Evaluators**:
  1. `ExperimentalSBIVideoAgent` (SBI EfficientNet-B4 + 25% boundary face tracking + temporal aggregation)
  2. `ExperimentalVideoAgent` (DeepfakeBench Xception + temporal aggregation)
  3. `VideoAgent` (Production ResNet-18 + LSTM baseline)

---

### 3.2 Video-Level Performance Comparison: Forced Binary Mode (Cutoff = 0.50)

| Metric | SBI EfficientNet-B4 Video | DeepfakeBench Xception Video | Production ResNet-18 + LSTM Video |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 14 / 20 | **20 / 20** | 10 / 20 |
| **False Negatives (FN)** | 6 / 20 | **0 / 20** | 10 / 20 |
| **True Negatives (TN)** | **18 / 20** | 8 / 20 | 16 / 20 |
| **False Positives (FP)** | **2 / 20 (Resolved False Alarms)** | **12 / 20 (Severe Domain Collapse)** | 4 / 20 |
| **Accuracy** | **80.0%** (32 / 40) | 70.0% (28 / 40) | 65.0% (26 / 40) |
| **Precision** | **87.5%** (14 / 16) | 62.5% (20 / 32) | 71.4% (10 / 14) |
| **Recall (TPR)** | 70.0% (14 / 20) | **100.0%** (20 / 20) | 50.0% (10 / 20) |
| **Specificity (TNR)** | **90.0%** (18 / 20) | 40.0% (8 / 20) | 80.0% (16 / 20) |
| **False Positive Rate (FPR)** | **10.0%** | **60.0%** | 20.0% |
| **False Negative Rate (FNR)** | 30.0% | **0.0%** | 50.0% |
| **F1 Score** | **0.7778** | 0.7692 | 0.5882 |
| **Mean Video Latency** | 3.44s | **2.46s** | 5.21s |

---

### 3.3 Video-Level Safety-Gated 3-State Performance (`SafetyDecisionSystem`)

| Model | Ground Truth REAL (20) | Ground Truth FAKE (20) | Overall Abstention Rate |
| :--- | :---: | :---: | :---: |
| **SBI EfficientNet-B4 Video** | **11 `LIKELY_REAL`**<br>9 `UNCERTAIN`<br>**0 `FAKE` (FP = 0.0%)** | 11 `FAKE`<br>6 `UNCERTAIN`<br>**3 `LIKELY_REAL` (FN = 15.0%)** | **37.5% (15 / 40)** |
| **DeepfakeBench Xception Video** | 2 `LIKELY_REAL`<br>17 `UNCERTAIN` (85%)<br>**1 `FAKE` (FP = 5.0%)** | **12 `FAKE`**<br>8 `UNCERTAIN`<br>**0 `LIKELY_REAL` (FN = 0.0%)** | 62.5% (25 / 40) |
| **Production ResNet-18 Video** | 0 `LIKELY_REAL`<br>20 `UNCERTAIN`<br>0 `FAKE` (FP = 0.0%) | 0 `FAKE`<br>20 `UNCERTAIN`<br>0 `LIKELY_REAL` (FN = 0.0%) | **100.0% (40 / 40)** |

---

### 3.4 Cross-Dataset Domain Shift Comparison (Video Level)

| Dataset Domain | Metric | SBI EfficientNet-B4 Video | DeepfakeBench Xception Video |
| :--- | :--- | :---: | :---: |
| **Cross-Domain: Celeb-DF v2 (20 clips)** | **Accuracy** | **90.0%** | 55.0% |
| | **False Positive Rate (FPR)** | **20.0% (2/10)** | **90.0% (9/10)** |
| | **False Negative Rate (FNR)** | **0.0% (0/10)** | **0.0% (0/10)** |
| | **Recall (TPR)** | **100.0%** | **100.0%** |
| | **Precision** | **83.3%** | 52.6% |
| | **F1 Score** | **0.9091** | 0.6897 |
| **In-Domain: FF++ C23 (20 clips)** | **Accuracy** | 70.0% | **85.0%** |
| | **False Positive Rate (FPR)** | **0.0% (0/10)** | 30.0% (3/10) |
| | **False Negative Rate (FNR)** | 60.0% (6/10) | **0.0% (0/10)** |
| | **Recall (TPR)** | 40.0% | **100.0%** |
| | **Precision** | **100.0%** | 76.9% |
| | **F1 Score** | 0.5714 | **0.8696** |

---

## 4. Synthesis of Forensic Findings

1. **Resolution of Cross-Dataset False Alarm Inflation**:
   * Xception suffered severe domain collapse on Celeb-DF real videos (FPR = **90.0%**), misidentifying genuine YouTube and celebrity footage as manipulated.
   * SBI resolves this failure mode, achieving **90.0% cross-domain accuracy on Celeb-DF** (FPR = **20.0%**), with **0.0% False Positive Rate on real FF++ videos** under 3-state gating.
2. **Specialized Sensitivity Profile**:
   * **Where SBI Excels**: Facial boundary tampering and face replacement (Celeb-DF synthesis: $100\%$ accuracy, FF++ FaceSwap: $100\%$, FF++ Deepfakes: $80\%$, AI synthetic faces: $100\%$).
   * **Where SBI Struggles**: Subtle reenactments and expression warping (FF++ Face2Face: $40\%$, NeuralTextures: $25\%$). Because the original facial boundary is preserved during reenactment, SBI's boundary-blending prior is largely bypassed.
   * **Where Xception Excels**: Detecting expression reenactments ($100\%$ sensitivity across Face2Face and NeuralTextures), but at the cost of aggressive false alarms on out-of-domain pristine faces.
3. **Operational Viability (NVIDIA GeForce GTX 1650)**:
   * Still image latency: **0.52s** per image.
   * Video latency (8 frames): **3.44s** per clip.
   * VRAM footprint: $\sim650\text{ MB}$, well within the 4 GB GPU budget.

---

## 5. Controlled Production Trial Recommendation

### Recommendation: **CONDITIONAL APPROVAL FOR CONTROLLED SHADOW/CANARY TRIAL ONLY**

#### Rationale:
1. **Strengths**: SBI demonstrates superior cross-dataset generalization on face swaps and AI synthesis ($90\%$ on Celeb-DF), reducing Xception's $60\%$ video false alarm rate down to $10\%$. Under SatyaDrishti's 3-state safety system, it produced **zero false alarms on 20 real videos**.
2. **Identified Blind Spot**: SBI has a notable false-negative rate on subtle expression reenactments (Face2Face, NeuralTextures). It should **not** operate as a sole single-point detector without reenactment/frequency evidence.
3. **Licensing Gate**: Because official SBI weights are restricted to **Academic / Non-Commercial Research**, they **cannot** be deployed into commercial production without licensing clearance from Yamasaki Lab (The University of Tokyo). For academic, research, or evaluation deployments, it is technically sound.

#### Proposed Architecture for Controlled Trial:
Deploy SBI alongside Xception in a **two-expert gated fusion pipeline**:
- SBI serves as the primary **Boundary & Swapping Specialist** (high specificity, low false alarm rate).
- Xception / Frequency analysis serves as the **Reenactment & Expression Specialist** (high sensitivity to expression warping).
- Fusion logic: If SBI detects a boundary seam $\implies$ high-confidence `FAKE`. If SBI indicates authentic boundaries but Xception flags expression anomaly $\implies$ route safely to `UNCERTAIN` for human review.

No production code, weights, or thresholds have been altered. Awaiting your approval before taking any further steps.
