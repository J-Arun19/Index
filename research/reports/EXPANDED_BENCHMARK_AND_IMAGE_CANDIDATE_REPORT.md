# SatyaDrishti: Forensic Benchmark & Pretrained Image Model Evaluation

## 1. Executive Summary & Decisions Upheld

In accordance with system directives:
1. **Hugging Face ViT (`dima806/deepfake_vs_real_image_detection`) is strictly rejected** from production due to catastrophic framing sensitivity (full-image predicted ~0.001 fake prob; face-crop predicted ~0.992 fake prob on the exact same pristine files).
2. **DeepfakeBench Xception remains strictly experimental for video analysis** and is evaluated herein on a balanced multi-generator test set.
3. **No models were trained or fine-tuned.**
4. **No production weights, temperature scaling parameters, API schemas, or fusion thresholds were altered.**
5. **Disk space and hardware constraints were strictly validated** prior to evaluation:
   - System drive `C:` has **5.59 GB free** (constraining large foundation model downloads).
   - Dataset drive `D:` has **90.58 GB free**.
   - Inference GPU: **NVIDIA GeForce GTX 1650 (4 GB VRAM)** running PyTorch `2.11.0+cu128` on Windows.

---

## 2. Investigation of Pretrained Image Deepfake Detectors

### 2.1 Evaluated Forensic Methodologies

| Candidate | Architecture | Forensic Paradigm | Official License & Source | Pretrained Checkpoint Availability | Hardware / Windows Viability (GTX 1650) | Cross-Dataset Generalization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SBI (Self-Blended Images)** *(Shiohara & Yamasaki, CVPR 2022)* | EfficientNet-B4 / ResNet-34 | **Self-Blended Boundary Forensics**: Synthesizes pseudo-forgeries from single pristine faces to detect boundary seams, color inconsistencies, and blending masks. | **Academic / Non-Commercial Only** (`mapooon/SelfBlendedImages` / Yamasaki Lab, Univ of Tokyo). DeepfakeBench implementation is **CC BY-NC 4.0**. | Readily available via official Google Drive & DeepfakeBench release (`sbi_effnb4.pth`, ~70 MB). | **Fully Compatible**: ~70 MB footprint, ~600 MB VRAM, pure PyTorch. | **Superior**: Highest cross-dataset transfer across FF++, Celeb-DF, DFDC, and DFD in academic literature. |
| **F3-Net (Thinking in Frequency)** *(Qian et al., ECCV 2020)* | Dual-stream Xception + DCT Frequency subnetworks | **Frequency-Aware Decomposition (FAD) & Local Frequency Statistics (LFS)**: Uses DCT to detect frequency discrepancies across low/mid/high bands. | **CC BY-NC 4.0** (standardized in `SCLBD/DeepfakeBench`; third-party implementations lack commercial rights). | Available via DeepfakeBench release (`f3net_best.pth`, ~160 MB). | **Fully Compatible**: ~160 MB checkpoint, ~1.2 GB VRAM, torch-based DCT. | **Moderate / Compression-Sensitive**: Generalizes well within matched compression (e.g. C23), but DCT features degrade under varied compression codecs. |
| **SPSL (Spatial-Phase Shallow Learning)** *(Liu et al., CVPR 2021)* | Shallow CNN (6 conv blocks) | **Phase Spectrum Analysis**: Applies 2D Fast Fourier Transform (FFT) to extract phase spectrum (retaining edge discontinuities and upsampling artifacts). | **CC BY-NC 4.0** (standardized in `SCLBD/DeepfakeBench`). | Available via DeepfakeBench release (`spsl_best.pth`, ~45 MB). | **Fully Compatible**: ~45 MB checkpoint, <800 MB VRAM, native `torch.fft`. | **Moderate**: Retains boundary phase disruptions, but struggles with in-the-wild facial occlusions and low-light noise. |
| **UniversalFakeDetect (UnivFD)** *(Ojha et al., CVPR 2023)* | Frozen CLIP ViT-L/14 + Linear Probe | **Foundation Feature Invariance**: Leverages general semantic features of large-scale pretrained CLIP to identify universal generative artifacts. | **MIT License** (`WisconsinAIVision/UniversalFakeDetect` - Fully permissive for commercial use). | Official linear head is tiny (`fc_weights.pth`, ~3 KB). CLIP backbone is ~1.7 GB. | **Marginal**: Requires 1.7 GB backbone download and ~2.8 GB VRAM. Tight on C: drive cache and 4 GB GPU limit. | **High on Whole-Image Synthesis**, but weaker on subtle facial boundary blending (face-swaps / reenactment). |
| **CNNDetection** *(Wang et al., CVPR 2020)* | ResNet-50 (`blur_jpg_prob0.5.pth`) | **CNN Artifact Fingerprinting**: Trained on ProGAN generations with blur and JPEG augmentations to spot general CNN synthesis. | **CC BY-NC-SA 4.0** (`PeterWang512/CNNDetection`). | Checkpoint already located in local workspace `scratch/models/blur_jpg_prob0.5.pth` (282 MB). | **Fully Compatible**: ResNet-50 runs at ~1.0 GB VRAM. | **Failed on Face Manipulation**: Live testing on canonical deepfakes yielded raw probability `0.0000` because it seeks full-scene GAN signatures rather than facial swap seams. |

---

### 2.2 Strongest Image Model Candidate

#### **Selected Candidate: Self-Blended Images (SBI) with EfficientNet-B4 Backbone**
- **Forensic Rationale**: Unlike models trained on specific GAN outputs that memorize generator signatures, SBI synthesizes self-blends (source and target from the same subject) with synthetic blending masks, landmark warping, and color jitter. This forces the network to learn genuine boundary artifacts (edge blurring, color mismatches, frequency disruptions) common to virtually all facial manipulations (Deepfakes, Face2Face, FaceSwap, FaceShifter).
- **Benchmark Track Record**: In the official CVPR 2022 paper and independent DeepfakeBench evaluations, SBI achieved the highest cross-dataset AUC when tested on unseen datasets (Celeb-DF: 93.2% AUC, DFDC: 85.6% AUC), vastly outperforming Xception and standard EfficientNet-B4 baselines.
- **Resource Footprint**: The 70 MB checkpoint is well within the 5.59 GB disk limit on `C:` and runs with <1 GB VRAM on the local GTX 1650.
- **Licensing Constraint (CRITICAL)**: The official code from `mapooon/SelfBlendedImages` is governed by a **strict Non-Commercial Academic Research Agreement** (Yamasaki Lab). If commercial production rights are required without contacting the authors, the fallback candidate is **UniversalFakeDetect (MIT License)**, though UnivFD requires significant disk space for CLIP ViT-L/14 and is less specialized for facial boundary localization.

---

## 3. Expanded Video Benchmark: DeepfakeBench Xception

### 3.1 Evaluation Protocol & Dataset Composition
An expanded, balanced benchmark of **40 video clips (20 REAL, 20 FAKE)** was evaluated using `ExperimentalVideoAgent` on the local NVIDIA GeForce GTX 1650:
- **20 Ground-Truth REAL Clips**:
  - 5 clips from Celeb-DF v2 official test set (`Celeb-real`)
  - 5 clips from Celeb-DF v2 official test set (`YouTube-real`)
  - 10 clips from FaceForensics++ C23 official test split (`original`)
- **20 Ground-Truth FAKE Clips**:
  - 10 clips from Celeb-DF v2 official test set (`Celeb-synthesis`)
  - 10 clips from FaceForensics++ C23 test split:
    - 3 Deepfakes
    - 3 Face2Face
    - 2 FaceSwap
    - 2 NeuralTextures

Each video was sampled uniformly across 8 frames with Haar-cascade face extraction, spatial Xception inference, temporal variance estimation, and 3-state safety fusion.

---

### 3.2 Forced Binary Predictions (Cutoff = 0.50 on Raw Score)

#### Confusion Matrix (Forced Binary)
| | Predicted REAL | Predicted FAKE | Total |
| :--- | :---: | :---: | :---: |
| **Actual REAL** | **8 (TN)** | **12 (FP)** | 20 |
| **Actual FAKE** | **0 (FN)** | **20 (TP)** | 20 |
| **Total** | 8 | 32 | 40 |

#### Performance Metrics (Forced Binary)
- **Accuracy**: 70.0% (28 / 40)
- **Precision**: 62.5% (20 / 32)
- **Recall (TPR / Sensitivity)**: **100.0%** (20 / 20)
- **Specificity (TNR)**: 40.0% (8 / 20)
- **False Positive Rate (FPR)**: **60.0%** (12 / 20)
- **False Negative Rate (FNR)**: **0.0%** (0 / 20)
- **F1 Score**: 0.7692

---

### 3.3 Domain Shift Analysis: In-Domain (FF++) vs Cross-Domain (Celeb-DF)

Comparing the performance across the two dataset distributions reveals a critical forensic insight:

| Metric | FaceForensics++ C23 (In-Domain, N=20) | Celeb-DF v2 (Cross-Domain, N=20) |
| :--- | :---: | :---: |
| **True Positives (TP)** | 10 / 10 | 10 / 10 |
| **False Negatives (FN)** | 0 / 10 (FNR = 0.0%) | 0 / 10 (FNR = 0.0%) |
| **True Negatives (TN)** | 7 / 10 | 1 / 10 |
| **False Positives (FP)** | 3 / 10 (FPR = 30.0%) | **9 / 10 (FPR = 90.0%)** |
| **Accuracy** | **85.0%** | **55.0%** |
| **Precision** | **76.9%** | **52.6%** |
| **Recall** | **100.0%** | **100.0%** |
| **F1 Score** | **0.8696** | **0.6897** |

**Observation**: DeepfakeBench Xception was trained on FaceForensics++ C23. While it maintains 85% accuracy in-domain, transferring to Celeb-DF v2 causes severe domain shift: genuine real videos from YouTube and celebrity interviews receive elevated raw scores (0.53 - 0.78), resulting in an unacceptable **90% False Positive Rate** under forced binary classification.

---

### 3.4 Safety-Gated 3-State Evaluation (`SafetyDecisionSystem`)

The SatyaDrishti safety engine explicitly separates raw classifier outputs from final verdicts by incorporating abstention margins ($[\tau_{\text{real}}, \tau_{\text{fake}}] = [0.35, 0.65]$), temperature scaling ($T=1.885$), and temporal variance gating ($>0.22$).

#### 3-State Verdict Breakdown
| Ground Truth | Verdict: LIKELY_REAL | Verdict: UNCERTAIN (Abstained) | Verdict: FAKE | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Actual REAL (20)** | 2 (10.0%) | **17 (85.0%)** | **1 (5.0%)** | 20 |
| **Actual FAKE (20)** | 0 (0.0%) | **8 (40.0%)** | **12 (60.0%)** | 20 |
| **Total** | 2 | 25 | 13 | 40 |

#### Safety System Impact
1. **Dramatic False Positive Suppression**: 
   - Forced binary produced **12 false positives** out of 20 real videos.
   - The safety decision system reduced confident false positives down to **1** (a 91.7% reduction), routing 17 ambiguous or out-of-domain cases to `UNCERTAIN`.
2. **Zero False Negatives**: 
   - Across all 20 manipulated clips, exactly zero were erroneously classified as `LIKELY_REAL`.
3. **High Precision on Resolved Decisions**:
   - Excluding `UNCERTAIN` cases, 15 decisions were resolved:
     - **Resolved Precision**: **92.3%** (12 TP / [12 TP + 1 FP])
     - **Resolved Recall**: **100.0%** (12 TP / [12 TP + 0 FN])
     - **Resolved F1**: **0.9600**
4. **Trade-Off**:
   - The overall abstention rate is **62.5%** (85% on real clips, 40% on fake clips), reflecting the classifier's out-of-domain probability inflation on real videos.

---

### 3.5 Latency Telemetry (NVIDIA GeForce GTX 1650)
- **Mean Processing Time**: 2.46 seconds per video (8 frames sampled)
- **Median Processing Time**: 2.15 seconds per video
- **Minimum Processing Time**: 0.62 seconds per video
- **Maximum Processing Time**: 6.88 seconds per video
- **Standard Deviation**: 1.40 seconds

---

## 4. Proposed Next Steps (Awaiting User Approval)

1. **Experimental SBI Integration in Isolation**:
   - Create an isolated `ExperimentalSBIImageAgent` in `backend/agents/experimental/` using pretrained SBI EfficientNet-B4 weights (`sbi_effnb4.pth`), keeping all existing production models untouched.
   - Evaluate SBI across both canonical still images and video frame crops to test if its boundary-based representations resolve Xception's 90% cross-domain false-positive rate on Celeb-DF real media.
2. **Abstention Gate Calibration for Cross-Dataset Video**:
   - Design a cross-dataset temperature and variance calibration profile for experimental video evaluation to mitigate over-abstention without lowering safety thresholds.
