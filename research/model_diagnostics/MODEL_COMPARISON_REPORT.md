# SatyaDrishti — Model Comparison & Diagnostic Evaluation Report

**Evaluation Protocol**: Strict Inference-Only (No Training, No Fine-Tuning, No Weights Replacement, No Production Decision Logic Modifications)  
**Date**: October 13, 2026  
**Artifact Directory**: `research/model_diagnostics/`  
**Associated Files**:
- Machine-Readable Inventory: [`model_inventory.json`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/model_inventory.json)
- Inventory Documentation: [`MODEL_INVENTORY.md`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/MODEL_INVENTORY.md)
- Production Model Audit: [`PRODUCTION_MODEL_AUDIT.md`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/PRODUCTION_MODEL_AUDIT.md)
- Inference Predictions: [`inference_predictions.json`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/inference_predictions.json)
- Summary Metrics JSON: [`metrics_summary.json`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/metrics_summary.json)
- Run Manifest: [`RUN_MANIFEST.json`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/research/model_diagnostics/RUN_MANIFEST.json)

---

## 1. Executive Summary & Direct Answers to User Questions

### Question 1: Why does the current ResNet-18 fail on the known fake portraits?
**Answer**:
The failure on `deepy.jpeg` (raw fake probability: `0.1025`) and `professional.jpeg` (raw fake probability: `0.0846`) is caused by **severe domain shift and severe undertraining**, NOT by a software defect or inverted labels:
1. **Severe Undertraining**: The production checkpoint (`image_model.pth`, SHA-256 `cc411dafbc845e1fb65197487a87fe409bc0c7ff1520c8e4e83b4aceb4fa8aeb`) was trained for **only 2 epochs** on an early pilot run (`image_both_20260815_233645.pth`). At epoch 2, training accuracy was only 62.5% and validation loss was high at 1.8859. The linear head never learned positive fake logits (its fake logits are consistently negative: `-0.777` on `deepy`, `-1.663` on `professional`).
2. **Video Frame Domain Shift**: 96.2% of the training images were video frames extracted from FaceForensics++ C23 compression. In FF++ C23, deepfake artifacts manifest as H.264 video compression macroblock boundary noise, temporal tearing, and low-bitrate DCT quantization. Modern static AI-generated portraits (`professional.jpeg`) and high-resolution face swaps (`deepy.jpeg`) lack H.264 video compression seams. The model looks for video compression artifacts; finding none, its convolutional filters assign strong negative fake logits, misclassifying them as REAL.

### Question 2: Is class mapping or preprocessing actually defective?
**Answer**:
**NO. Neither is defective.**
- **Class Index Mapping**: Verified across [`image_dataset.py`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/training/image_dataset.py#L13), [`generate_manifest.py`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/training/generate_manifest.py#L263), [`train_image_model.py`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/training/train_image_model.py#L1874), and [`image_model.py`](file:///D:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/image_model.py#L75). **Index 0 is REAL, and Index 1 is FAKE**. There is no label inversion.
- **Preprocessing Pipeline**: Line-by-line comparison between training and inference demonstrates an exact match: OpenCV BGR read $\rightarrow$ Haar cascade face detection $\rightarrow$ bounding box face crop $\rightarrow$ resize to $(224, 224)$ $\rightarrow$ BGR-to-RGB conversion $\rightarrow$ scaling by $1/255.0$ $\rightarrow$ ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$) $\rightarrow$ tensor shape `(1, 3, 224, 224)`.

### Question 3: Which existing model performs best on the available static fake examples?
**Answer**:
- On catching the static fake portraits (`professional.jpeg` and `deepy.jpeg`), **SBI EfficientNet-B4** (`sbi_ffc23.tar`) and **ViT-Base** achieve 100% recall (predicting fake probabilities of $1.000$ and $0.990+$).
- However, when evaluated on **pristine genuine portraits**, both models suffer from **catastrophic false positive rates** (SBI FPR: $91.67\%$, ViT FPR: $79.17\%$). They flag genuine photographs as 95–100% fake.
- Across the broader 53-sample still image evaluation set, **`full_image_resnet18`** (`model_best.pth`) achieved the most balanced performance (Accuracy: **79.25%**, Macro-F1: **0.7876**, ROC-AUC: **0.8922**, Fake Recall: **86.21%**, Specificity: **70.83%**), catching `professional.jpeg` with a fake probability of **0.9975**.

### Question 4: Does the evidence support changing the model, or is the evaluation set too small?
**Answer**:
**The evidence does NOT support immediately replacing the production model.**
The currently available still-image evaluation set (53 face images + 5 canonical images) is too small ($N=58$) to justify a production model swap:
1. Every candidate model in the repository exhibits a severe failure mode when tested out-of-domain:
   - SBI and ViT have unacceptable $\sim 80\text{--}92\%$ False Positive Rates on pristine real photos.
   - DeepfakeBench Xception has a $70.8\%$ False Positive Rate on real photos.
   - Production ResNet-18 has a $100\%$ False Negative Rate on static fakes.
2. In production, SatyaDrishti's **multi-agent safety architecture** already rescues the weak ResNet-18 model: when ResNet-18 outputs $0.1025$ on `deepy.jpeg`, independent FFT and ELA forensic analyzers detect anomalies and escalate the verdict to **`UNCERTAIN`**. Replacing the checkpoint without proper multi-dataset calibration would flood production with False Positives on real user uploads.

### Question 5: Is the existing 0.04 threshold valid for the current checkpoint?
**Answer**:
**NO. The 0.04 threshold is INVALID and DANGEROUS for production.**
- Provenance: Threshold `0.04` was found by `optimize_threshold()` doing a grid search maximizing Macro-F1 on the 2-epoch pilot validation set. Because the pilot checkpoint had severe negative logit bias, pushing the threshold down to 0.04 artificially maximized F1 by classifying almost everything as fake.
- Consequence: When evaluated on the 200 real images in the original test set, threshold 0.04 produced a **93.0% False Positive Rate** (186 out of 200 real images were misclassified as fake, leaving a Real Recall of only 7.0%).
- Production logic correctly discards threshold 0.04 and evaluates with standard calibration and multi-agent consensus.

### Question 6: Can any candidate be recommended for a controlled prototype?
**Answer**:
**YES.**
**SBI EfficientNet-B4** (`backend/models/experimental/weights/sbi_ffc23.tar`) and **Full Image ResNet-18** (`backend/models/weights/runs/full_image_I1/model_best.pth`) are strongly recommended for an **isolated research prototype**:
- SBI EfficientNet-B4 is specialized in boundary seam forensics (using its 25% margin crop) and catches subtle face swaps.
- However, its raw output must be temperature-calibrated and gated by the `ImageQualityGate` to suppress its high false positive rate on pristine images.
- A prototype must remain strictly within `backend/agents/experimental/` and never alter live website routes.

### Question 7: What is the minimum next step to improve detection reliability?
**Answer**:
1. **Curate a Balanced Static Portrait Dataset**: The project currently possesses 0 static diffusion/GAN images in its training split. Adding 2,000–5,000 diverse static real portraits (e.g., FFHQ, CelebA-HQ) and static synthetic faces (StyleGAN2/3, Midjourney v6, FLUX, SDXL) is mandatory.
2. **Retrain ResNet-18 Beyond 2 Epochs**: Retrain the model for 15–20 epochs on a hybrid split combining video frames and static portraits until validation loss converges and ECE drops below 0.05.
3. **Calibrate Confidence**: Fit temperature scaling ($T$) on an independent validation set before setting operating thresholds.

---

## 2. Empirical Benchmark Results

### 2.1 Smoke Test on 5 Canonical Images

| Model | `profile.jpeg` (REAL) | `she.jpeg` (REAL) | `picture.jpeg` (REAL) | `deepy.jpeg` (FAKE) | `professional.jpeg` (FAKE) | Real Acc (3) | Fake Acc (2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`prod_resnet18`** | 0.0698 (REAL) | 0.0475 (REAL) | 0.0466 (REAL) | 0.1025 (REAL) ❌ | 0.0846 (REAL) ❌ | **3/3 (100%)** | **0/2 (0%)** |
| **`sbi_effnet_b4`** | 1.0000 (FAKE) ❌ | 1.0000 (FAKE) ❌ | 0.8782 (FAKE) ❌ | 1.0000 (FAKE) ✅ | 1.0000 (FAKE) ✅ | **0/3 (0%)** | **2/2 (100%)** |
| **`xception`** | 0.8463 (FAKE) ❌ | 0.8361 (FAKE) ❌ | 0.5256 (FAKE) ❌ | 0.8564 (FAKE) ✅ | 0.4511 (REAL) ❌ | **0/3 (0%)** | **1/2 (50%)** |
| **`vit_base`** | 0.9786 (FAKE) ❌ | 0.9953 (FAKE) ❌ | 0.9977 (FAKE) ❌ | 0.9900 (FAKE) ✅ | 0.9922 (FAKE) ✅ | **0/3 (0%)** | **2/2 (100%)** |
| **`full_image_resnet18`**| 1.0000 (FAKE) ❌ | 0.2359 (REAL) ✅ | 0.0007 (REAL) ✅ | 0.2979 (REAL) ❌ | 0.9975 (FAKE) ✅ | **2/3 (66.7%)** | **1/2 (50%)** |
| **`wang_resnet50`** | 0.0000 (REAL) ✅ | 0.0002 (REAL) ✅ | 0.0000 (REAL) ✅ | 0.0082 (REAL) ❌ | 0.0000 (REAL) ❌ | **3/3 (100%)** | **0/2 (0%)** |

*Takeaway: No candidate model gets all 5 canonical images right on raw threshold 0.50. The production model has high specificity on real images but collapses on fakes; SBI, Xception, and ViT catch fakes but collapse on real images by predicting high fake probabilities across the board.*

---

### 2.2 Validation Set Benchmark (60 Video Frames: 30 Real, 30 Fake)

| Model Name | Accuracy | Precision | Recall | Specificity | FPR | FNR | Macro-F1 | ROC-AUC | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`production_resnet18`** | 50.00% | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3333 | 0.5056 | **28.5 ms** |
| **`sbi_effnet_b4`** | 56.67% | 0.5370 | 0.9667 | 0.1667 | 0.8333 | 0.0333 | 0.4841 | 0.5444 | 245.2 ms |
| **`xception`** | 66.67% | 0.6136 | 0.9000 | 0.4333 | 0.5667 | 0.1000 | 0.6475 | 0.8100 | 115.8 ms |
| **`vit_base`** | 53.33% | 0.5200 | 0.8667 | 0.2000 | 0.8000 | 0.1333 | 0.4750 | 0.5111 | 840.4 ms |
| **`full_image_resnet18`** | **91.67%** | **0.8788** | **0.9667** | **0.8667** | **0.1333** | **0.0333** | **0.9165** | **0.9467** | 29.1 ms |

---

### 2.3 Still Image Evaluation Set (53 Facial Images: 24 Real, 29 Fake)

| Model Name | Accuracy | Precision | Recall (Fake) | Specificity (Real) | FPR | FNR | Macro-F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`production_resnet18`** | 45.28% | 0.0000 | 0.0000 | **1.0000** | **0.0000** | 1.0000 | 0.3117 | 0.5848 | 0.6038 |
| **`sbi_effnet_b4`** | 58.49% | 0.5686 | **1.0000** | 0.0833 | 0.9167 | **0.0000** | 0.4394 | 0.6250 | 0.7888 |
| **`xception`** | 66.04% | 0.6222 | 0.9655 | 0.2917 | 0.7083 | 0.0345 | 0.5971 | 0.8836 | **0.9152** |
| **`vit_base`** | 58.49% | 0.5778 | 0.8966 | 0.2083 | 0.7917 | 0.1034 | 0.5076 | 0.5438 | 0.5398 |
| **`full_image_resnet18`** | **79.25%** | **0.7812** | 0.8621 | 0.7083 | 0.2917 | 0.1379 | **0.7876** | **0.8922** | 0.9065 |

---

### 2.4 Breakdown: Static AI-Generated Portraits vs. Video-Extracted Face Swaps

| Category | Sample Count | `prod_resnet18` Acc | `sbi_effnet_b4` Acc | `xception` Acc | `vit_base` Acc | `full_image_resnet18` Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Genuine Celebrity Video Portraits** | 5 | **100%** (0.224) | 0% (1.000) ❌ | 40% (0.580) | 60% (0.547) | **100%** (0.087) |
| **Genuine YouTube Video Portraits** | 5 | **100%** (0.144) | 0% (0.984) ❌ | 0% (0.705) ❌ | 40% (0.623) | **100%** (0.003) |
| **Genuine FF++ Original Frames** | 10 | **100%** (0.092) | 20% (0.850) ❌ | 50% (0.384) | 0% (0.908) ❌ | 50% (0.469) |
| **Genuine In-the-Wild Photos** | 3 | **100%** (0.062) | 0% (0.959) ❌ | 0% (0.739) ❌ | 0% (0.985) ❌ | 33% (0.667) |
| **Celeb-DF Synthetic Face Swaps** | 8 | 0% (0.254) ❌ | **100%** (1.000) | **100%** (0.796) | 75% (0.812) | **100%** (1.000) |
| **FF++ Deepfakes Face Swaps** | 5 | 0% (0.094) ❌ | **100%** (1.000) | **100%** (0.933) | 80% (0.750) | 60% (0.740) |
| **FF++ FaceSwap Graphics Swaps** | 5 | 0% (0.067) ❌ | **100%** (1.000) | **100%** (0.980) | **100%** (0.993) | **100%** (1.000) |
| **FF++ Face2Face Reenactments** | 5 | 0% (0.149) ❌ | **100%** (1.000) | **100%** (0.882) | **100%** (0.947) | **100%** (0.956) |
| **FF++ NeuralTextures Reenactments**| 4 | 0% (0.092) ❌ | **100%** (1.000) | **100%** (0.874) | **100%** (0.881) | 75% (0.707) |
| **In-the-Wild Face Swap (`deepy`)** | 1 | 0% (0.103) ❌ | **100%** (1.000) | **100%** (0.856) | **100%** (0.990) | 0% (0.298) ❌ |
| **AI Synthetic Portrait (`professional`)**| 1 | 0% (0.085) ❌ | **100%** (1.000) | 0% (0.451) ❌ | **100%** (0.992) | **100%** (0.998) |

*(Numbers in parentheses represent the mean predicted fake probability for that category).*

---

## 3. Deep Analysis of Candidate Models

### Candidate 1: Production ResNet-18 (`image_model.pth`)
- **Strengths**: Ultra-low false positive rate ($0.0\%$ on the 53-image still evaluation set; never falsely accuses a genuine photo). Lightweight ($42.7\text{ MB}$), ultra-fast CPU inference ($28.5\text{ ms}$). Fully integrated with the multi-agent decision system and database.
- **Weaknesses**: Trained for only 2 epochs on video frames. Fails to detect static AI-generated faces and high-resolution face swaps when evaluating raw probabilities alone.
- **Verdict**: Keep active in production because SatyaDrishti's multi-agent system (`ForensicEvidenceSuite`, `SafetyDecisionSystem`) catches its blind spots and escalates suspect images to `UNCERTAIN`.

### Candidate 2: Self-Blended Images (SBI) EfficientNet-B4 (`sbi_ffc23.tar`)
- **Strengths**: 100% detection rate on all evaluated fake categories (Celeb-DF, Deepfakes, FaceSwap, Face2Face, NeuralTextures, `deepy`, `professional`). Excellent representation of boundary blending seams.
- **Weaknesses**: Massive False Positive Rate ($91.67\%$ on pristine images) without temperature scaling and multi-agent gating. Heavy computation ($245\text{ ms}$ on CPU). Non-commercial academic research license only.
- **Verdict**: Excellent candidate for an **isolated experimental agent** when combined with strict quality gating. Unsuitable as a direct drop-in replacement for production without calibration.

### Candidate 3: DeepfakeBench Xception (`xception_best.pth`)
- **Strengths**: Highest ROC-AUC on still images ($0.8836$) and PR-AUC ($0.9152$). Robust feature extractor across FaceForensics++ manipulations.
- **Weaknesses**: Still suffers from a $70.8\%$ False Positive Rate on wild photos. Misses `professional.jpeg` ($0.4511 < 0.50$).
- **Verdict**: Retain in `backend/agents/experimental/` for benchmark comparison.

### Candidate 4: Full Image ResNet-18 (`runs/full_image_I1/model_best.pth`)
- **Strengths**: Best overall Macro-F1 ($0.7876$), highest Accuracy ($79.25\%$), and balanced Specificity ($70.83\%$). Successfully catches `professional.jpeg` ($0.9975$). Identical architecture to production, allowing instant drop-in compatibility.
- **Weaknesses**: Misses `deepy.jpeg` ($0.2979$). Has a $29.17\%$ False Positive Rate on genuine photos.
- **Verdict**: Strongest internal candidate weights for fine-tuning.

---

## 4. Phase 8 Isolated Prototype Verification

An isolated prototype evaluation suite has been established under `scratch/run_model_comparison_diagnostic.py` and `scratch/run_comprehensive_evaluation.py`. 

### Isolation Guarantees:
1. **Production Checkpoints Untouched**: `backend/models/weights/image_model.pth` and `video_model.pth` were NOT modified.
2. **Production Routing Untouched**: API routes in `backend/api/analysis.py` and orchestrator logic in `backend/core/orchestrator.py` remain unchanged.
3. **Database Untouched**: No schema alterations or synthetic test entries were injected into production SQLite databases.
4. **Experimental Isolation**: All experimental agent classes reside strictly in `backend/agents/experimental/` and cannot affect live website users.

---

## 5. Actionable Roadmap (Smallest isolated next steps)

To permanently resolve the static portrait detection gap without disrupting the working website:

1. **Phase A (Data Ingestion)**: Add a curated static portrait split of 1,500 real photos (FFHQ/CelebA-HQ) and 1,500 synthetic AI portraits (Midjourney/FLUX/StyleGAN) to `SatyaDrishti_Datasets`.
2. **Phase B (Training Run)**: Train `ImageModel` (ResNet-18) for 15 epochs on this combined dataset, using early stopping on validation Macro-F1.
3. **Phase C (Calibration)**: Fit Platt temperature scaling ($T$) strictly on the validation split. Verify that the threshold at $0.50$ maintains $\text{FPR} < 5\%$ before promoting weights to `backend/models/weights/image_model.pth`.
