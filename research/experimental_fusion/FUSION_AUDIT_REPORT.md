# Independent Forensic Audit Report: SatyaDrishti SBI + Xception Fusion Benchmark

**Audit Date**: 2026-10-12  
**Auditor**: Antigravity Independent Forensic Evaluation Agent  
**Target Artifacts**:
- Report: [`research/experimental_fusion/FUSION_BENCHMARK_REPORT.md`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/FUSION_BENCHMARK_REPORT.md)
- Data: [`research/experimental_fusion/fusion_benchmark_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/fusion_benchmark_results.json)
- Code: [`backend/agents/experimental/experimental_fusion_agent.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/agents/experimental/experimental_fusion_agent.py)
- Evaluator: [`scratch/run_fusion_benchmark.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/scratch/run_fusion_benchmark.py)

---

## 1. Audit Verdict

### **VERDICT: VERIFIED WITH LIMITATIONS**

The mathematical calculations, confusion matrices, binary metrics, and three-state safety decisions reported in the SBI + Xception fusion benchmark were **independently recomputed from raw sample-level data and verified to be 100% computationally authentic and reproducible**.

However, the verdict is assigned **WITH LIMITATIONS** due to four specific architectural and statistical constraints identified during this audit:
1. **Sample Size & Wide Confidence Intervals**: Evaluation sample sizes ($N=53$ facial images, $N=40$ video clips) are exploratory. 95% Wilson confidence intervals span $\pm 10\%$ to $\pm 12\%$. The results demonstrate proof-of-concept directional superiority, **not definitive evidence of open-world generalization**.
2. **Duplicate Image in Evaluation Manifest**: A bit-for-bit duplicate was confirmed (`inwild_profile.jpeg` and `inwild_HON1.jpg`, MD5: `4c3e1c3789ccd396bbfaa53390bf259b`). Recomputing on the deduplicated dataset yields 75.0% accuracy (vs 75.5%), confirming robustness, but the duplicate should be removed in production trials.
3. **Pre-Fixed Calibration Transfer**: Temperature scaling ($T=1.885$) was pre-fixed from the production ResNet-18 validation split (preventing test set leakage), but was transferred zero-shot to SBI and Xception rather than fitted to architecture-specific validation splits.
4. **Minor Narrative Text Discrepancies**: In Section 6.1 of the report, minor rounding discrepancies in Macro-F1 text were identified (0.7419 vs 0.7415 for images; 0.8742 vs 0.8749 for videos), though the primary tables in Sections 3.1 and 4.1 are exact.

---

## 2. Independent Metric Recomputation & Verification

Every metric was recalculated directly from sample-level labels ($y_{\text{true}}$) and predictions ($y_{\text{pred}}$) without relying on intermediate summaries. 95% confidence intervals were computed using the **Wilson Score Interval**.

### 2.1 Still Images: Forced Binary Mode (Cutoff = 0.50)
Evaluated on $N = 53$ facial images (24 REAL, 29 FAKE) excluding 2 non-face safety edge cases:

| Metric | Production ResNet-18 | Experimental Xception | Pretrained SBI (B4) | **SBI + Xception Fusion** | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 0 / 29 | 28 / 29 | 21 / 29 | **26 / 29** | **VERIFIED** |
| **False Negatives (FN)** | 29 / 29 | 1 / 29 | 8 / 29 | **3 / 29** | **VERIFIED** |
| **True Negatives (TN)** | 19 / 24 | 4 / 24 | 15 / 24 | **14 / 24** | **VERIFIED** |
| **False Positives (FP)** | 5 / 24 | 20 / 24 | 9 / 24 | **10 / 24** | **VERIFIED** |
| **Accuracy** | 35.9% [24.3% - 49.3%] | 60.4% [46.9% - 72.4%] | 67.9% [54.5% - 78.9%] | **75.5% [62.4% - 85.1%]** | **VERIFIED** |
| **Precision** | 0.0% [0.0% - 43.5%] | 58.3% [44.3% - 71.2%] | 70.0% [52.1% - 83.3%] | **72.2% [56.0% - 84.2%]** | **VERIFIED** |
| **Recall (TPR)** | 0.0% [0.0% - 11.7%] | 96.5% [82.8% - 99.4%] | 72.4% [54.3% - 85.3%] | **89.7% [73.6% - 96.4%]** | **VERIFIED** |
| **Specificity (TNR)** | 79.2% [59.5% - 90.8%] | 16.7% [6.7% - 35.9%] | 62.5% [42.7% - 78.8%] | **58.3% [38.8% - 75.5%]** | **VERIFIED** |
| **False Positive Rate** | 20.8% [9.2% - 40.5%] | 83.3% [64.1% - 93.3%] | 37.5% [21.2% - 57.3%] | **41.7% [24.5% - 61.2%]** | **VERIFIED** |
| **False Negative Rate** | 100.0% [88.3% - 100.0%] | 3.5% [0.6% - 17.2%] | 27.6% [14.7% - 45.7%] | **10.3% [3.6% - 26.4%]** | **VERIFIED** |
| **F1 Score (Binary)** | 0.0000 | 0.7273 | 0.7119 | **0.8000** | **VERIFIED** |
| **Macro-F1 Score** | 0.2639 | 0.5016 | 0.6751 | **0.7415** | **VERIFIED** |
| **Mean Latency per Image** | 0.94s | 0.32s | 0.52s | **0.84s** | **VERIFIED** |

---

### 2.2 Still Images: Safety-Gated Three-State Performance

| Model | Ground Truth REAL (24) | Ground Truth FAKE (29) | Overall Abstention | Non-Face Rejection | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline** | 7 `REAL`, 17 `UNC`, **0 `FAKE` (0.0% FP)** | 0 `FAKE`, 20 `UNC`, **9 `REAL` (31.0% FN)** | 69.8% [56.5% - 80.5%] | **2 / 2 (100.0%)** | **VERIFIED** |
| **Experimental Xception** | 1 `REAL`, 18 `UNC`, **5 `FAKE` (20.8% FP)** | 21 `FAKE`, 8 `UNC`, **0 `REAL` (0.0% FN)** | 49.1% [36.1% - 62.1%] | **2 / 2 (100.0%)** | **VERIFIED** |
| **Pretrained SBI (B4)** | 7 `REAL`, 16 `UNC`, **1 `FAKE` (4.2% FP)** | 17 `FAKE`, 10 `UNC`, **2 `REAL` (6.9% FN)** | 49.1% [36.1% - 62.1%] | **2 / 2 (100.0%)** | **VERIFIED** |
| **SBI + Xception Fusion** | 10 `REAL`, 12 `UNC`, **2 `FAKE` (8.3% FP)** | 24 `FAKE`, 3 `UNC`, **2 `REAL` (6.9% FN)** | **28.3% [18.0% - 41.6%]** | **2 / 2 (100.0%)** | **VERIFIED** |

---

### 2.3 Videos: Forced Binary Mode (Cutoff = 0.50)
Evaluated on $N = 40$ balanced video clips (20 REAL, 20 FAKE) with 8 uniform frames sampled per video:

| Metric | Production Baseline | Experimental Xception | SBI EfficientNet-B4 | **SBI + Xception Fusion** | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 10 / 20 | 20 / 20 | 14 / 20 | **18 / 20** | **VERIFIED** |
| **False Negatives (FN)** | 10 / 20 | 0 / 20 | 6 / 20 | **2 / 20** | **VERIFIED** |
| **True Negatives (TN)** | 16 / 20 | 8 / 20 | 18 / 20 | **17 / 20** | **VERIFIED** |
| **False Positives (FP)** | 4 / 20 | 12 / 20 | 2 / 20 | **3 / 20** | **VERIFIED** |
| **Accuracy** | 65.0% [49.5% - 77.9%] | 70.0% [54.6% - 81.9%] | 80.0% [65.2% - 89.5%] | **87.5% [73.9% - 94.5%]** | **VERIFIED** |
| **Precision** | 71.4% [45.4% - 88.3%] | 62.5% [45.2% - 77.1%] | 87.5% [64.0% - 96.5%] | **85.7% [65.4% - 95.0%]** | **VERIFIED** |
| **Recall (TPR)** | 50.0% [29.9% - 70.1%] | 100.0% [83.9% - 100.0%] | 70.0% [48.1% - 85.5%] | **90.0% [69.9% - 97.2%]** | **VERIFIED** |
| **Specificity (TNR)** | 80.0% [58.4% - 91.9%] | 40.0% [21.9% - 61.3%] | 90.0% [69.9% - 97.2%] | **85.0% [64.0% - 94.8%]** | **VERIFIED** |
| **False Positive Rate** | 20.0% [8.1% - 41.6%] | 60.0% [38.7% - 78.1%] | 10.0% [2.8% - 30.1%] | **15.0% [5.2% - 36.0%]** | **VERIFIED** |
| **False Negative Rate** | 50.0% [29.9% - 70.1%] | 0.0% [0.0% - 16.1%] | 30.0% [14.5% - 51.9%] | **10.0% [2.8% - 30.1%]** | **VERIFIED** |
| **F1 Score (Binary)** | 0.5882 | 0.7692 | 0.7778 | **0.8780** | **VERIFIED** |
| **Macro-F1 Score** | 0.6419 | 0.6703 | 0.7980 | **0.8749** | **VERIFIED** |
| **Mean Latency per Video** | 5.21s | 2.46s | 3.44s | **5.91s** | **VERIFIED** |

---

### 2.4 Videos: Safety-Gated Three-State Performance

| Model | Ground Truth REAL (20) | Ground Truth FAKE (20) | Overall Abstention | False Positives | False Negatives | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline** | 0 `REAL`, 20 `UNC` | 0 `FAKE`, 20 `UNC` | 100.0% [91.2% - 100.0%] | **0 (0.0%)** | **0 (0.0%)** | **VERIFIED** |
| **Experimental Xception** | 2 `REAL`, 17 `UNC`, 1 `FAKE` | 12 `FAKE`, 8 `UNC` | 62.5% [47.0% - 75.8%] | 1 (5.0%) | **0 (0.0%)** | **VERIFIED** |
| **SBI EfficientNet-B4** | 11 `REAL`, 9 `UNC` | 11 `FAKE`, 6 `UNC`, 3 `REAL` | 37.5% [24.2% - 53.0%] | **0 (0.0%)** | 3 (15.0%) | **VERIFIED** |
| **SBI + Xception Fusion** | 13 `REAL`, 7 `UNC` | 11 `FAKE`, 9 `UNC` | **40.0% [26.4% - 55.4%]** | **0 (0.0%)** | **0 (0.0%)** | **VERIFIED** |

---

## 3. Discrepancy Analysis: 41.7% Binary FPR vs 8.3% Three-State FPR

The prompt requested a root-cause explanation for the substantial divergence between the forced binary false-positive rate (**41.7%**, 10/24) and the three-state false-positive rate (**8.3%**, 2/24) on still images under fusion.

### Trace of All 10 Binary False-Positive Samples:
Auditing the 24 REAL image samples revealed the exact trace of the 10 samples flagged `FAKE` under forced binary mode:

| Filename | Dataset Category | SBI Raw Score | Xception Raw Score | Fused Calibrated Prob | Binary Verdict ($p \ge 0.50$) | Three-State Decision System Verdict | Root-Cause Explanation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `real_celeb_portrait_00.jpg` | genuine_portrait_celeb | 0.757 | 0.715 | **0.634** | `FAKE` | `UNCERTAIN` | Fused prob (0.634) falls in ambiguous margin $[0.35, 0.65]$. |
| `real_celeb_portrait_04.jpg` | genuine_portrait_celeb | 0.500 | 0.500 | **0.500** | `FAKE` | `UNCERTAIN` | Quality gate defect $\rightarrow$ 0.50 fallback safely abstained. |
| `real_youtube_portrait_00.jpg` | genuine_portrait_youtube | 0.583 | 0.862 | **0.696** | `FAKE` | **`FAKE` (FP #1)** | Both models leaned synthetic ($p > 0.65$); true false alarm. |
| `real_youtube_portrait_01.jpg` | genuine_portrait_youtube | 0.500 | 0.500 | **0.500** | `FAKE` | `UNCERTAIN` | Quality fallback safely abstained. |
| `real_youtube_portrait_02.jpg` | genuine_portrait_youtube | 0.849 | 0.770 | **0.690** | `FAKE` | **`FAKE` (FP #2)** | Both models leaned synthetic ($p > 0.65$); true false alarm. |
| `real_ff_original_04.jpg` | genuine_portrait_ff | 0.683 | 0.585 | **0.584** | `FAKE` | `UNCERTAIN` | Fused prob (0.584) falls in ambiguous margin $[0.35, 0.65]$. |
| `real_ff_original_05.jpg` | genuine_portrait_ff | 0.368 | 0.795 | **0.602** | `FAKE` | `UNCERTAIN` | Fused prob (0.602) falls in ambiguous margin $[0.35, 0.65]$. |
| `real_ff_original_06.jpg` | genuine_portrait_ff | 0.500 | 0.500 | **0.500** | `FAKE` | `UNCERTAIN` | Quality fallback safely abstained. |
| `real_ff_original_08.jpg` | genuine_portrait_ff | 0.500 | 0.500 | **0.500** | `FAKE` | `UNCERTAIN` | Quality fallback safely abstained. |
| `safety_she.jpeg` | degraded_lowlight_real | 0.500 | 0.500 | **0.500** | `FAKE` | `UNCERTAIN` | Low-light quality failure safely abstained. |

### Mathematical Finding:
- **8 out of the 10 binary false positives** had calibrated probabilities falling strictly inside the indeterminate decision boundary $[0.35, 0.65]$ (including 5 low-quality fallback samples at $p = 0.500$, and 3 weak/borderline samples at $p = 0.584, 0.602, 0.634$).
- Forced binary classification applies an arbitrary step function at $0.50$, forcing all 8 ambiguous samples into `FAKE`, producing a misleadingly elevated **41.7% FPR**.
- The three-state engine identifies that certainty is insufficient ($0.35 \le p \le 0.65$) and routes all 8 cases to `UNCERTAIN` abstention.
- Only **2 authentic images** (`real_youtube_portrait_00.jpg` and `real_youtube_portrait_02.jpg`) produced strong concordant synthetic predictions ($p > 0.65$), yielding an authentic three-state **FPR of 8.3% (2/24)**.

---

## 4. Calibration & Data Leakage Audit

### 4.1 Provenance of Temperature $T = 1.885$
A deep codebase audit was conducted to trace the origin of $T = 1.885$:
- **Trace Source**: In [`backend/models/model_registry.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/model_registry.json#L12), the ImageModel validation run recorded:
  ```json
  "val_loss": 1.8858883180618287,
  "ece": 0.6838030628561973
  ```
- **Config Timestamp**: Saved in [`backend/models/calibration_config.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/calibration_config.json) on `2026-09-16T16:00:00`:
  `"temperature": 1.885, "tau_real": 0.35, "tau_fake": 0.65, "fitted_on": "validation_split_only"`.
- **Audit Confirmation**:
  1. Temperature $T = 1.885$ was pre-configured nearly a month before this fusion benchmark.
  2. **Zero test-set tuning occurred**: Neither $T$ nor the thresholds ($\tau_{\text{real}}, \tau_{\text{fake}}$) were tuned on the 55-image or 40-video held-out evaluation sets.
  3. **Data Leakage Risk**: **NONE**. The test sets were strictly held out.

### 4.2 Split Integrity Audit (FaceForensics++ and Celeb-DF)
- **FaceForensics++ Splitting**: [`backend/training/splits/check_splits.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/training/splits/check_splits.py) confirmed:
  - `Train & Test overlap: set()` (0 sequence overlap)
  - `Val & Test overlap: set()` (0 sequence overlap)
  - All FF++ original and manipulated sequences evaluated in this benchmark (`000`, `003`, `010`, `012`, `015`, `019`, `022`, `026`, `031`, `035`) belong exclusively to `test.json`.
- **Celeb-DF Splitting**: All Celeb-DF real and synthetic videos were sampled from the official held-out test list (`List_of_testing_videos.txt`).

---

## 5. Deduplication Audit & Sensitivity Analysis

### 5.1 Audit Findings:
- Hash audit confirmed that `inwild_profile.jpeg` and `inwild_HON1.jpg` share the identical MD5 digest:
  `4c3e1c3789ccd396bbfaa53390bf259b`.
- The evaluation manifest originally contained 55 entries, but only **54 distinct physical images** (52 facial images, 2 non-face documents).

### 5.2 Deduplicated Dataset Sensitivity Evaluation:
Excluding `inwild_HON1.jpg` leaves $N = 52$ facial images (23 REAL, 29 FAKE):

| Setup | Deduplicated Accuracy (Binary) | Deduplicated Precision | Deduplicated Recall | Deduplicated F1 | Deduplicated 3-State FP Rate | Deduplicated 3-State FN Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production Baseline** | 34.6% (18/52) | 0.0% | 0.0% | 0.0000 | 0.0% (0/23) | 31.0% (9/29) |
| **Experimental Xception** | 61.5% (32/52) | 59.6% | 96.5% | 0.7368 | 17.4% (4/23) | 0.0% (0/29) |
| **SBI EfficientNet-B4** | 67.3% (35/52) | 70.0% | 72.4% | 0.7119 | 4.3% (1/23) | 6.9% (2/29) |
| **SBI + Xception Fusion** | **75.0% (39/52)** | **72.2%** | **89.7%** | **0.8000** | **8.7% (2/23)** | **6.9% (2/29)** |

**Conclusion**: The metric delta from deduplication is only 0.5% (75.0% vs 75.5%), and the binary F1 score is identical (0.8000). The benchmark findings are completely robust against sample duplication.

---

## 6. Verification of UNCERTAIN Handling in Metrics

The prompt requested verification that `UNCERTAIN` cases are handled consistently and are **not silently counted as correct** in binary metrics.

### Audit Findings:
1. **Forced Binary Metrics**:
   - In forced binary mode, every sample is evaluated against the threshold $0.50$ ($p \ge 0.50 \rightarrow \text{FAKE}$, $p < 0.50 \rightarrow \text{REAL}$).
   - `UNCERTAIN` does not exist in binary mode.
   - For example, real samples with $p = 0.500$ or $p = 0.584$ were forced to `FAKE` and counted as **False Positives (errors)**, not correct predictions.
2. **Three-State Metrics**:
   - `UNCERTAIN` cases were recorded as abstentions.
   - In the benchmark report and JSON, `UNCERTAIN` is tracked in its own column (`uncertain_abstained`).
   - It was **NEVER** added to True Positives ($TP$) or True Negatives ($TN$).
3. **Accuracy Comparison**:
   - **Forced Binary Accuracy**: 75.5% (40/53) for images; 87.5% (35/40) for videos.
   - **3-State Raw Accuracy** (counting `UNCERTAIN` as an incorrect non-decision):
     $$\text{Acc}_{\text{raw}} = \frac{TP + TN}{N} = \frac{24 + 10}{53} = \mathbf{64.15\%} \quad (\text{Images}); \quad \frac{11 + 13}{40} = \mathbf{60.00\%} \quad (\text{Videos})$$
   - **3-State Decided Accuracy** (accuracy strictly on non-abstained decisions):
     $$\text{Acc}_{\text{decided}} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{34}{38} = \mathbf{89.47\%} \quad (\text{Images}); \quad \frac{24}{24} = \mathbf{100.00\%} \quad (\text{Videos})$$

**Conclusion**: There was **zero silent inflation**. All metrics strictly adhere to formal decision-theoretic standards.

---

## 7. Category-Level Recalculation & Reproducibility Audit

All 14 still-image categories and 8 video subsets were recalculated from the underlying sample-level results:

### Still-Image Category Recalculation:
| Category | GT Label | Sample Count | SBI Correct | Xception Correct | Production Correct | **Fusion Correct** | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ai_generated_synthetic_face` | FAKE | 1 | 1/1 | 0/1 | 0/1 | **1/1** | **VERIFIED** |
| `degraded_lowlight_real` | REAL | 1 | 0/1 | 0/1 | 0/1 | **0/1** | **VERIFIED** |
| `face_swap_celebsynth` | FAKE | 8 | 8/8 | 8/8 | 0/8 | **8/8** | **VERIFIED** |
| `face_swap_ff_deepfakes` | FAKE | 5 | 4/5 | 5/5 | 0/5 | **5/5** | **VERIFIED** |
| `face_swap_ff_faceswap` | FAKE | 5 | 5/5 | 5/5 | 0/5 | **5/5** | **VERIFIED** |
| `genuine_portrait_celeb` | REAL | 5 | 3/5 | 1/5 | 4/5 | **3/5** | **VERIFIED** |
| `genuine_portrait_ff` | REAL | 10 | 7/10 | 3/10 | 8/10 | **6/10** | **VERIFIED** |
| `genuine_portrait_youtube` | REAL | 5 | 2/5 | 0/5 | 4/5 | **2/5** | **VERIFIED** |
| `in_the_wild_face_swap` | FAKE | 1 | 0/1 | 1/1 | 0/1 | **0/1** | **VERIFIED** |
| `in_the_wild_real_portrait` | REAL | 3 | 3/3 | 0/3 | 3/3 | **3/3** | **VERIFIED** |
| `non_face_document` | NON-FACE | 1 | 1/1 | 1/1 | 1/1 | **1/1** | **VERIFIED** |
| `non_face_graphic` | NON-FACE | 1 | 1/1 | 1/1 | 1/1 | **1/1** | **VERIFIED** |
| `reenactment_ff_face2face` | FAKE | 5 | 2/5 | 5/5 | 0/5 | **4/5** | **VERIFIED** |
| `reenactment_ff_neuraltextures` | FAKE | 4 | 1/4 | 4/4 | 0/4 | **3/4** | **VERIFIED** |

### Video Subset Recalculation:
| Subset | GT Label | Clip Count | SBI Correct | Xception Correct | Production Correct | **Fusion Correct** | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Celeb-DF_Celeb-real` | REAL | 5 | 4/5 | 1/5 | 2/5 | **4/5** | **VERIFIED** |
| `Celeb-DF_YouTube-real` | REAL | 5 | 4/5 | 0/5 | 4/5 | **3/5** | **VERIFIED** |
| `Celeb-DF_synthesis` | FAKE | 10 | 10/10 | 10/10 | 10/10 | **10/10** | **VERIFIED** |
| `FF++_Deepfakes` | FAKE | 3 | 2/3 | 3/3 | 0/3 | **2/3** | **VERIFIED** |
| `FF++_Face2Face` | FAKE | 3 | 1/3 | 3/3 | 0/3 | **3/3** | **VERIFIED** |
| `FF++_FaceSwap` | FAKE | 2 | 1/2 | 2/2 | 0/2 | **2/2** | **VERIFIED** |
| `FF++_NeuralTextures` | FAKE | 2 | 0/2 | 2/2 | 0/2 | **1/2** | **VERIFIED** |
| `FF++_original` | REAL | 10 | 10/10 | 7/10 | 10/10 | **10/10** | **VERIFIED** |

---

## 8. Summary of All Identified Discrepancies & Audit Log

| ID | Issue Description | Impact Level | Resolution / Clarification |
| :---: | :--- | :---: | :--- |
| **D-1** | Exact duplicate image file: `inwild_profile.jpeg` and `inwild_HON1.jpg` have identical MD5 hash. | Minor | Deduplicated accuracy recomputed at 75.0% (vs 75.5%). Delta is $+0.5\%$. Documented in Section 5. |
| **D-2** | Typo in Report Section 6.1 narrative text: Macro-F1 listed as `0.7419` for still images, whereas Table 3.1 and JSON calculate `0.7415`. | Minor | Typo identified. Table 3.1 is the authoritative source ($0.7415$). |
| **D-3** | Typo in Report Section 6.1 narrative text: Macro-F1 listed as `0.8742` for videos, whereas Table 4.1 and JSON calculate `0.8749`. | Minor | Typo identified. Table 4.1 is the authoritative source ($0.8749$). |
| **D-4** | High forced-binary FPR on real images (41.7%). | Substantive | Fully explained in Section 3: forced binary cutoff $0.50$ forces 8 ambiguous/fallback samples into `FAKE`, whereas 3-state system routes them to `UNCERTAIN`, lowering FPR to 8.3%. |
| **D-5** | Small sample sizes ($N=53$ facial images, $N=40$ videos). | Substantive | Wilson 95% confidence intervals are wide ($\pm 10\% - 12\%$). Results must be treated as exploratory proof-of-concept, not conclusive proof of generalized deployment. |

---

## 9. Final Conclusion

The SBI + Xception fusion benchmark is mathematically sound, reproducible, and leak-free. The conservative disagreement rule functions as intended, suppressing false alarms and protecting against blind spots in 3-state gating. 

With the limitations and confidence intervals clearly established in this audit report, the experimental evaluation is **VERIFIED WITH LIMITATIONS** and is ready for stakeholder review. Do not deploy to production without large-scale validation.
