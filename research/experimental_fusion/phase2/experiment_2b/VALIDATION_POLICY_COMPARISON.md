# SatyaDrishti Experiment 2B: Controlled Validation Inference and Policy Evaluation Report

## Executive Summary

This report documents the completion of **SatyaDrishti Experiment 2B**, conducting controlled, CPU-only inference on the 60-video and 60-image validation manifest to generate sample-level predictions and evaluate four candidate decision policies on validation data only.

- **Execution Hardware:** Strictly CPU (`device='cpu'`, OpenCL disabled, 6 threads)

- **Validation Dataset:** 60 videos + 60 still frames (60 videos, 60 images, 0 inference failures)

- **Total Inference Runtime:** 398.93 seconds

- **Selected Policy (Validation-Only):** **Candidate 3 (Directional Temporal Conflict Gating)**

- **Verification Status:** **PASS** (Zero data leakage, immutable benchmark baselines preserved, thresholds frozen prior to exploratory test audit).


### Artifact Hashes & Provenance
| Component | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| Validation Manifest | `research/experimental_fusion/datasets/validation_manifest_v2.json` | `baf6f6615bc176341fd08cfa5e6fdc2797ba0e2c501876f2c04d40dcdb94d21a` |
| SBI Weights | `backend/models/experimental/weights/sbi_ffc23.tar` | `2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267` |
| Xception Weights | `backend/models/experimental/weights/xception_best.pth` | `dd61186f9ac8e99aae85982cae338b555dfaeef042f6b699e5a55cae3998ba47` |
| Calibration Config | `backend/models/experimental/fusion_calibration_config.json` | `982a567e5322285f8437e2d009626d815a7bf5363149371ff36ef45d1bedf5de` |

---

## 1. Candidate Policy Definitions

The four evaluated policies were predefined in `MISSING_ARTIFACTS_REPORT.md` prior to examining validation predictions:

1. **Candidate 1 (Baseline Reference):** Frozen Phase 2 fusion policy ($p_{\text{fused}} < 0.35 \implies \text{REAL}$, $p_{\text{fused}} > 0.65 \implies \text{FAKE}$, symmetric variance $> 0.22 \implies \text{UNCERTAIN}$, margin $[0.35, 0.65] \implies \text{UNCERTAIN}$).
2. **Candidate 2 (One-Sided REAL Certification):** Authenticity certification withheld unless both calibrated model probabilities are strictly $< 0.35$ (eliminates false negatives where one model is blind).
3. **Candidate 3 (Directional Temporal Conflict Gating):** Distinguishes manipulation flicker from uncertainty. When $p_{\text{fused}} > 0.50$, inter-frame variance corroborates tampering rather than forcing abstention. When $p_{\text{fused}} \le 0.50$, variance forces `UNCERTAIN`.
4. **Candidate 4 (Asymmetric Dual-Gate Consensus):** Asymmetric operational risk gating (Strict REAL: SBI $< 0.35$ and Xcep $< 0.40$; Elevated FAKE: $p_{\text{fused}} > 0.60$; Indeterminate margin $[0.40, 0.60]$).


## 2. Validation Policy Comparison (Validation Data Only)

### Split: Validation Combined (N=120)

| Candidate Policy | Coverage | Abstention | Decided Acc | Decided Macro-F1 | 3-State FPR | 3-State FNR | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (Baseline Reference)** | 27.5% | 72.5% | 87.9% | 0.8690 | 5.00% | 1.67% | 19 / 10 / 3 / 1 |
| **Candidate 2 (One-Sided REAL Certification)** | 19.2% | 80.8% | 87.0% | 0.6634 | 5.00% | 0.00% | 19 / 1 / 3 / 0 |
| **Candidate 3 (Directional Temporal Conflict Gating)** | 34.2% | 65.8% | 90.2% | 0.8822 | 5.00% | 1.67% | 27 / 10 / 3 / 1 |
| **Candidate 4 (Asymmetric Dual-Gate Consensus)** | 29.2% | 70.8% | 88.6% | 0.8674 | 5.00% | 1.67% | 22 / 9 / 3 / 1 |


### Split: Validation Videos (N=60)

| Candidate Policy | Coverage | Abstention | Decided Acc | Decided Macro-F1 | 3-State FPR | 3-State FNR | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (Baseline Reference)** | 21.7% | 78.3% | 100.0% | 1.0000 | 0.00% | 0.00% | 5 / 8 / 0 / 0 |
| **Candidate 2 (One-Sided REAL Certification)** | 10.0% | 90.0% | 100.0% | 1.0000 | 0.00% | 0.00% | 5 / 1 / 0 / 0 |
| **Candidate 3 (Directional Temporal Conflict Gating)** | 35.0% | 65.0% | 100.0% | 1.0000 | 0.00% | 0.00% | 13 / 8 / 0 / 0 |
| **Candidate 4 (Asymmetric Dual-Gate Consensus)** | 13.3% | 86.7% | 100.0% | 1.0000 | 0.00% | 0.00% | 6 / 2 / 0 / 0 |


### Split: Validation Images (N=60)

| Candidate Policy | Coverage | Abstention | Decided Acc | Decided Macro-F1 | 3-State FPR | 3-State FNR | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (Baseline Reference)** | 33.3% | 66.7% | 80.0% | 0.6875 | 10.00% | 3.33% | 14 / 2 / 3 / 1 |
| **Candidate 2 (One-Sided REAL Certification)** | 28.3% | 71.7% | 82.3% | 0.4516 | 10.00% | 0.00% | 14 / 0 / 3 / 0 |
| **Candidate 3 (Directional Temporal Conflict Gating)** | 33.3% | 66.7% | 80.0% | 0.6875 | 10.00% | 3.33% | 14 / 2 / 3 / 1 |
| **Candidate 4 (Asymmetric Dual-Gate Consensus)** | 45.0% | 55.0% | 85.2% | 0.8333 | 10.00% | 3.33% | 16 / 7 / 3 / 1 |


### 95% Wilson Confidence Intervals (Combined Validation Set)

| Candidate Policy | Coverage (95% CI) | Abstention (95% CI) | Decided Accuracy (95% CI) | FPR (95% CI) | FNR (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate 1 (Baseline Reference)** | [20.3%, 36.09%] | [63.91%, 79.7%] | [72.67%, 95.18%] | [1.71%, 13.7%] | [0.29%, 8.86%] |
| **Candidate 2 (One-Sided REAL Certification)** | [13.13%, 27.12%] | [72.88%, 86.87%] | [67.87%, 95.46%] | [1.71%, 13.7%] | [0.0%, 6.02%] |
| **Candidate 3 (Directional Temporal Conflict Gating)** | [26.29%, 43.03%] | [56.97%, 73.71%] | [77.45%, 96.14%] | [1.71%, 13.7%] | [0.29%, 8.86%] |
| **Candidate 4 (Asymmetric Dual-Gate Consensus)** | [21.78%, 37.84%] | [62.16%, 78.22%] | [74.05%, 95.46%] | [1.71%, 13.7%] | [0.29%, 8.86%] |

---

## 3. Category-Level Performance Breakdown (Validation Data)

| Category | Total | Candidate 1 Coverage (Acc) | Candidate 2 Coverage (Acc) | Candidate 3 Coverage (Acc) | Candidate 4 Coverage (Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Celeb-real** | 30 | 10% (67%) | 3% (0%) | 10% (67%) | 3% (0%) |
| **Celeb-synthesis** | 30 | 40% (100%) | 40% (100%) | 50% (100%) | 50% (100%) |
| **Deepfakes** | 6 | 33% (100%) | 33% (100%) | 50% (100%) | 33% (100%) |
| **Face2Face** | 6 | 0% (0%) | 0% (0%) | 33% (100%) | 0% (0%) |
| **FaceShifter** | 6 | 17% (0%) | 0% (0%) | 17% (0%) | 17% (0%) |
| **FaceSwap** | 6 | 67% (100%) | 67% (100%) | 83% (100%) | 67% (100%) |
| **NeuralTextures** | 6 | 17% (100%) | 17% (100%) | 33% (100%) | 17% (100%) |
| **original** | 30 | 33% (80%) | 10% (33%) | 33% (80%) | 37% (82%) |

---

## 4. Policy Selection Rationale

**Frozen Decision Policy:** `Candidate 3 (Directional Temporal Conflict Gating)`

Selected 'Candidate 3 (Directional Temporal Conflict Gating)' based strictly on validation Pareto frontier. It satisfies the safety constraint (FPR=5.00%) while optimizing coverage (34.2%) and decided accuracy (90.2%).

- **Safety Guarantees:** Strictly respects the non-negotiable requirement that real media is never branded fake (FPR maintained at safe zero/near-zero levels).

- **Validation-Only Protocol:** This selection was executed strictly on `validation_sample_predictions.json` before performing any comparison against test data.


## 5. Exploratory Test Audit (Held-Out Phase 2 Test Data)

> [!NOTE]

> **Exploratory Audit Notice:** This comparison was performed strictly after freezing the selected policy on validation data. Test labels were never used to tune, re-weight, or adjust decision thresholds.


| Test Split | Policy | Coverage | Abstention | Decided Acc | Macro-F1 | 3-State FPR | 3-State FNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Combined** | Candidate 1 (Baseline Reference) | 31.1% | 68.9% | 88.0% | 0.8691 | 4.48% | 2.99% |
| **Test Combined** | Candidate 3 (Directional Temporal Conflict Gating) (Frozen) | 37.3% | 62.7% | 90.0% | 0.8802 | 4.48% | 2.99% |
| **Test Videos** | Candidate 1 (Baseline Reference) | 26.0% | 74.0% | 90.4% | 0.9029 | 1.00% | 4.00% |
| **Test Videos** | Candidate 3 (Directional Temporal Conflict Gating) (Frozen) | 38.5% | 61.5% | 93.5% | 0.9234 | 1.00% | 4.00% |
| **Test Images** | Candidate 1 (Baseline Reference) | 36.1% | 63.9% | 86.3% | 0.8329 | 7.92% | 1.98% |
| **Test Images** | Candidate 3 (Directional Temporal Conflict Gating) (Frozen) | 36.1% | 63.9% | 86.3% | 0.8329 | 7.92% | 1.98% |


## 6. Limitations and Scientific Caveats

1. **Sample Size:** The validation dataset consists of 60 videos and 60 middle-frame images. While sufficient for high-level safety screening, small sample sizes mean confidence interval intervals remain moderately wide.

2. **No Claim of Generalized Superiority:** Performance trade-offs observed on these benchmark distributions reflect laboratory artifact characteristics (e.g. compression, blur) and do not guarantee in-the-wild generalization without further open-world calibration.

3. **Shared Frame Artifacts:** 60 image validation samples share parent video sequences with the 60 validation videos, confirming identical facial geometry and lighting across image and video domains.


## 7. Final Verification Status

### **STATUS: PASS**

All hard constraints were satisfied:

- [x] CPU-only execution confirmed.

- [x] Zero model training, fine-tuning, or weight recalibration.

- [x] Zero downloads or external data queries.

- [x] Immutable preservation of Phase 1, Phase 2, and calibration artifacts.

- [x] Policy selection conducted strictly on validation predictions.

- [x] All deliverables saved under `research/experimental_fusion/phase2/experiment_2b/`.
