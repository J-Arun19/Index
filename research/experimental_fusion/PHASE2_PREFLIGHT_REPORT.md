# SatyaDrishti Phase 2 Locked Benchmark: Preflight Audit Report

**Audit Date**: 2026-10-12  
**Auditor**: Antigravity Preflight Verification Engine  
**Target Artifacts**:
- Validation Manifest: [`research/experimental_fusion/datasets/validation_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/validation_manifest_v2.json)
- Locked Test Manifest: [`research/experimental_fusion/datasets/locked_test_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/locked_test_manifest_v2.json)
- Calibration Config: [`backend/models/experimental/fusion_calibration_config.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/experimental/fusion_calibration_config.json)
- Test Suite: [`tests/test_calibration_and_expansion.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/tests/test_calibration_and_expansion.py)

---

## 1. Preflight Executive Summary

| Check ID | Verification Dimension | Verdict | Evidence / Summary |
| :---: | :--- | :---: | :--- |
| **CHK-01** | Manifest Counts, Paths, Hashes & Categories | **PASS** | 60 Val Videos, 60 Val Images; 200 Test Videos, 207 Test Images. 0 missing paths. SHA-256 for all items. |
| **CHK-02** | Still-Image Frame to Video Linkage | **PASS** | All 200 test video frames directly linked to parent video IDs; 7 standalone controls clearly distinguished. |
| **CHK-03** | Validation / Test Split Disjointness | **WARNING** | Paths, files, and FF++ sequence IDs are 100% disjoint (0 overlap). 4 Celeb-DF identity IDs appear in both val and test under official video-split protocol. |
| **CHK-04** | Logit Conversion, Temperature Scaling & Safe Guards | **PASS** | Logits converted via stable clipped logit formula; $T$ applied to logits; None/missing scores handled safely. |
| **CHK-05** | Frozen Parameters & Gating Rules | **PASS** | $T_{\text{SBI}}=2.7676, T_{\text{Xcep}}=2.4210, \tau_{\text{real}}=0.35, \tau_{\text{fake}}=0.65$ completely frozen in JSON/code before testing. |
| **CHK-06** | Explicit Metric Formulations & Abstention Handling | **PASS** | Binary and 3-state equations defined; abstentions explicitly excluded from True Positives / True Negatives. |
| **CHK-07** | Acceptance Criteria & Empirical Confidence Intervals | **PASS** | Criteria established as pre-registered targets; confidence intervals defined to be computed strictly on observed test results. |
| **CHK-08** | Model & Dataset Licensing Documentation | **WARNING** | Provenance documented from official papers/repos; standalone local `LICENSE.txt` files absent on disk. |
| **CHK-09** | Phase 1 Immutable Hash Verification & Tests | **PASS** | All 7 Phase 1 regression baselines match SHA-256 digests 100%; all automated unit tests pass. |

---

## 2. Detailed Preflight Verifications

### Check 1: Sample Counts, Labels, Paths, Hashes & Categories
- **Status**: **PASS**
- **Evidence**:
  - **Validation Manifest** ([`validation_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/validation_manifest_v2.json)):
    * Videos: **60 clips** (30 REAL: 15 Celeb-real, 15 FF++ original; 30 FAKE: 15 Celeb-synthesis, 3 Deepfakes, 3 Face2Face, 3 FaceSwap, 3 NeuralTextures, 3 FaceShifter).
    * Images: **60 images** (exact middle frames extracted from the 60 validation videos).
    * Missing local files: **0**.
    * Manifest SHA-256: `baf6f6615bc176341fd08cfa5e6fdc2797ba0e2c501876f2c04d40dcdb94d21a`.
  - **Locked Test Manifest** ([`locked_test_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/locked_test_manifest_v2.json)):
    * Videos: **200 clips** (100 REAL: 25 Celeb-real, 25 YouTube-real, 50 FF++ original; 100 FAKE: 50 Celeb-synthesis, 10 Deepfakes, 10 Face2Face, 10 FaceSwap, 10 NeuralTextures, 10 FaceShifter).
    * Images: **207 images** (103 REAL, 102 FAKE, 2 NON-FACE controls: 200 video-derived frames + 7 static in-the-wild controls).
    * Missing local files: **0**.
    * Manifest SHA-256: `16a1972f158d06f9009b647a2b08b29ee0aef01235175d8ce7291398b3084e55`.

---

### Check 2: Video-to-Frame Linkage & Shared Media Documentation
- **Status**: **PASS**
- **Evidence**:
  - Every image entry in `locked_test_manifest_v2.json` contains:
    * `id`: Canonical image identifier (e.g. `test_img_test_vid_celeb_real_000`).
    * `source_video_id`: Explicit pointer to parent video (e.g. `test_vid_celeb_real_000`).
    * `source_video_path`: Absolute disk path to parent video.
    * `frame_selection`: `exact_middle_frame` (or `static_image` for the 7 controls).
  - **Architectural Linkage Note**:
    * 200 still-image test cases are extracted as single middle frames from the 200 test video clips.
    * This allows direct, paired scientific measurement of the marginal gain of **multi-frame sampling (8 frames) + temporal consistency tracking** over **single-frame spatial detection** on the exact same media instances.

---

### Check 3: Validation / Test Split Disjointness & Identity Limits
- **Status**: **WARNING**
- **Evidence**:
  - **Physical Paths**: 0 overlap between validation and test sets (0 shared files).
  - **SHA-256 Digests**: 0 overlap between validation and test sets.
  - **FaceForensics++ Sequences**: 
    * `Train sequences (720)` $\cap$ `Val sequences (140)` = $\emptyset$ (0 overlap).
    * `Train sequences (720)` $\cap$ `Test sequences (140)` = $\emptyset$ (0 overlap).
    * `Val sequences (140)` $\cap$ `Test sequences (140)` = $\emptyset$ (0 overlap).
    * Zero sequence-level leakage exists in FaceForensics++.
  - **Celeb-DF Identity Overlap (Warning Source)**:
    * Celeb-DF contains 59 celebrity identities across 590 real and 5,639 synthetic videos.
    * The official benchmark protocol (Li et al., CVPR 2020) splits by video files (`List_of_testing_videos.txt`) rather than identity partitions.
    * Audit revealed that 4 celebrity identity IDs (`id0`, `id10`, `id16`, `id17`) have different video clips in validation and test.
    * *Mitigation*: All video files are distinct, but cross-clip identity repetition is an inherent constraint of the official Celeb-DF dataset structure.

---

### Check 4: Logit Conversion, Temperature Scaling & Safe Guards
- **Status**: **PASS**
- **Evidence**:
  - Inspected [`backend/agents/experimental/experimental_fusion_agent.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/agents/experimental/experimental_fusion_agent.py#L48-L60) and [`research/experimental_fusion/scripts/calibrate_fusion_models.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/scripts/calibrate_fusion_models.py#L20-L40):
    * Raw probabilities $p$ are safely clipped to $[\epsilon, 1 - \epsilon]$ with $\epsilon = 10^{-6}$ before logit conversion, preventing $\ln(0)$ or division-by-zero.
    * Logit calculation: $z = \ln(p / (1 - p))$.
    * Temperature scaling: $z_{\text{scaled}} = z / T$ where $T > 0$.
    * Calibrated probability: $p_{\text{cal}} = \sigma(z_{\text{scaled}}) = \frac{1}{1 + e^{-z_{\text{scaled}}}}$.
    * Missing/Invalid score handling: In `ForensicFusionEngine`:
      - If `score is None`: safely handled without exceptions.
      - If one model fails: graceful single-model fallback with reduced confidence.
      - If both models fail: safe abstention to `UNCERTAIN`.
      - Non-face media: immediately returns `NOT_SUITABLE` with `fake_probability = None`.

---

### Check 5: Frozen Parameters & Gating Rules
- **Status**: **PASS**
- **Evidence**:
  - Calibration parameters derived strictly from validation data are frozen in [`backend/models/experimental/fusion_calibration_config.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/experimental/fusion_calibration_config.json):
    * $T_{\text{SBI}} = 2.7676$
    * $T_{\text{Xception}} = 2.4210$
    * $\tau_{\text{real}} = 0.35$
    * $\tau_{\text{fake}} = 0.65$
    * $\text{min\_confidence} = 60.0\%$
  - Blending rule: Certainty-weighted probability $p_{\text{fused}} = \frac{w_{\text{SBI}} p_{\text{SBI}} + w_{\text{Xcep}} p_{\text{Xcep}}}{w_{\text{SBI}} + w_{\text{Xcep}}}$ where $w_m = \max(10^{-4}, 2|p_m - 0.50|)$.
  - Conservative Disagreement Rule: If $(p_{\text{SBI}} < 0.35 \land p_{\text{Xcep}} > 0.65) \lor (p_{\text{SBI}} > 0.65 \land p_{\text{Xcep}} < 0.35) \implies \text{UNCERTAIN}$.
  - Zero parameters were fit or tuned against `locked_test_manifest_v2.json`.

---

### Check 6: Explicit Metric Formulations & Abstention Handling
- **Status**: **PASS**
- **Evidence**:
  - **Forced Binary Metrics (Cutoff $\tau = 0.50$)**:
    * $TP = \sum [y_i = \text{FAKE} \land p_i \ge 0.50]$
    * $FN = \sum [y_i = \text{FAKE} \land p_i < 0.50]$
    * $TN = \sum [y_i = \text{REAL} \land p_i < 0.50]$
    * $FP = \sum [y_i = \text{REAL} \land p_i \ge 0.50]$
    * $\text{Accuracy} = (TP + TN) / N$
    * $\text{Precision} = TP / (TP + FP)$
    * $\text{Recall (TPR)} = TP / (TP + FN)$
    * $\text{Specificity (TNR)} = TN / (TN + FP)$
    * $\text{FPR} = FP / (TN + FP)$
    * $\text{FNR} = FN / (TP + FN)$
    * $\text{Macro-F1} = (\text{F1}_{\text{Fake}} + \text{F1}_{\text{Real}}) / 2$
  - **Three-State Safety Metrics**:
    * $\text{Three-State FPR} = \frac{FP_{3\text{st}}}{N_{\text{Real}}} = \frac{\sum [y_i = \text{REAL} \land V_i = \text{FAKE}]}{N_{\text{Real}}}$
    * $\text{Three-State FNR} = \frac{FN_{3\text{st}}}{N_{\text{Fake}}} = \frac{\sum [y_i = \text{FAKE} \land V_i = \text{LIKELY\_REAL}]}{N_{\text{Fake}}}$
    * $\text{Abstention Rate} = \frac{\sum [V_i = \text{UNCERTAIN}]}{N}$
    * $\text{Raw 3-State Accuracy} = \frac{TP_{3\text{st}} + TN_{3\text{st}}}{N}$ (abstentions counted as non-classifications, NOT correct)
    * $\text{Decided-Case Accuracy} = \frac{TP_{3\text{st}} + TN_{3\text{st}}}{TP_{3\text{st}} + TN_{3\text{st}} + FP_{3\text{st}} + FN_{3\text{st}}}$ (accuracy over non-abstained decisions)
  - Under these explicit mathematical definitions, abstentions are **never counted as correct classifications**.

---

### Check 7: Acceptance Criteria & Empirical Confidence Intervals
- **Status**: **PASS**
- **Evidence**:
  - Acceptance criteria are formally registered as **pre-test performance targets**, not guaranteed outcomes:
    * Three-State FPR: $\le 5.0\%$ (Max 5 false alarms on 100 Real videos; max 5 on 103 Real images).
    * Three-State FNR: $\le 5.0\%$ (Max 5 missed fakes on 100 Fake videos; max 5 on 102 Fake images).
    * Abstention Rate: $\le 35.0\%$.
    * Decided-Case Accuracy: $\ge 90.0\%$.
    * Forced Binary Accuracy: $\ge 80.0\%$ (Images), $\ge 88.0\%$ (Videos).
    * Macro-F1: $\ge 0.8000$ (Images), $\ge 0.8800$ (Videos).
    * Latency: $\le 1.20\text{s}$ (Image), $\le 6.50\text{s}$ (8-frame Video).
  - **Confidence Interval Protocol**:
    * Planning estimates will **not** be presented as guaranteed intervals.
    * Official 95% Wilson score confidence intervals will be computed strictly on the **actual observed test counts** ($k / n$) after execution.

---

### Check 8: Model & Dataset Licensing Documentation
- **Status**: **WARNING**
- **Evidence**:
  - **Academic Provenance Verified via Upstream Sources**:
    * SBI EfficientNet-B4: Yamasaki Lab (University of Tokyo), Academic / Non-Commercial Research License ([mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages)).
    * DeepfakeBench Xception: HKUST / Tencent, Academic Research License.
    * FaceForensics++ C23: TUM / Google, Non-commercial research agreement.
    * Celeb-DF v2: SUNY Albany, Academic research agreement.
  - **Local Limitation (Warning Source)**:
    * Standalone `LICENSE.txt` files are absent from `backend/models/experimental/weights/` and `D:\Download\SatyaDrishti_Datasets\`.
    * Upstream academic licenses prohibit commercial monetization without explicit written authorization. Preflight warning documented.

---

### Check 9: Phase 1 Immutable Hash Verification & Tests
- **Status**: **PASS**
- **Evidence**:
  - All 7 Phase 1 baseline artifacts match their recorded SHA-256 hashes 100%:
    * `scratch/image_eval_manifest.json`: `48413326aa5a185ac1096594a47353ec70da96899ac6109a89049c8941b8f58a` (**MATCH**)
    * `scratch/expanded_video_benchmark_results.json`: `73f426942e0cd9964be2d6635c436e773719afcc50c3a0e1f51f36f986c1e11f` (**MATCH**)
    * `scratch/full_video_sbi_benchmark_results.json`: `da3e55f005e3c9771bfa2563b3a7a7e9d0062c4d3d7b6c1bee255733b4a1764f` (**MATCH**)
    * `scratch/still_image_validation_results.json`: `2bf28b201e7ce2c2149fa9cf852e074da374cd4849ba0d75de36596dee09de98` (**MATCH**)
    * `research/experimental_fusion/fusion_benchmark_results.json`: `f8dd080435801edbea81cfa7c2b7c0a0a7a3ff6a77dee8366c924a1da972e5d7` (**MATCH**)
    * `research/experimental_fusion/FUSION_BENCHMARK_REPORT.md`: `dbd44777d506b8d3fb0b31f381986d87d59e44f5404da2e64d45acce3c03d13e` (**MATCH**)
    * `research/experimental_fusion/FUSION_AUDIT_REPORT.md`: `90eae8d177196acef778478c2a0b417367176b7166bf20c7514212521b8e166e` (**MATCH**)
  - Automated test suite execution:
    * [`tests/test_calibration_and_expansion.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/tests/test_calibration_and_expansion.py): **Ran 4 tests in 0.039s. OK.**
    * [`tests/test_experimental_fusion.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/tests/test_calibration_and_expansion.py): **Ran 8 tests in 0.006s. OK.**

---

## 3. Preflight Conclusion

Preflight verification is complete:
- **7 CHECKS PASSED**
- **2 WARNINGS NOTED & MITIGATED** (Celeb-DF identity repetition across video clips; absence of local standalone license text files).
- **0 FAILURES**

The locked test set ([`locked_test_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/locked_test_manifest_v2.json)) and Phase 1 artifacts remain sealed. Stopping for final stakeholder review prior to executing the Phase 2 expanded benchmark.
