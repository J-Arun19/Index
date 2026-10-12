# SatyaDrishti Experiment 2A: Offline Decision-Policy Simulation Report

**Experiment Date**: 2026-10-12  
**Evaluation Nature**: Offline Decision-Policy Simulation (Zero GPU Compute, Zero Retraining)  
**Baseline Dataset**: Phase 2 Locked Benchmark Predictions ([`research/experimental_fusion/phase2/phase2_benchmark_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json))  
**Simulation Output**: [`research/experimental_fusion/phase2/experiment_2a/experiment_2a_simulation_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/experiment_2a/experiment_2a_simulation_results.json)  
**Test Suite**: [`tests/test_experiment_2a_policy.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/tests/test_experiment_2a_policy.py) (5 tests passing in 0.002s)  
**Production Boundary**: **Simulation Only. Zero Code, API, Weight, or Threshold Alteration to Production.**  

---

## 1. Executive Summary

Experiment 2A investigates a non-training, decision-policy intervention to eliminate the **four video false negatives** (three FaceShifter clips and one NeuralTextures clip) uncovered during the Phase 2 Locked Benchmark. 

In the Phase 2 baseline, these four manipulated clips were falsely certified as `LIKELY_REAL` because the Self-Blended Images (SBI) detector—blind to seamless generative face swaps—produced an extremely low fake score ($p_{\text{SBI}} \le 0.098$). Under certainty weighting, this extreme score overpowered the neutral Xception detector ($p_{\text{Xcep}} \approx 0.43 - 0.50$), pulling the blended probability below $\tau_{\text{real}} = 0.35$ while bypassing the conservative disagreement rule (which only triggers when Xception is $> 0.65$).

To remedy this structural asymmetry without retraining or adjusting calibration temperatures, Experiment 2A introduces the **One-Sided REAL Certification Policy**:
> **Rule**: The system may issue `LIKELY_REAL` **only when both individual calibrated model probabilities are below 0.35** ($\text{cal\_sbi} < 0.35 \land \text{cal\_xcep} < 0.35$). If either model is neutral ($0.35 \le p \le 0.65$) or leaning fake, the system withholds authentic certification and defaults to **`UNCERTAIN`**. All `FAKE` decision logic and conservative disagreement safeguards are preserved unchanged.

### Headline Findings:
1. **100% of Video False Negatives Eliminated**: Video Three-State False Negative Rate dropped from **4.00% to 0.00%** (0 missed fakes out of 100 Fake clips).
2. **100% of Image False Negatives Eliminated**: Still-Image Three-State False Negative Rate dropped from **1.96% to 0.00%** (0 missed fakes out of 102 Fake images).
3. **Zero False-Alarm Inflation**: Three-State False Positive Rate remained completely unchanged (**1.00% on videos**, **7.77% on images**).
4. **Decided-Case Accuracy Rose to 96.77% on Video**: When the policy issues a definitive decision on video, it is **96.77% accurate** (30 correct out of 31 decided cases), up from 90.38%.
5. **The Operational Trade-off**: Enforcing unanimous agreement converted 17 Real videos from `LIKELY_REAL` into `UNCERTAIN`, increasing the video abstention rate from **74.00% to 84.50%** (+10.50% trade-off) and reducing video decided coverage from 26.00% down to 15.50%.

---

## 2. Decision Policy Formulation

The alternative decision policy modifies **only** the condition required to certify authenticity:

```
Baseline Policy (Phase 2):
  If p_fused < 0.35 (and texture gates pass):
      -> LIKELY_REAL  (Even if Xception is neutral, e.g. p_xcep = 0.48)

Experiment 2A (One-Sided REAL Certification Policy):
  If p_fused < 0.35:
      If (cal_sbi < 0.35) AND (cal_xcep < 0.35):
          -> LIKELY_REAL  (Unanimous authentic consensus)
      Else:
          -> UNCERTAIN    (Withholding authentic certification)
```

All other branches remain identical:
- Media suitability checks (`NOT_SUITABLE` for non-face media) are preserved.
- Temporal consistency checks (conflict / variance $> 0.22 \implies \text{UNCERTAIN}$) are preserved.
- Direct model disagreement checks ($\text{SBI} < 0.35 \land \text{Xcep} > 0.65 \implies \text{UNCERTAIN}$) are preserved.
- Ambiguous margin checks ($0.35 \le p_{\text{fused}} \le 0.65 \implies \text{UNCERTAIN}$) are preserved.
- High-confidence FAKE decisions ($p_{\text{fused}} > 0.65 \implies \text{FAKE}$) are preserved.

---

## 3. Video Benchmark Comparison (200 Clips)

Evaluated across the exact same **100 REAL** and **100 FAKE** video clips:

| Metric Dimension | Phase 2 Baseline Policy | Experiment 2A Policy | Absolute Delta | Wilson 95% CI (Exp 2A) |
| :--- | :---: | :---: | :---: | :---: |
| **Real: Likely Real (TN)** | 21 | 4 | -17 | [1.57%, 9.84%] |
| **Real: Uncertain (Abstained)** | 78 | 95 | +17 | [88.58%, 97.94%] |
| **Real: Fake (FP)** | 1 | 1 | 0 | [0.18%, 5.45%] |
| **Fake: Likely Real (FN)** | **4** | **0** | **-4 (100% eliminated)** | **[0.00%, 3.62%]** |
| **Fake: Uncertain (Abstained)** | 70 | 74 | +4 | [64.57%, 81.65%] |
| **Fake: Fake (TP)** | 26 | 26 | 0 | [18.35%, 35.43%] |
| **Three-State FPR** | **1.00%** (1 / 100) | **1.00%** (1 / 100) | **0.00% (No change)** | [0.18%, 5.45%] |
| **Three-State FNR** | **4.00%** (4 / 100) | **0.00%** (0 / 100) | **-4.00% (Zero FNs)** | **[0.00%, 3.62%]** |
| **Overall Abstention Rate** | 74.00% (148 / 200) | 84.50% (169 / 200) | +10.50% trade-off | [78.84%, 88.86%] |
| **Decided Cases Count** | 52 clips | 31 clips | -21 clips | [11.14%, 21.16%] |
| **Decided-Case Accuracy** | 90.38% (47 / 52) | **96.77%** (30 / 31) | **+6.39% gain** | **[83.81%, 99.43%]** |
| **Decided Coverage** | 26.00% (52 / 200) | 15.50% (31 / 200) | -10.50% trade-off | [11.14%, 21.16%] |
| **Raw Three-State Accuracy** | 23.50% (47 / 200) | 15.00% (30 / 200) | -8.50% | [10.72%, 20.59%] |

---

## 4. Still-Image Benchmark Comparison (205 Facial Items)

Evaluated across the exact same **103 REAL** and **102 FAKE** facial images:

| Metric Dimension | Phase 2 Baseline Policy | Experiment 2A Policy | Absolute Delta | Wilson 95% CI (Exp 2A) |
| :--- | :---: | :---: | :---: | :---: |
| **Real: Likely Real (TN)** | 16 | 2 | -14 | [0.53%, 6.81%] |
| **Real: Uncertain (Abstained)** | 77 | 91 | +14 | [80.79%, 93.30%] |
| **Real: Fake (FP)** | 8 | 8 | 0 | [3.99%, 14.58%] |
| **Fake: Likely Real (FN)** | **2** | **0** | **-2 (100% eliminated)** | **[0.00%, 3.55%]** |
| **Fake: Uncertain (Abstained)** | 52 | 54 | +2 | [43.34%, 62.33%] |
| **Fake: Fake (TP)** | 47 | 47 | 0 | [36.70%, 55.70%] |
| **Three-State FPR** | **7.77%** (8 / 103) | **7.77%** (8 / 103) | **0.00% (No change)** | [3.99%, 14.58%] |
| **Three-State FNR** | **1.96%** (2 / 102) | **0.00%** (0 / 102) | **-1.96% (Zero FNs)** | **[0.00%, 3.55%]** |
| **Overall Abstention Rate** | 62.93% (129 / 205) | 69.76% (143 / 205) | +6.83% trade-off | [63.15%, 75.64%] |
| **Decided Cases Count** | 73 images | 59 images | -14 images | [22.99%, 35.39%] |
| **Decided-Case Accuracy** | 86.30% (63 / 73) | **86.44%** (51 / 59) | **+0.14% gain** | **[75.51%, 93.00%]** |
| **Decided Coverage** | 35.61% (73 / 205) | 28.78% (59 / 205) | -6.83% trade-off | [22.99%, 35.39%] |
| **Raw Three-State Accuracy** | 30.73% (63 / 205) | 24.88% (51 / 205) | -5.85% | [19.46%, 31.24%] |

---

## 5. Direct Audit of the Four Video False Negatives

The four previously reported video false negatives were tracked to confirm their resolution under Experiment 2A:

| Sample ID | Filename | Manipulation Category | Ground Truth | Baseline Verdict | Exp 2A Verdict | Fused P | SBI Calibrated | Xception Calibrated | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`test_vid_ff_neuraltextures_180`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json) | `494_445.mp4` | NeuralTextures | **FAKE** | `LIKELY_REAL` (FN) | **`UNCERTAIN`** | 0.3038 | 0.2868 (< 0.35) | **0.4954 ($\ge 0.35$)** | **RESOLVED** |
| [`test_vid_ff_faceshifter_191`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json) | `257_420.mp4` | FaceShifter | **FAKE** | `LIKELY_REAL` (FN) | **`UNCERTAIN`** | 0.3187 | 0.2866 (< 0.35) | **0.4746 ($\ge 0.35$)** | **RESOLVED** |
| [`test_vid_ff_faceshifter_194`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json) | `732_691.mp4` | FaceShifter | **FAKE** | `LIKELY_REAL` (FN) | **`UNCERTAIN`** | 0.2598 | 0.2227 (< 0.35) | **0.4710 ($\ge 0.35$)** | **RESOLVED** |
| [`test_vid_ff_faceshifter_198`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json) | `625_650.mp4` | FaceShifter | **FAKE** | `LIKELY_REAL` (FN) | **`UNCERTAIN`** | 0.3105 | 0.2974 (< 0.35) | **0.5009 ($\ge 0.35$)** | **RESOLVED** |

### Audit Finding:
In all four cases, the One-Sided Real Certification Policy caught the discrepancy: Xception was neutral ($p \approx 0.47 - 0.50$), failing the unanimous $< 0.35$ condition. The policy withheld authentic certification, converting all four false negatives into `UNCERTAIN` abstentions.

---

## 6. Manipulation Category Breakdown (Video)

Comparison of Baseline vs Experiment 2A across each video category:

| Manipulation Category | Ground Truth | Total Clips | Baseline LIKELY_REAL | Exp 2A LIKELY_REAL | Baseline UNCERTAIN | Exp 2A UNCERTAIN | Baseline FAKE | Exp 2A FAKE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Celeb-real** | REAL | 25 | 4 | **1** | 21 | **24** | 0 | 0 |
| **YouTube-real** | REAL | 25 | 2 | **1** | 22 | **23** | 1 | 1 |
| **FF++ original** | REAL | 50 | 15 | **2** | 35 | **48** | 0 | 0 |
| **Celeb-synthesis** | FAKE | 50 | 0 | **0** | 34 | **34** | 16 | 16 |
| **Deepfakes** | FAKE | 10 | 0 | **0** | 3 | **3** | 7 | 7 |
| **Face2Face** | FAKE | 10 | 0 | **0** | 9 | **9** | 1 | 1 |
| **FaceSwap** | FAKE | 10 | 0 | **0** | 9 | **9** | 1 | 1 |
| **NeuralTextures** | FAKE | 10 | **1 (FN)** | **0 (0% FN)** | 8 | **9** | 1 | 1 |
| **FaceShifter** | FAKE | 10 | **3 (FN)** | **0 (0% FN)** | 7 | **10** | 0 | 0 |

---

## 7. Paired Decision Changes (All 21 Video Modifications)

Under the alternative policy, exactly **21 out of 200 video decisions changed**. Every single modification was a transition from `LIKELY_REAL` to `UNCERTAIN`:

- **Beneficial Changes (4 Fakes converted to UNCERTAIN)**:
  * NeuralTextures: `494_445.mp4` (eliminated false negative)
  * FaceShifter: `257_420.mp4` (eliminated false negative)
  * FaceShifter: `732_691.mp4` (eliminated false negative)
  * FaceShifter: `625_650.mp4` (eliminated false negative)
- **Trade-Off Changes (17 Reals converted to UNCERTAIN)**:
  * Celeb-real (3 clips): `id1_0000.mp4`, `id3_0009.mp4`, `id6_0003.mp4`
  * YouTube-real (1 clip): `00155.mp4`
  * FF++ original (13 clips): `003.mp4`, `006.mp4`, `035.mp4`, `169.mp4`, `214.mp4`, `249.mp4`, `420.mp4`, `445.mp4`, `452.mp4`, `550.mp4`, `650.mp4`, `714.mp4`, `842.mp4`
  * In each of these 17 Real clips, SBI was confident ($p < 0.35$), but Xception was slightly neutral ($p \approx 0.36 - 0.48$). Because Xception did not confirm authentic representation, the strict policy abstained.

---

## 8. Unit Test Suite Verification

A dedicated unit test suite ([`tests/test_experiment_2a_policy.py`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/tests/test_experiment_2a_policy.py)) was developed to verify policy edge cases:
- `test_boundary_values`: Confirmed that $p = 0.349$ allows `LIKELY_REAL`, while $p = 0.350$ triggers `UNCERTAIN`. Tested FaceShifter asymmetric score patterns.
- `test_missing_scores`: Confirmed safe fallback handling when one score is `None`.
- `test_model_disagreement_and_fake_preservation`: Confirmed that `FAKE` decisions ($P > 0.65$) and model disagreement abstentions are 100% preserved.
- `test_non_face_inputs_preserved`: Confirmed that `NOT_SUITABLE` verdicts remain completely untouched.
- `test_simulation_reproducibility`: Confirmed exact numerical reproducibility of all metrics against the saved Phase 2 JSON.
- **Result**: **5 tests ran in 0.002s. OK.**

---

## 9. Limitations & Non-Validation Notice

> [!WARNING]
> **Scientific Integrity Caveat**:
> This offline simulation evaluates an alternative hypothesis on the **same locked test split** that originally exposed the FaceShifter false negatives. Consequently, this simulation **must not be claimed as independent empirical validation**. Independent validation would require evaluating the One-Sided Real Certification Policy on an entirely new, held-out dataset.

---

## 10. Conclusion & Final Stance

Experiment 2A confirms the theoretical hypothesis:
1. **The One-Sided Real Certification Policy is 100% effective at eliminating false negatives** on seamless swaps (FaceShifter) and reenactments (NeuralTextures), dropping Three-State FNR from **4.00% to 0.00%** on videos and **1.96% to 0.00%** on images.
2. **False alarms did not increase**: Three-State FPR remained frozen at **1.00% on video** and **7.77% on images**.
3. **The trade-off is substantial**: Video abstention increased from **74.00% to 84.50%**, reducing operational coverage to 15.50%.

**Final Recommendation**:
In accordance with user instructions, this simulation **remains an offline research investigation**. The policy will **NOT** be integrated into production files, thresholds, or APIs.
