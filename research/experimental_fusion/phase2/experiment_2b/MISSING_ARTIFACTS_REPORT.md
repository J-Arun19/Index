# SatyaDrishti Experiment 2B: Validation Artifact Verification & Stop Report

**Audit Date**: 2026-10-12  
**Investigation Scope**: Validation Artifact Inspection for Decision-Policy Trade-Off Analysis  
**Protocol Directives Enforced**:
- *Task 2 Directive*: "Verify that sufficient sample-level validation predictions exist to evaluate alternative decision policies. **If they do not, stop and report the missing artifacts.**"
- *Constraint 1*: "Do not train or fine-tune any models."
- *Constraint 2*: "Do not run GPU inference or download datasets."
- *Constraint 5*: "Treat the Phase 2 test results as exploratory and do not optimize a policy against those test labels."
- *Task 5 Directive*: "Freeze candidate policies and their thresholds **using validation data only**. Do not select thresholds by maximizing results on Phase 2 test data."
**Execution Status**: **STOPPED AT TASK 2 CHECKPOINT DUE TO MISSING VALIDATION SAMPLE PREDICTIONS.**

---

## 1. Inventory & Inspection of Existing Artifacts

In compliance with Task 1, all existing calibration, validation, and benchmark records were inspected:

| Artifact Path | Artifact Type | Sample Count | Contents & Scope | Usable for Validation Policy Selection? |
| :--- | :--- | :---: | :--- | :---: |
| [`research/experimental_fusion/datasets/validation_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/validation_manifest_v2.json) | Dataset Manifest | 60 vids, 60 imgs | Disjoint validation file paths, labels, and SHA-256 digests. | **NO** (Manifest metadata only; zero model predictions). |
| [`backend/models/experimental/fusion_calibration_config.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/backend/models/experimental/fusion_calibration_config.json) | Calibration Config | Summary (N=60) | Scalar temperatures ($T_{\text{SBI}}=2.7676, T_{\text{Xcep}}=2.4210$) and ECE before/after. | **NO** (Aggregated parameters only; zero per-sample scores). |
| `scratch/temp_val/` | Raw Media Frames | 60 JPEG frames | Extracted middle frames (`val_frame_001.jpg` to `val_frame_060.jpg`). | **NO** (Image files; no prediction scores). |
| `task-658.log` | Execution Log | Terminal log | Log from `calibrate_fusion_models.py`; printed 6 progress lines. | **NO** (No structured prediction records). |
| [`scratch/still_image_validation_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/scratch/still_image_validation_results.json) | Legacy Phase 1 Predictions | 55 still images | Phase 1 preliminary image set; evaluated under legacy calibration ($T=1.885$). | **NO** (Image only; does not match Phase 2 split; zero video predictions). |
| [`research/experimental_fusion/phase2/phase2_benchmark_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json) | Phase 2 Benchmark Results | 200 vids, 207 imgs | Full per-sample predictions on the **LOCKED TEST SPLIT**. | **STRICTLY BARRED** (Using this to tune or freeze policies violates Task 5 & Constraint 5). |
| [`research/experimental_fusion/phase2/experiment_2a/experiment_2a_simulation_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/experiment_2a/experiment_2a_simulation_results.json) | Exp 2A Offline Simulation | 200 vids, 205 imgs | Simulation on the **LOCKED TEST SPLIT**. | **STRICTLY BARRED** (Test data). |

---

## 2. Verification Finding: Missing Validation Artifacts

In compliance with Task 2:
> **"Verify that sufficient sample-level validation predictions exist to evaluate alternative decision policies. If they do not, stop and report the missing artifacts."**

### Verification Result: **FAIL — INSUFFICIENT VALIDATION PREDICTIONS EXIST ON DISK**

The following critical sample-level artifacts are missing:
1. **Missing Artifact 1: `validation_video_predictions.json`**:
   - The 60 validation videos specified in [`validation_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/validation_manifest_v2.json) have **zero persisted prediction records**.
   - During Phase 2 calibration, the calibration script (`calibrate_fusion_models.py`) only evaluated single middle frames to fit temperature scaling. Multi-frame video evaluation (8-frame sampling, variance calculation, and temporal conflict detection) was never executed on the validation split.
   - Consequently, there are zero validation data points for temporal variance, inter-frame conflict, or multi-frame consensus.
2. **Missing Artifact 2: `validation_image_predictions.json`**:
   - Although `calibrate_fusion_models.py` evaluated the 60 validation middle frames in memory to minimize scalar negative log-likelihood, it persisted **only** the fitted scalars ($T_{\text{SBI}}=2.7676, T_{\text{Xcep}}=2.4210$) to `fusion_calibration_config.json`.
   - The individual per-sample raw model scores ($p_{\text{SBI}}, p_{\text{Xcep}}$) and forensic signals for the 60 validation images were not saved to a JSON artifact.

---

## 3. Methodological Rationale for Stopping

Under the pre-registered scientific protocol:
1. **Task 5 mandates**: *"Freeze candidate policies and their thresholds using validation data only. Do not select thresholds by maximizing results on Phase 2 test data."*
2. **Constraint 5 mandates**: *"Treat the Phase 2 test results as exploratory and do not optimize a policy against those test labels."*
3. **Constraint 2 mandates**: *"Do not run GPU inference or download datasets."*

If the benchmark runner were to bypass the validation requirement and select, tune, or freeze candidate policies using [`phase2_benchmark_results.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/phase2/phase2_benchmark_results.json), it would commit **data snooping / target leakage**, conflating model evaluation with post-hoc parameter fitting.

Therefore, execution has been halted immediately at the Task 2 checkpoint, strictly adhering to the prompt directive.

---

## 4. Pre-Defined Candidate Policies (Ready for Validation Data)

As requested in Tasks 3 and 4, the following candidate policies have been conceptually formulated and are ready to be frozen and evaluated the moment validation sample predictions become available:

### Candidate 1: Baseline Reference (Frozen Phase 2 Policy)
- **Real Certification**: $p_{\text{fused}} < 0.35$ (subject to texture gate).
- **Fake Decision**: $p_{\text{fused}} > 0.65$.
- **Disagreement / Margin**: Disagreement or $0.35 \le p_{\text{fused}} \le 0.65 \implies \text{UNCERTAIN}$.
- **Temporal Variance Gate**: Variance $> 0.22 \implies \text{UNCERTAIN}$.

### Candidate 2: One-Sided REAL Certification (Reference from Exp 2A)
- **Real Certification**: $(p_{\text{fused}} < 0.35) \land (\text{cal\_sbi} < 0.35) \land (\text{cal\_xcep} < 0.35)$.
- **Fake Decision**: $p_{\text{fused}} > 0.65$.
- **Disagreement / Margin**: Unchanged.
- **Hypothesis**: Eliminates FaceShifter / NeuralTextures false negatives by withholding authenticity when Xception is neutral.

### Candidate 3: Directional Temporal Conflict Gating
- **Rationale**: Frame-to-frame boundary seam flicker is a physical characteristic of face swaps (FaceSwap, Deepfakes). In the baseline, variance $> 0.22$ forces `UNCERTAIN` symmetrically.
- **Rule**: If mean probability leans fake ($p_{\text{fused}} > 0.50$), inter-frame variance is treated as corroborating manipulation flicker rather than uncertainty, allowing decisive `FAKE` verdicts. If $p_{\text{fused}} \le 0.50$, variance $> 0.22$ forces `UNCERTAIN`.
- **Hypothesis**: Recovers 20–30 manipulated videos from abstention, reducing the 74% video abstention rate to ~55% without inflating false alarms.

### Candidate 4: Asymmetric Dual-Gate Consensus
- **Rationale**: Rebalances the decision margins based on asymmetric operational risk:
  * False Positive Risk (branding/legal damage) requires strict unanimous proof: $\text{cal\_sbi} < 0.35 \land \text{cal\_xcep} < 0.40$.
  * False Negative Risk (synthetic media dissemination) allows decisive action if fused score is moderately elevated: $p_{\text{fused}} > 0.60$.
  * Indeterminate zone narrows from $[0.35, 0.65]$ to $[0.40, 0.60]$.

---

## 5. Path to Unblocking Experiment 2B

To proceed with Experiment 2B without violating scientific integrity or data leakage constraints, the following two-step workflow is proposed for user authorization:

1. **Step 1 (Approved Inference on Validation Split Only)**:
   - Run an isolated, sequential inference runner on the **60 validation videos** and **60 validation images** in [`validation_manifest_v2.json`](file:///d:/Download/satya-vision-guard-main/satya-vision-guard-main/research/experimental_fusion/datasets/validation_manifest_v2.json) using the existing frozen checkpoints.
   - Save the sample-level telemetry directly to `research/experimental_fusion/phase2/experiment_2b/validation_sample_predictions.json` (runtime: ~60 seconds on GTX 1650).
   - Zero test data is touched; zero models are trained.
2. **Step 2 (Validation-Driven Policy Selection)**:
   - Compare Candidates 1–4 strictly on `validation_sample_predictions.json`.
   - Select and freeze the winning policy based on the validation safety/coverage Pareto frontier.
   - Evaluate the frozen policy against the Phase 2 test set as a clearly labeled exploratory audit.

---

## 6. Conclusion

In strict adherence to Task 2, execution is stopped. No test data was snooped or fitted, no GPU inference was run, and no production code was modified. Awaiting user guidance on whether to generate the missing validation prediction artifact.
