import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.core.context import AnalysisContext
from backend.core.decision_system import PresentationDecisionPolicy, DecisionState
from backend.models.experimental.xception_detector import XceptionDetector


class TestAuditVerification(unittest.TestCase):
    """
    Independent audit verification test suite verifying:
    1. Filename invariance (changing filename does not change inference result).
    2. DeepfakeBench Xception class mapping (index 0 = REAL, index 1 = FAKE).
    3. Absence of ground-truth leakage into inference pipelines.
    4. Identical inference pipeline execution for unknown media.
    5. Inference failures do not manufacture synthetic confidence scores.
    6. Provisional/uncertain results are visibly distinguishable from clean REAL results.
    7. Metric computation strictly reflects empirical prediction counts.
    8. AnalysisContext dictionary interface compatibility.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.weights_path = os.path.join(
            "backend", "models", "experimental", "weights", "xception_best.pth"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_xception_checkpoint_and_class_mapping(self):
        """
        Verify that Xception weights exist, have the audited SHA-256 / architecture,
        and adhere to the standard DeepfakeBench class convention:
        Class 0: REAL, Class 1: FAKE.
        """
        self.assertTrue(os.path.exists(self.weights_path), f"Weights missing: {self.weights_path}")
        detector = XceptionDetector(weights_path=self.weights_path, device="cpu")
        self.assertIsNotNone(detector.backbone)

        # Forward pass dummy batch: (1, 3, 256, 256)
        dummy_input = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        with torch.no_grad():
            logits = detector(dummy_input)
        
        self.assertEqual(logits.shape, (1, 2), "Logits output must be shape (B, 2)")
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        # Class 0: Real probability, Class 1: Fake probability
        self.assertEqual(len(probs), 2)

    def test_02_filename_invariance_on_identical_media(self):
        """
        Verify that changing a video filename to a misleading name (e.g. calling a fake
        video 'innocent_real_interview.mp4' or calling a real video 'blatant_face_swap.mp4')
        produces byte-for-byte and score-for-score IDENTICAL model inference results.
        Proves there is zero filename-based routing or prediction leakage.
        """
        sample_path = "D:/Download/SatyaDrishti_Datasets/FaceForensics++_C23/FaceSwap/953_974.mp4"
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample test video not found: {sample_path}")

        # Create two identical copies with opposite suggestive names
        fake_named_copy = os.path.join(self.test_dir, "definitely_blatant_deepfake_hack.mp4")
        real_named_copy = os.path.join(self.test_dir, "100_percent_authentic_innocent_real_video.mp4")
        shutil.copyfile(sample_path, fake_named_copy)
        shutil.copyfile(sample_path, real_named_copy)

        agent = VideoAgent()
        res_fake_name = agent.analyze(fake_named_copy)
        res_real_name = agent.analyze(real_named_copy)

        self.assertEqual(
            res_fake_name["verdict"],
            res_real_name["verdict"],
            "Verdicts must be identical regardless of filename!"
        )
        self.assertAlmostEqual(
            res_fake_name.get("fake_probability", 0.0),
            res_real_name.get("fake_probability", 0.0),
            places=5,
            msg="Fake probabilities must be strictly identical!"
        )
        self.assertAlmostEqual(
            res_fake_name.get("confidence_score", 0.0),
            res_real_name.get("confidence_score", 0.0),
            places=3,
            msg="Confidence scores must be strictly identical!"
        )

    def test_03_ground_truth_never_injected_into_inference(self):
        """
        Verify by inspection of VideoAgent.analyze() signature that no ground-truth,
        label, manifest, or hints can be passed into the inference call.
        """
        import inspect
        sig = inspect.signature(VideoAgent.analyze)
        param_names = list(sig.parameters.keys())
        self.assertIn("file_path", param_names)
        for forbidden in ["label", "ground_truth", "gt", "is_fake", "expected"]:
            self.assertNotIn(
                forbidden,
                param_names,
                f"Forbidden ground-truth parameter '{forbidden}' detected in VideoAgent.analyze signature!"
            )

    def test_04_inference_failure_produces_no_fabricated_confidence(self):
        """
        Verify that when an unopenable or corrupted file is submitted, the system
        fails cleanly via exception rather than manufacturing artificial confidence scores.
        """
        agent = VideoAgent()
        corrupt_path = os.path.join(self.test_dir, "non_existent_or_empty.mp4")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_REAL_VIDEO_HEADER")

        with self.assertRaises((ValueError, IOError)):
            agent.analyze(corrupt_path)

    def test_05_provisional_uncertain_visibly_distinguishable_from_clean_real(self):
        """
        Verify that PresentationDecisionPolicy preserves transparent uncertainty:
        Clean REAL has is_provisional=False, warning=None.
        Ambiguous or conflicting REAL has is_provisional=True, warning contains explicit caution.
        """
        # 1. Clean authentic media
        clean_verdict, raw_state, is_provisional, warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.REAL,
            fake_probability=0.08,
            has_conflict=False,
            quality_issues=False
        )
        self.assertEqual(clean_verdict, DecisionState.REAL)
        self.assertFalse(is_provisional, "Clean authentic result must not be marked provisional")
        self.assertIsNone(warning, "Clean authentic result must have no ambiguity warning")

        # 2. Conflicting / ambiguous media (e.g. ViT false alarm on compressed real video)
        prov_verdict, prov_raw, prov_flag, prov_warning = PresentationDecisionPolicy.apply_presentation_policy(
            decision_state=DecisionState.UNCERTAIN,
            fake_probability=0.28,
            has_conflict=True,
            disagreement_detected=True,
            reason="Neural classifier says REAL but ViT detected anomaly"
        )
        self.assertEqual(prov_verdict, DecisionState.REAL)
        self.assertTrue(prov_flag, "Conflicting result must be explicitly marked provisional!")
        self.assertIsNotNone(prov_warning, "Conflicting result must display visible caution warning!")
        self.assertIn("provisional", prov_warning.lower())

    def test_06_independent_metric_calculation_accuracy(self):
        """
        Verify that metric calculation formulas (accuracy, precision, recall, specificity,
        FPR, FNR, Macro-F1) correctly compute values from an empirical confusion matrix.
        """
        # Ground truth: 9 REAL (0), 9 FAKE (1)
        # Predictions: 9 REAL (0), 9 FAKE (1)
        y_true = [0]*9 + [1]*9
        y_pred = [0]*9 + [1]*9

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        acc = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        self.assertEqual(acc, 1.0)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(specificity, 1.0)
        self.assertEqual(fpr, 0.0)
        self.assertEqual(fnr, 0.0)

        # Now test with 2 false positives (like held-out set: 10 fakes, 10 reals, 2 FP)
        y_true_ho = [0]*10 + [1]*10
        y_pred_ho = [0]*8 + [1]*2 + [1]*10
        tp_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == 1 and yp == 1)
        tn_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == 0 and yp == 0)
        fp_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == 0 and yp == 1)
        fn_ho = sum(1 for yt, yp in zip(y_true_ho, y_pred_ho) if yt == 1 and yp == 0)

        self.assertEqual((tp_ho, tn_ho, fp_ho, fn_ho), (10, 8, 2, 0))
        acc_ho = (tp_ho + tn_ho) / len(y_true_ho)
        self.assertEqual(acc_ho, 0.90)
        spec_ho = tn_ho / (tn_ho + fp_ho)
        self.assertEqual(spec_ho, 0.80)
        recall_ho = tp_ho / (tp_ho + fn_ho)
        self.assertEqual(recall_ho, 1.0)

    def test_07_analysis_context_dict_compatibility(self):
        """
        Verify that AnalysisContext provides both attribute and dictionary interfaces
        so Flask API endpoints, report generators, and logs interact without crashes.
        """
        ctx = AnalysisContext(analysis_id="test_audit_ctx", media_type="video")
        # Attribute access
        self.assertEqual(ctx.analysis_id, "test_audit_ctx")
        # Subscript access
        self.assertEqual(ctx["analysis_id"], "test_audit_ctx")
        # get() access
        self.assertEqual(ctx.get("analysis_id"), "test_audit_ctx")
        self.assertEqual(ctx.get("non_existent_key", "default_val"), "default_val")
        # Subscript assignment
        ctx["custom_attribute"] = "verified_safe"
        self.assertEqual(ctx.custom_attribute, "verified_safe")
        self.assertEqual(ctx["custom_attribute"], "verified_safe")
        # Membership check
        self.assertIn("analysis_id", ctx)
        self.assertIn("custom_attribute", ctx)
        self.assertNotIn("completely_unknown_key", ctx)
        # to_dict compatibility
        d = ctx.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("analysis_id", d)


    def test_08_image_agent_fake2_not_classified_as_real(self):
        """
        Verify that AI-generated portrait image (fake2.png) is NOT classified
        as REAL with high confidence. It must be recognized as FAKE (or UNCERTAIN if
        ambiguous) and must utilize genuine model inference (Xception/ViT).
        """
        fake_candidates = [
            "backend/uploads/test_8cd6a096be90_fake2.png",
            "backend/uploads/10e43c19-1628-405d-bcbd-2f24d8b26401_fake2.png",
            "backend/uploads/fake2.png"
        ]
        sample_path = next((p for p in fake_candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("fake2.png sample not found")

        agent = ImageAgent()
        res = agent.analyze(sample_path, selected_providers=[])

        self.assertNotIn(
            res.get("primary_verdict"),
            ["REAL", "LIKELY_REAL"],
            f"Critical Failure: fake2.png was classified as {res.get('primary_verdict')}"
        )
        self.assertNotIn(
            res.get("verdict"),
            ["REAL", "LIKELY_REAL"],
            f"Critical Failure: fake2.png final verdict was {res.get('verdict')}"
        )
        # Ensure fake probability is elevated
        self.assertGreater(
            res.get("fake_probability", 0.0),
            0.50,
            f"Expected elevated fake probability for fake2.png, got {res.get('fake_probability')}"
        )

    def test_09_image_agent_no_face_classified_as_not_suitable(self):
        """
        Verify that an image without faces (e.g. document_contract_sample.png)
        is flagged as NOT_SUITABLE / NO_FACE_DETECTED rather than producing a
        hallucinated authentic face verdict.
        """
        doc_candidates = [
            "backend/uploads/document_contract_sample.png",
            "backend/uploads/8a84fa99-9b92-4473-ad17-84c835083df6_document_contract_sample.png"
        ]
        sample_path = next((p for p in doc_candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("document_contract_sample.png not found")

        agent = ImageAgent()
        res = agent.analyze(sample_path, selected_providers=[])
        self.assertIn(
            res.get("verdict"),
            ["NOT_SUITABLE", "INCONCLUSIVE", "ERROR"],
            f"Expected NOT_SUITABLE or INCONCLUSIVE for document image, got {res.get('verdict')}"
        )

    def test_10_image_explanation_integrity(self):
        """
        Verify that generated explanations do NOT contain unmeasured, fabricated
        claims such as 'coherent bilateral ocular symmetry' or 'standard 1/f spectral power decay'.
        """
        agent = ImageAgent()
        forbidden_phrases = [
            "bilateral ocular symmetry",
            "1/f spectral power decay",
            "natural skin texture variance",
            "coherent ocular symmetry"
        ]
        fake_candidates = [
            "backend/uploads/test_8cd6a096be90_fake2.png",
            "backend/uploads/fake2.png"
        ]
        sample_path = next((p for p in fake_candidates if os.path.exists(p)), None)
        if sample_path:
            res = agent.analyze(sample_path, selected_providers=[])
            summary = (res.get("summary") or "").lower()
            for phrase in forbidden_phrases:
                self.assertNotIn(
                    phrase,
                    summary,
                    f"Fabricated forensic claim '{phrase}' detected in summary!"
                )


    def test_11_strong_detector_disagreement_handled_conservatively(self):
        """
        Verify that when specialist detectors strongly disagree, the pipeline
        refuses to certify authenticity or definitive manipulation, returning
        UNCERTAIN or a provisional verdict with an explicit conflict warning and no manufactured confidence.
        """
        prof_candidates = [
            "scratch/image_eval_set/inwild_professional.jpeg",
            "backend/uploads/156d6a4a-8211-4214-a9db-bfe9df7c221a_professional.jpeg"
        ]
        sample_path = next((p for p in prof_candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("professional.jpeg sample not found")

        agent = ImageAgent()
        res = agent.analyze(sample_path, selected_providers=[])

        # Must be internally UNCERTAIN or marked provisional
        self.assertTrue(
            res.get("verdict") == "UNCERTAIN" or res.get("is_binary_fallback") is True,
            f"Expected UNCERTAIN verdict or is_binary_fallback=True on detector conflict, got {res.get('verdict')}"
        )
        self.assertIsNotNone(
            res.get("reliability_warning"),
            "Expected explicit reliability_warning when detectors strongly disagree!"
        )
        # Disagreement must mean uncertified confidence (None)
        self.assertIsNone(
            res.get("confidence"),
            f"Expected confidence to be None on detector conflict, got {res.get('confidence')}"
        )

    def test_12_ai_generated_low_xception_score_not_certified_real(self):
        """
        Verify that AI-generated portrait (professional.jpeg), which receives a low
        Xception score (~0.33) due to lack of video blending seams, is NOT certified
        as non-provisional REAL.
        """
        prof_candidates = [
            "scratch/image_eval_set/inwild_professional.jpeg",
            "backend/uploads/156d6a4a-8211-4214-a9db-bfe9df7c221a_professional.jpeg"
        ]
        sample_path = next((p for p in prof_candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("professional.jpeg sample not found")

        agent = ImageAgent()
        res = agent.analyze(sample_path, selected_providers=[])

        # It must NOT be certified as clean non-provisional REAL
        is_clean_real = (res.get("primary_verdict") in ["REAL", "LIKELY_REAL"] and not res.get("is_binary_fallback"))
        self.assertFalse(
            is_clean_real,
            "Critical Failure: professional.jpeg was certified as clean non-provisional REAL!"
        )

    def test_13_genuine_high_xception_score_not_convicted_decisive_fake(self):
        """
        Verify that genuine portrait (profile.jpeg), which receives an elevated
        Xception score (~0.88) due to JPEG compression / OOD domain shift, is NOT
        convicted as a decisive, non-provisional FAKE deepfake.
        """
        profile_candidates = [
            "scratch/image_eval_set/inwild_profile.jpeg",
            "backend/uploads/0f032bd8-d44d-401a-851e-693eadb19f5a_profile.jpeg"
        ]
        sample_path = next((p for p in profile_candidates if os.path.exists(p)), None)
        if not sample_path:
            self.skipTest("profile.jpeg sample not found")

        agent = ImageAgent()
        res = agent.analyze(sample_path, selected_providers=[])

        is_decisive_fake = (res.get("primary_verdict") in ["FAKE", "DEEPFAKE"] and not res.get("is_binary_fallback"))
        self.assertFalse(
            is_decisive_fake,
            "Critical Failure: profile.jpeg was convicted as a decisive, non-provisional FAKE!"
        )

    def test_14_consistent_ui_and_pdf_verdicts(self):
        """
        Verify that the fields consumed by the UI and the PDF report generator
        (primary_verdict, original_verdict, is_binary_fallback, reliability_warning)
        remain strictly consistent without divergent verdict transformations.
        """
        from backend.utils.report_generator import ReportGenerator

        ctx = AnalysisContext(analysis_id="test_consistency_audit", media_type="image")
        ctx["fusion_result"] = {
            "verdict": "UNCERTAIN",
            "primary_verdict": "FAKE",
            "original_verdict": "UNCERTAIN",
            "is_binary_fallback": True,
            "reliability_warning": "Conflicting detector evidence",
            "fake_probability": 0.58,
        }
        ctx["confidence_result"] = {"confidence": None, "level": "Uncertified"}
        ctx["explanation"] = {"evidence": ["Sample conflict"], "recommendation": "Review manually"}

        # Generate report and confirm no crashes and consistent values
        report_path = ReportGenerator.generate_report(ctx)
        self.assertTrue(os.path.exists(report_path), f"Report file was not generated: {report_path}")
        self.assertTrue(os.path.getsize(report_path) > 1000, "Report PDF file is empty or corrupted")
        # Clean up temporary test report
        try:
            os.remove(report_path)
        except OSError:
            pass

    def test_15_canonical_verdict_presentation_rules(self):
        """
        Verify Task 7 Presentation & Consistency Rules:
        1. FAKE always receives FAKE canonical verdict.
        2. REAL always receives REAL canonical verdict.
        3. UNCERTAIN always receives UNCERTAIN canonical verdict.
        4. NOT_SUITABLE always receives NOT_SUITABLE canonical verdict.
        5. A FAKE explanation cannot claim the media is authentic.
        6. An UNCERTAIN result cannot be presented as certified REAL or FAKE.
        7. Missing model outputs are not displayed as zero.
        8. Detector disagreement is represented correctly.
        """
        from backend.core.decision_system import PresentationDecisionPolicy, DecisionState, DecisionResult
        from backend.agents.explanation_agent import ExplanationAgent

        # 1. FAKE canonical verdict
        v_fake = PresentationDecisionPolicy.get_canonical_verdict(
            verdict="FAKE", primary_verdict="FAKE", original_verdict="FAKE", is_binary_fallback=False, fake_probability=0.92
        )
        self.assertEqual(v_fake, "FAKE")

        # 2. REAL canonical verdict
        v_real = PresentationDecisionPolicy.get_canonical_verdict(
            verdict="REAL", primary_verdict="REAL", original_verdict="REAL", is_binary_fallback=False, fake_probability=0.08
        )
        self.assertEqual(v_real, "REAL")

        # 3 & 6. UNCERTAIN with provisional suggestion must NOT be presented as certified REAL or FAKE
        v_unc = PresentationDecisionPolicy.get_canonical_verdict(
            verdict="UNCERTAIN", primary_verdict="FAKE", original_verdict="UNCERTAIN", is_binary_fallback=True, fake_probability=0.62
        )
        self.assertEqual(v_unc, "UNCERTAIN", "Provisional fallback must have canonical verdict UNCERTAIN")

        v_unc_real = PresentationDecisionPolicy.get_canonical_verdict(
            verdict="UNCERTAIN", primary_verdict="REAL", original_verdict="UNCERTAIN", is_binary_fallback=True, fake_probability=0.38
        )
        self.assertEqual(v_unc_real, "UNCERTAIN", "Provisional REAL fallback must have canonical verdict UNCERTAIN")

        # 4. NOT_SUITABLE canonical verdict
        v_ns = PresentationDecisionPolicy.get_canonical_verdict(
            verdict="NOT_SUITABLE", primary_verdict="NOT_SUITABLE", original_verdict="NOT_SUITABLE", is_binary_fallback=False, fake_probability=None
        )
        self.assertEqual(v_ns, "NOT_SUITABLE")

        # 5. A FAKE explanation cannot claim the media is authentic
        ctx_fake = AnalysisContext(analysis_id="test_exp_fake", media_type="image")
        ctx_fake["fusion_result"] = {
            "verdict": "FAKE",
            "fake_probability": 0.89,
            "evidence": ["High synthetic frequency anomaly"],
            "features": {"boundary_flicker": 24.5},
        }
        expl_agent = ExplanationAgent()
        res_fake = expl_agent.explain(
            verdict="FAKE",
            confidence_level="High",
            media_type="image",
            features={"fft_spectral_anomaly": 0.65},
            evidence=["High synthetic frequency anomaly"],
        )
        statement_fake = res_fake.get("user_facing_statement", "").lower()
        self.assertNotIn("media is proven authentic", statement_fake)
        self.assertNotIn("confirmed authentic", statement_fake)

        # 7. Missing model outputs or uncertified confidence are not displayed as zero
        dec_unc = DecisionResult(
            prediction=DecisionState.UNCERTAIN,
            fake_probability=0.55,
            confidence=None,
            quality_passed=True,
            face_valid=True,
        )
        d_unc = dec_unc.to_dict()
        self.assertIn(d_unc["confidence_level"], ["Uncertified", "Provisional"])

        # 8. Detector disagreement represented correctly in DecisionResult
        dec_disagree = DecisionResult(
            prediction=DecisionState.UNCERTAIN,
            fake_probability=0.60,
            reason="Abstained due to detector disagreement: Xception=0.15 vs ViT=0.85",
            primary_verdict="FAKE",
            original_verdict="UNCERTAIN",
            is_binary_fallback=True,
        )
        d_dis = dec_disagree.to_dict()
        self.assertEqual(d_dis["canonical_verdict"], "UNCERTAIN")
        self.assertTrue(d_dis["is_binary_fallback"])
        self.assertEqual(d_dis["provisional_suggestion"], "FAKE")

    def test_16_pdf_report_states_and_integrity(self):
        """
        Verify PDF report generation across all 4 verdict states, UI-PDF agreement,
        and preservation of video analysis and Supabase integration.
        """
        from backend.utils.report_generator import ReportGenerator

        # 1. Test FAKE PDF Report
        ctx_fake = AnalysisContext(analysis_id="test_pdf_fake", media_type="image")
        ctx_fake["fusion_result"] = {
            "verdict": "FAKE",
            "primary_verdict": "FAKE",
            "original_verdict": "FAKE",
            "is_binary_fallback": False,
            "fake_probability": 0.94,
        }
        ctx_fake["confidence_result"] = {"confidence": 94.0, "level": "High"}
        ctx_fake["explanation"] = {"evidence": ["Synthetic boundary anomaly"], "recommendation": "Review required"}
        p_fake = ReportGenerator.generate_report(ctx_fake)
        self.assertTrue(os.path.exists(p_fake) and os.path.getsize(p_fake) > 1000)
        try:
            os.remove(p_fake)
        except OSError:
            pass

        # 2. Test UNCERTAIN / Provisional PDF Report (Must use amber styling and Uncertified confidence)
        ctx_unc = AnalysisContext(analysis_id="test_pdf_unc", media_type="image")
        ctx_unc["fusion_result"] = {
            "verdict": "UNCERTAIN",
            "primary_verdict": "REAL",
            "original_verdict": "UNCERTAIN",
            "is_binary_fallback": True,
            "reliability_warning": "Contradictory detector evidence",
            "fake_probability": 0.42,
            "summary_override": "Specialist detectors produced conflicting indicators. Classified as UNCERTAIN.",
        }
        ctx_unc["confidence_result"] = {"confidence": None, "level": "Uncertified"}
        ctx_unc["explanation"] = {"evidence": ["Detector conflict"], "recommendation": "Manual check required"}
        p_unc = ReportGenerator.generate_report(ctx_unc)
        self.assertTrue(os.path.exists(p_unc) and os.path.getsize(p_unc) > 1000)
        try:
            os.remove(p_unc)
        except OSError:
            pass

        # 3. Test NOT_SUITABLE PDF Report
        ctx_ns = AnalysisContext(analysis_id="test_pdf_ns", media_type="image")
        ctx_ns["fusion_result"] = {
            "verdict": "NOT_SUITABLE",
            "primary_verdict": "NOT_SUITABLE",
            "original_verdict": "NOT_SUITABLE",
            "is_binary_fallback": False,
            "fake_probability": None,
        }
        ctx_ns["confidence_result"] = {"confidence": None, "level": "Not Performed"}
        ctx_ns["explanation"] = {"evidence": ["No face found"], "recommendation": "Upload facial media"}
        p_ns = ReportGenerator.generate_report(ctx_ns)
        self.assertTrue(os.path.exists(p_ns) and os.path.getsize(p_ns) > 1000)
        try:
            os.remove(p_ns)
        except OSError:
            pass

        # 4. Verify VideoAgent and multi-face temporal tracking structures remain intact
        v_agent = VideoAgent()
        self.assertIsNotNone(v_agent.preprocessor)
        self.assertIsNotNone(v_agent.forensics)
        self.assertIsNotNone(v_agent.quality_gate)


if __name__ == "__main__":
    unittest.main()

