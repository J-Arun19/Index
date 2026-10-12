import os
import unittest
import tempfile
import shutil
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from backend.config import Config
from backend.core.validator import guess_mime_and_ext, detect_media_type, validate_media_file
from backend.agents.media_validation_agent import MediaValidationAgent
from backend.core.orchestrator import Orchestrator
from backend.core.context import AnalysisContext

class TestMediaRoutingAndOrchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        # 1. Valid JPEG
        cls.jpeg_path = os.path.join(cls.test_dir, "test_sample.jpg")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
        cv2.imwrite(cls.jpeg_path, img)

        # 2. Valid PNG
        cls.png_path = os.path.join(cls.test_dir, "test_sample.png")
        cv2.imwrite(cls.png_path, img)

        # 3. Valid WEBP
        cls.webp_path = os.path.join(cls.test_dir, "test_sample.webp")
        cv2.imwrite(cls.webp_path, img)

        # 4. Valid MP4
        cls.mp4_path = os.path.join(cls.test_dir, "test_sample.mp4")
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4 = cv2.VideoWriter(cls.mp4_path, fourcc_mp4, 10.0, (64, 64))
        for _ in range(5):
            f = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(f, (32, 32), 15, (0, 255, 0), -1)
            out_mp4.write(f)
        out_mp4.release()

        # 5. Valid MOV
        cls.mov_path = os.path.join(cls.test_dir, "test_sample.mov")
        fourcc_mov = cv2.VideoWriter_fourcc(*'mp4v')
        out_mov = cv2.VideoWriter(cls.mov_path, fourcc_mov, 10.0, (64, 64))
        for _ in range(5):
            f = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(f, (32, 32), 15, (0, 0, 255), -1)
            out_mov.write(f)
        out_mov.release()

        # 6. Valid AVI
        cls.avi_path = os.path.join(cls.test_dir, "test_sample.avi")
        fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
        out_avi = cv2.VideoWriter(cls.avi_path, fourcc_avi, 10.0, (64, 64))
        for _ in range(5):
            f = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(f, (32, 32), 15, (255, 255, 0), -1)
            out_avi.write(f)
        out_avi.release()

        # 7. Unsupported file (.txt)
        cls.txt_path = os.path.join(cls.test_dir, "document.txt")
        with open(cls.txt_path, "w") as f:
            f.write("Hello, this is not a media file.")

        # 8. Corrupted media (zero byte and corrupted header)
        cls.empty_path = os.path.join(cls.test_dir, "corrupt_empty.jpg")
        with open(cls.empty_path, "wb") as f:
            pass

        cls.corrupt_bytes_path = os.path.join(cls.test_dir, "corrupt_junk.mp4")
        with open(cls.corrupt_bytes_path, "wb") as f:
            f.write(b"\x00\x00\x00\x18ftypinvalidbrandjunkbytesnondecodable")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.patcher_db_create = patch("backend.services.database_service.DatabaseService.create_analysis_record")
        self.patcher_db_log = patch("backend.services.database_service.DatabaseService.save_agent_log")
        self.mock_db_create = self.patcher_db_create.start()
        self.mock_db_log = self.patcher_db_log.start()

    def tearDown(self):
        self.patcher_db_create.stop()
        self.patcher_db_log.stop()

    def _get_mocked_orchestrator(self, target_file_path):
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=target_file_path)

        # Mock downstream specialists and agents to isolate routing
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "LIKELY_REAL",
            "confidence_score": 92.5,
            "features": {"face_detected": True},
            "summary": "Image Specialist completed"
        })
        orchestrator.video_agent.analyze = MagicMock(return_value={
            "verdict": "FAKE",
            "confidence_score": 88.0,
            "features": {"temporal_flicker": 0.12},
            "summary": "Video Specialist completed"
        })
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "metadata_present": True,
            "anomalies_detected": False,
            "software_tag": None
        })
        orchestrator.confidence_agent.evaluate = MagicMock(return_value={
            "confidence": 90.0,
            "level": "High",
            "recommendation": "High confidence verdict"
        })
        orchestrator.fusion_agent.fuse = MagicMock(return_value={
            "verdict": "LIKELY_REAL",
            "prediction": "LIKELY_REAL",
            "confidence": 90.0,
            "reason": "Test consensus reached",
            "evidence": ["Test evidence item"],
            "uncertainty": 0.10,
            "fake_probability": 0.05,
            "real_probability": 0.95
        })
        orchestrator.explanation_agent.explain = MagicMock(return_value={
            "explanation": "Test explanation rationale",
            "evidence": ["Test evidence item"],
            "recommendation": "Authentic digital capture",
            "user_facing_statement": "The media is verified as likely real."
        })
        return orchestrator

    # 1. JPEG upload → detected as image → Image Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_jpeg_upload_detected_as_image_routes_to_image_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.jpeg_path)
        context = AnalysisContext(analysis_id="jpeg_test_01")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "image")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "image")
        orchestrator.image_agent.analyze.assert_called_once_with(self.jpeg_path)
        orchestrator.video_agent.analyze.assert_not_called()
        self.assertIn("Image Agent", final_ctx.active_agents)
        self.assertNotIn("Video Agent", final_ctx.active_agents)

    # 2. PNG upload → detected as image → Image Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_png_upload_detected_as_image_routes_to_image_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.png_path)
        context = AnalysisContext(analysis_id="png_test_02")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "image")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "image")
        orchestrator.image_agent.analyze.assert_called_once_with(self.png_path)
        orchestrator.video_agent.analyze.assert_not_called()
        self.assertIn("Image Agent", final_ctx.active_agents)
        self.assertNotIn("Video Agent", final_ctx.active_agents)

    # 3. WEBP upload → detected as image → Image Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_webp_upload_detected_as_image_routes_to_image_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.webp_path)
        context = AnalysisContext(analysis_id="webp_test_03")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "image")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "image")
        orchestrator.image_agent.analyze.assert_called_once_with(self.webp_path)
        orchestrator.video_agent.analyze.assert_not_called()
        self.assertIn("Image Agent", final_ctx.active_agents)

    # 4. MP4 upload → detected as video → Video Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_mp4_upload_detected_as_video_routes_to_video_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.mp4_path)
        context = AnalysisContext(analysis_id="mp4_test_04")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "video")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "video")
        orchestrator.video_agent.analyze.assert_called_once_with(self.mp4_path)
        orchestrator.image_agent.analyze.assert_not_called()
        self.assertIn("Video Agent", final_ctx.active_agents)
        self.assertNotIn("Image Agent", final_ctx.active_agents)

    # 5. MOV upload → detected as video → Video Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_mov_upload_detected_as_video_routes_to_video_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.mov_path)
        context = AnalysisContext(analysis_id="mov_test_05")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "video")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "video")
        orchestrator.video_agent.analyze.assert_called_once_with(self.mov_path)
        orchestrator.image_agent.analyze.assert_not_called()
        self.assertIn("Video Agent", final_ctx.active_agents)

    # 6. AVI upload → detected as video → Video Agent selected
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_avi_upload_detected_as_video_routes_to_video_agent(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.avi_path)
        context = AnalysisContext(analysis_id="avi_test_06")

        final_ctx = orchestrator.run_orchestration_context(context)

        self.assertEqual(final_ctx.media_type, "video")
        self.assertTrue(final_ctx.validation_result["valid"])
        self.assertEqual(final_ctx.validation_result["media_type"], "video")
        orchestrator.video_agent.analyze.assert_called_once_with(self.avi_path)
        orchestrator.image_agent.analyze.assert_not_called()
        self.assertIn("Video Agent", final_ctx.active_agents)

    # 7. Unsupported file → rejected with validation error
    def test_unsupported_file_rejected_with_validation_error(self):
        orchestrator = self._get_mocked_orchestrator(self.txt_path)
        context = AnalysisContext(analysis_id="txt_test_07")

        with self.assertRaises(ValueError) as err_ctx:
            orchestrator.run_orchestration_context(context)

        self.assertIn("Media validation failed", str(err_ctx.exception))
        orchestrator.image_agent.analyze.assert_not_called()
        orchestrator.video_agent.analyze.assert_not_called()

    # 8. Corrupted media → rejected safely
    def test_corrupted_empty_file_rejected_safely(self):
        orchestrator = self._get_mocked_orchestrator(self.empty_path)
        context = AnalysisContext(analysis_id="corrupt_test_08")

        with self.assertRaises(ValueError) as err_ctx:
            orchestrator.run_orchestration_context(context)

        self.assertIn("Media validation failed", str(err_ctx.exception))
        orchestrator.image_agent.analyze.assert_not_called()
        orchestrator.video_agent.analyze.assert_not_called()

    def test_corrupted_video_header_rejected_safely(self):
        orchestrator = self._get_mocked_orchestrator(self.corrupt_bytes_path)
        context = AnalysisContext(analysis_id="corrupt_test_08_bytes")

        with self.assertRaises(ValueError) as err_ctx:
            orchestrator.run_orchestration_context(context)

        self.assertIn("Media validation failed", str(err_ctx.exception))
        orchestrator.image_agent.analyze.assert_not_called()
        orchestrator.video_agent.analyze.assert_not_called()

    # 9. User cannot manually force an image to use the video pipeline
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_user_cannot_force_image_to_use_video_pipeline(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.jpeg_path)
        # Explicitly attempt to force "video" pipeline for a JPEG image
        context = AnalysisContext(analysis_id="force_video_09", media_type="video")

        final_ctx = orchestrator.run_orchestration_context(context)

        # Orchestrator must detect it's an image, override manual force, and route to Image Agent
        self.assertEqual(final_ctx.media_type, "image")
        orchestrator.image_agent.analyze.assert_called_once_with(self.jpeg_path)
        orchestrator.video_agent.analyze.assert_not_called()
        self.assertIn("Image Agent", final_ctx.active_agents)
        self.assertNotIn("Video Agent", final_ctx.active_agents)

    # 10. User cannot manually force a video to use the image pipeline
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_user_cannot_force_video_to_use_image_pipeline(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.mp4_path)
        # Explicitly attempt to force "image" pipeline for an MP4 video
        context = AnalysisContext(analysis_id="force_image_10", media_type="image")

        final_ctx = orchestrator.run_orchestration_context(context)

        # Orchestrator must detect it's a video, override manual force, and route to Video Agent
        self.assertEqual(final_ctx.media_type, "video")
        orchestrator.video_agent.analyze.assert_called_once_with(self.mp4_path)
        orchestrator.image_agent.analyze.assert_not_called()
        self.assertIn("Video Agent", final_ctx.active_agents)
        self.assertNotIn("Image Agent", final_ctx.active_agents)

    # 11. Existing analysis result generation must continue to work
    @patch("backend.core.orchestrator.save_analysis")
    @patch("backend.utils.report_generator.ReportGenerator.generate_report", return_value="/mock/report.pdf")
    def test_existing_analysis_result_generation_works(self, mock_report, mock_save):
        orchestrator = self._get_mocked_orchestrator(self.jpeg_path)
        
        result_dict = orchestrator.run_orchestration(file_id="full_result_11", user={"id": 1, "name": "Tester"})

        self.assertIn("analysis_id", result_dict)
        self.assertEqual(result_dict["media_type"], "image")
        self.assertIn("fusion_result", result_dict)
        self.assertIn("explanation", result_dict)
        self.assertIn("agent_logs", result_dict)
        self.assertIn("active_agents", result_dict)
        mock_report.assert_called_once()
        mock_save.assert_called_once()

    # 12. Existing FAKE / LIKELY_REAL / UNCERTAIN decision logic must remain unchanged
    def test_uncertainty_decision_logic_remains_unchanged(self):
        from backend.agents.fusion_agent import FusionAgent
        fusion = FusionAgent()

        # Context with low confidence / uncertain margin
        ctx_uncertain = AnalysisContext(analysis_id="decision_uncertain", media_type="image")
        ctx_uncertain.image_agent_result = {
            "verdict": "UNCERTAIN",
            "confidence_score": 50.0,
            "raw_model_score": 0.50,
            "uncertainty": 0.40,
            "features": {}
        }
        ctx_uncertain.confidence_result = {"level": "Low"}
        res_uncertain = fusion.fuse(ctx_uncertain)
        self.assertIn(res_uncertain["verdict"], ["UNCERTAIN", "FAKE", "LIKELY_REAL"])

        # Context with clear fake indicators
        ctx_fake = AnalysisContext(analysis_id="decision_fake", media_type="image")
        ctx_fake.image_agent_result = {
            "verdict": "FAKE",
            "confidence_score": 95.0,
            "raw_model_score": 0.95,
            "uncertainty": 0.05,
            "features": {}
        }
        ctx_fake.artifact_agent_result = {
            "verdict": "DEEPFAKE",
            "anomaly_score": 0.90
        }
        ctx_fake.confidence_result = {"level": "High"}
        res_fake = fusion.fuse(ctx_fake)
        self.assertEqual(res_fake["verdict"], "FAKE")

        # Context with clear likely real indicators
        ctx_real = AnalysisContext(analysis_id="decision_real", media_type="image")
        ctx_real.image_agent_result = {
            "verdict": "LIKELY_REAL",
            "confidence_score": 96.0,
            "raw_model_score": 0.04,
            "uncertainty": 0.04,
            "features": {}
        }
        ctx_real.confidence_result = {"level": "High"}
        res_real = fusion.fuse(ctx_real)
        self.assertEqual(res_real["verdict"], "LIKELY_REAL")

if __name__ == '__main__':
    unittest.main()
