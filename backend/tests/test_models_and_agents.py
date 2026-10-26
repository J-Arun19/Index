import os
import unittest
import torch
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from backend.utils.config import PipelineConfig
from backend.models.image_model import ImageModel
from backend.models.video_model import VideoModel
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.core.orchestrator import Orchestrator
from backend.core.context import AnalysisContext
from backend.agents.media_validation_agent import MediaValidationAgent

class TestModelsAndAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test artifacts
        cls.test_dir = tempfile.mkdtemp()
        cls.image_path = os.path.join(cls.test_dir, "test_face.jpg")
        cls.video_path = os.path.join(cls.test_dir, "test_video.mp4")
        
        # 1. Create a dummy image with a green circle representing a face
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 80, (0, 255, 0), -1)
        cv2.imwrite(cls.image_path, img)
        
        # 2. Create a dummy video with 20 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cls.video_path, fourcc, 10.0, (256, 256))
        for _ in range(20):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.circle(frame, (128, 128), 50, (0, 255, 0), -1)
            out.write(frame)
        out.release()
        
        # 3. Create dummy weights for the models
        cls.image_weights = os.path.join(cls.test_dir, "image_model_dummy.pth")
        cls.video_weights = os.path.join(cls.test_dir, "video_model_dummy.pth")
        
        image_model = ImageModel(num_classes=2)
        video_model = VideoModel(num_classes=2)
        
        torch.save(image_model.state_dict(), cls.image_weights)
        torch.save(video_model.state_dict(), cls.video_weights)

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_image_model_shapes(self):
        """Test that ImageModel forward and predict methods return correct tensor shapes."""
        model = ImageModel(num_classes=2)
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Test forward() returns logits
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, 2))
        
        # Test predict() returns structured outputs
        outputs = model.predict(dummy_input)
        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertIn("predicted_class", outputs)
        self.assertIn("confidence_score", outputs)
        
        self.assertEqual(outputs["logits"].shape, (2, 2))
        self.assertEqual(outputs["probabilities"].shape, (2, 2))
        self.assertEqual(outputs["predicted_class"].shape, (2,))
        self.assertEqual(outputs["confidence_score"].shape, (2,))
        
        # Sum of probabilities per sample should be close to 1
        probs_sum = outputs["probabilities"].sum(dim=1)
        for val in probs_sum:
            self.assertAlmostEqual(val.item(), 1.0, places=5)

    def test_video_model_initialization(self):
        """Test that VideoModel initializes correct layers and parameters."""
        model = VideoModel(num_classes=2)
        # Check LSTM details
        self.assertTrue(hasattr(model, "lstm"))
        self.assertIsInstance(model.lstm, torch.nn.LSTM)
        self.assertEqual(model.lstm.input_size, 512)
        self.assertEqual(model.lstm.hidden_size, 256)
        self.assertEqual(model.lstm.num_layers, 1)
        self.assertTrue(model.lstm.batch_first)
        
        # Check fc classification head details
        self.assertTrue(hasattr(model, "fc"))
        self.assertIsInstance(model.fc, torch.nn.Linear)
        self.assertEqual(model.fc.in_features, 256)
        self.assertEqual(model.fc.out_features, 2)

    def test_video_model_forward_shape(self):
        """Test that VideoModel forward() returns the correct logits tensor shape."""
        model = VideoModel(num_classes=2)
        # Input batch size 2, sequence length 30, 3 channels, 224x224
        dummy_input = torch.randn(2, 30, 3, 224, 224)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, 2))

    @patch('backend.utils.config.PipelineConfig')
    def test_video_model_predict_detailed(self, mock_config):
        """Test predict() details, output dict, probabilities shape, class mapping, confidence, and threshold behavior."""
        mock_config.CLASSIFICATION_THRESHOLD = 0.65
        model = VideoModel(num_classes=2)
        dummy_input = torch.randn(2, 30, 3, 224, 224)
        
        outputs = model.predict(dummy_input)
        
        # 1. Dictionary fields check
        self.assertIn("logits", outputs)
        self.assertIn("probabilities", outputs)
        self.assertIn("predicted_class", outputs)
        self.assertIn("confidence_score", outputs)
        
        # 2. Probability shape check
        self.assertEqual(outputs["probabilities"].shape, (2, 2))
        probs_sum = outputs["probabilities"].sum(dim=1)
        for val in probs_sum:
            self.assertAlmostEqual(val.item(), 1.0, places=5)
            
        # 3. Predicted class values, mapping, and threshold checks
        # Let's inspect the values and check manual math alignment
        fake_prob_0 = outputs["probabilities"][0, 1].item()
        pred_class_0 = outputs["predicted_class"][0].item()
        confidence_0 = outputs["confidence_score"][0].item()
        
        # If fake probability >= 0.65, predicted class should be 1 (FAKE). Else 0 (REAL).
        expected_class_0 = 1 if fake_prob_0 >= 0.65 else 0
        self.assertEqual(pred_class_0, expected_class_0)
        
        # 4. Confidence must represent the probability of the predicted class
        expected_conf_0 = fake_prob_0 if expected_class_0 == 1 else outputs["probabilities"][0, 0].item()
        self.assertAlmostEqual(confidence_0, expected_conf_0, places=5)

    @patch('backend.agents.image_agent.PipelineConfig')
    def test_image_agent_inference_flow(self, mock_config):
        """Test ImageAgent with real inference model weights."""
        mock_config.IMAGE_MODEL_PATH = self.image_weights
        mock_config.CLASSIFICATION_THRESHOLD = 0.5
        mock_config.IMAGE_SIZE = (224, 224)
        mock_config.DEVICE = "cpu"
        
        agent = ImageAgent()
        result = agent.analyze(self.image_path)
        
        # Verify structure
        self.assertIn("verdict", result)
        self.assertIn("confidence_score", result)
        self.assertIn("features", result)
        self.assertIn("summary", result)
        self.assertIn("processing_time", result)
        
        self.assertIn(result["verdict"], ["REAL", "DEEPFAKE", "FAKE", "NOT_SUITABLE"])
        if result["confidence_score"] is not None:
            self.assertGreaterEqual(result["confidence_score"], 0)
            self.assertLessEqual(result["confidence_score"], 100)

    @patch('backend.agents.video_agent.PipelineConfig')
    def test_video_agent_inference_flow(self, mock_config):
        """Test VideoAgent with real inference model weights."""
        mock_config.VIDEO_MODEL_PATH = self.video_weights
        mock_config.CLASSIFICATION_THRESHOLD = 0.5
        mock_config.IMAGE_SIZE = (224, 224)
        mock_config.DEVICE = "cpu"
        # Re-initialize to ensure it uses video size configuration
        mock_config.FRAME_INTERVAL = 10
        mock_config.MAX_FRAMES_PER_VIDEO = 30
        
        agent = VideoAgent()
        result = agent.analyze(self.video_path)
        
        # Verify structure
        self.assertIn("verdict", result)
        self.assertIn("confidence_score", result)
        self.assertIn("features", result)
        self.assertIn("summary", result)
        self.assertIn("processing_time", result)
        
        self.assertIn(result["verdict"], ["REAL", "DEEPFAKE", "FAKE", "UNCERTAIN", "NOT_SUITABLE"])
        if result["confidence_score"] is not None:
            self.assertGreaterEqual(result["confidence_score"], 0)
            self.assertLessEqual(result["confidence_score"], 100)

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    @patch('backend.agents.image_agent.PipelineConfig')
    def test_orchestrator_mock_end_to_end(self, mock_config, mock_gen_report, mock_save_analysis):
        """Test Orchestrator runs successfully using the agentic pipeline."""
        mock_config.IMAGE_MODEL_PATH = self.image_weights
        mock_config.CLASSIFICATION_THRESHOLD = 0.5
        mock_config.IMAGE_SIZE = (224, 224)
        mock_config.DEVICE = "cpu"
        
        # Setup mocks to avoid external API / storage calls
        mock_gen_report.return_value = "/mock/report/path.pdf"
        mock_save_analysis.return_value = True
        
        orchestrator = Orchestrator()
        # Mock locate_file to return our dummy image path
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        context = AnalysisContext(analysis_id="test_analysis_123", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # Verify execution context logs
        self.assertEqual(final_context.uploaded_file, "test_face.jpg")
        self.assertIsNotNone(final_context.validation_result)
        self.assertTrue(final_context.validation_result["valid"])
        self.assertIsNotNone(final_context.image_agent_result)
        self.assertIsNotNone(final_context.confidence_result)
        self.assertIsNotNone(final_context.fusion_result)
        self.assertIsNotNone(final_context.explanation)
        
        # Ensure database/report integration was triggered
        mock_gen_report.assert_called_once()
        mock_save_analysis.assert_called_once()

    def test_validation_valid_image(self):
        """Test that a valid image passes validation."""
        val_agent = MediaValidationAgent()
        res = val_agent.validate(self.image_path, expected_media_type="image")
        self.assertTrue(res["valid"])
        self.assertEqual(res["media_type"], "image")
        self.assertTrue(res["checks"]["file_exists"])
        self.assertTrue(res["checks"]["non_empty"])
        self.assertTrue(res["checks"]["signature_valid"])
        self.assertTrue(res["checks"]["decodable"])
        self.assertTrue(res["checks"]["dimensions_valid"])
        self.assertTrue(res["checks"]["frame_validation"])

    def test_validation_valid_video(self):
        """Test that a valid video passes validation."""
        val_agent = MediaValidationAgent()
        res = val_agent.validate(self.video_path, expected_media_type="video")
        self.assertTrue(res["valid"])
        self.assertEqual(res["media_type"], "video")
        self.assertTrue(res["checks"]["file_exists"])
        self.assertTrue(res["checks"]["non_empty"])
        self.assertTrue(res["checks"]["signature_valid"])
        self.assertTrue(res["checks"]["decodable"])
        self.assertTrue(res["checks"]["dimensions_valid"])
        self.assertTrue(res["checks"]["frame_validation"])

    def test_validation_non_existent_file(self):
        """Test that validation fails for a non-existent file."""
        val_agent = MediaValidationAgent()
        res = val_agent.validate(os.path.join(self.test_dir, "ghost.jpg"), expected_media_type="image")
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["file_exists"])

    def test_validation_empty_file(self):
        """Test that validation fails for an empty (0-byte) file."""
        empty_path = os.path.join(self.test_dir, "empty.jpg")
        with open(empty_path, "wb") as f:
            pass
        val_agent = MediaValidationAgent()
        res = val_agent.validate(empty_path, expected_media_type="image")
        self.assertFalse(res["valid"])
        self.assertTrue(res["checks"]["file_exists"])
        self.assertFalse(res["checks"]["non_empty"])

    def test_validation_corrupted_image(self):
        """Test that an image with valid signature but corrupted content fails validation."""
        # Read the first few bytes of the real image to get signature, but write a truncated file
        with open(self.image_path, "rb") as f:
            header = f.read(16)
        corrupted_path = os.path.join(self.test_dir, "corrupted_image.jpg")
        with open(corrupted_path, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 20)
            
        val_agent = MediaValidationAgent()
        res = val_agent.validate(corrupted_path, expected_media_type="image")
        self.assertFalse(res["valid"])
        self.assertTrue(res["checks"]["signature_valid"])
        self.assertFalse(res["checks"]["decodable"])

    def test_validation_corrupted_video(self):
        """Test that a video with valid signature but corrupted content fails validation."""
        # Read headers of valid video, write truncated file
        with open(self.video_path, "rb") as f:
            header = f.read(32)
        corrupted_path = os.path.join(self.test_dir, "corrupted_video.mp4")
        with open(corrupted_path, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 20)
            
        val_agent = MediaValidationAgent()
        res = val_agent.validate(corrupted_path, expected_media_type="video")
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["frame_validation"])

    def test_validation_image_renamed_as_video(self):
        """Test that an image file renamed with a video extension fails validation."""
        renamed_path = os.path.join(self.test_dir, "image_fake.mp4")
        shutil.copy(self.image_path, renamed_path)
        
        val_agent = MediaValidationAgent()
        res = val_agent.validate(renamed_path, expected_media_type="video")
        self.assertFalse(res["valid"])
        self.assertEqual(res["media_type"], "image")
        self.assertIn("Expected media type 'video', but detected 'image'", res["errors"])

    def test_validation_video_renamed_as_image(self):
        """Test that a video file renamed with an image extension fails validation."""
        renamed_path = os.path.join(self.test_dir, "video_fake.jpg")
        shutil.copy(self.video_path, renamed_path)
        
        val_agent = MediaValidationAgent()
        res = val_agent.validate(renamed_path, expected_media_type="image")
        self.assertFalse(res["valid"])
        self.assertEqual(res["media_type"], "video")
        self.assertIn("Expected media type 'image', but detected 'video'", res["errors"])

    def test_validation_invalid_dimensions(self):
        """Test that invalid dimensions fail validation."""
        val_agent = MediaValidationAgent()
        with patch('cv2.imread') as mock_imread:
            # mock returns an array with shape (0, 0, 3)
            mock_imread.return_value = np.zeros((0, 0, 3), dtype=np.uint8)
            res = val_agent.validate(self.image_path, expected_media_type="image")
            self.assertFalse(res["valid"])
            self.assertFalse(res["checks"]["dimensions_valid"])

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_gatekeeper_invalid_media_does_not_call_downstream(self, mock_gen_report, mock_save_analysis):
        """Test that invalid media aborts early and does not call any downstream agent."""
        orchestrator = Orchestrator()
        
        # Mock validation to return invalid result
        mock_invalid = {
            "valid": False,
            "media_type": "image",
            "checks": {
                "file_exists": True,
                "non_empty": True,
                "signature_valid": False,
                "decodable": False,
                "dimensions_valid": False,
                "frame_validation": False
            },
            "errors": ["Signature validation failed"],
            "warnings": [],
            "summary": "Media validation failed."
        }
        orchestrator.validation_agent.validate = MagicMock(return_value=mock_invalid)
        
        # Spy on downstream agents
        orchestrator.image_agent.analyze = MagicMock()
        orchestrator.video_agent.analyze = MagicMock()
        orchestrator.confidence_agent.evaluate = MagicMock()
        orchestrator.fusion_agent.fuse = MagicMock()
        orchestrator.explanation_agent.explain = MagicMock()
        
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        context = AnalysisContext(analysis_id="test_invalid_gate", media_type="image")
        
        with self.assertRaises(ValueError) as context_err:
            orchestrator.run_orchestration_context(context)
            
        self.assertIn("Media validation failed: Signature validation failed", str(context_err.exception))
        
        # Verify that downstream agents were NOT called
        orchestrator.image_agent.analyze.assert_not_called()
        orchestrator.video_agent.analyze.assert_not_called()
        orchestrator.confidence_agent.evaluate.assert_not_called()
        orchestrator.fusion_agent.fuse.assert_not_called()
        orchestrator.explanation_agent.explain.assert_not_called()
        mock_gen_report.assert_not_called()
        
        # Verify database recorded the failure as FAILED_VALIDATION
        mock_save_analysis.assert_called_once_with(
            analysis_id="test_invalid_gate",
            user_id=None,
            user_name="Guest",
            file_name="test_face.jpg",
            media_type="image",
            prediction="FAILED_VALIDATION",
            confidence=0.0,
            report_path=None
        )

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    @patch('backend.agents.image_agent.PipelineConfig')
    def test_gatekeeper_valid_media_routes_correctly(self, mock_config, mock_gen_report, mock_save_analysis):
        """Test that valid media executes downstream agents and routes to the correct specialized agent."""
        mock_config.IMAGE_MODEL_PATH = self.image_weights
        mock_config.CLASSIFICATION_THRESHOLD = 0.5
        mock_config.IMAGE_SIZE = (224, 224)
        mock_config.DEVICE = "cpu"
        
        orchestrator = Orchestrator()
        
        # Spy on image and video agents
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 90,
            "features": {"noise_inconsistency": 0.05, "edge_boundary_artifacts": 0.02, "metadata_anomaly": False},
            "summary": "Mock summary",
            "processing_time": 0.1
        })
        orchestrator.video_agent.analyze = MagicMock()
        orchestrator.confidence_agent.evaluate = MagicMock(return_value={"level": "High", "confidence": 0.9})
        orchestrator.fusion_agent.fuse = MagicMock(return_value={"verdict": "REAL", "confidence": 0.9})
        orchestrator.explanation_agent.explain = MagicMock(return_value={"evidence": [], "recommendation": "None"})
        mock_gen_report.return_value = "/mock/report/path.pdf"
        
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        context = AnalysisContext(analysis_id="test_valid_gate", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # Verify specialized image agent was called
        orchestrator.image_agent.analyze.assert_called_once_with(self.image_path)
        # Verify video agent was NOT called
        orchestrator.video_agent.analyze.assert_not_called()
        
        # Verify fusion, confidence, explanation, and report generation were called
        orchestrator.confidence_agent.evaluate.assert_called_once()
        orchestrator.fusion_agent.fuse.assert_called_once()
        orchestrator.explanation_agent.explain.assert_called_once()
        mock_gen_report.assert_called_once()
        
        # Verify database saved the successful result
        mock_save_analysis.assert_called_once_with(
            analysis_id="test_valid_gate",
            user_id=None,
            user_name="Guest",
            file_name="test_face.jpg",
            media_type="image",
            prediction="REAL",
            confidence=0.9,
            report_path="/mock/report/path.pdf"
        )

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    @patch('backend.agents.video_agent.PipelineConfig')
    def test_gatekeeper_valid_video_routes_correctly(self, mock_config, mock_gen_report, mock_save_analysis):
        """Test that valid video media executes downstream agents and routes to the correct specialized agent."""
        mock_config.VIDEO_MODEL_PATH = self.video_weights
        mock_config.CLASSIFICATION_THRESHOLD = 0.5
        mock_config.MAX_FRAMES_PER_VIDEO = 5
        mock_config.DEVICE = "cpu"
        
        orchestrator = Orchestrator()
        
        # Spy on image and video agents
        orchestrator.image_agent.analyze = MagicMock()
        orchestrator.video_agent.analyze = MagicMock(return_value={
            "verdict": "FAKE",
            "confidence_score": 85,
            "features": {"temporal_inconsistency": 0.45, "frame_anomalies": [0, 1, 0, 0, 1]},
            "summary": "Mock video summary",
            "processing_time": 0.2
        })
        orchestrator.confidence_agent.evaluate = MagicMock(return_value={"level": "High", "confidence": 0.85})
        orchestrator.fusion_agent.fuse = MagicMock(return_value={"verdict": "FAKE", "confidence": 0.85})
        orchestrator.explanation_agent.explain = MagicMock(return_value={"evidence": ["temporal anomaly"], "recommendation": "Review"})
        mock_gen_report.return_value = "/mock/report/video_path.pdf"
        
        orchestrator.locate_file = MagicMock(return_value=self.video_path)
        
        context = AnalysisContext(analysis_id="test_valid_video_gate", media_type="video")
        final_context = orchestrator.run_orchestration_context(context)
        
        # Verify specialized video agent was called
        orchestrator.video_agent.analyze.assert_called_once_with(self.video_path)
        # Verify image agent was NOT called
        orchestrator.image_agent.analyze.assert_not_called()
        
        # Verify fusion, confidence, explanation, and report generation were called
        orchestrator.confidence_agent.evaluate.assert_called_once()
        orchestrator.fusion_agent.fuse.assert_called_once()
        orchestrator.explanation_agent.explain.assert_called_once()
        mock_gen_report.assert_called_once()
        
        # Verify database saved the successful result
        mock_save_analysis.assert_called_once_with(
            analysis_id="test_valid_video_gate",
            user_id=None,
            user_name="Guest",
            file_name="test_video.mp4",
            media_type="video",
            prediction="FAKE",
            confidence=0.85,
            report_path="/mock/report/video_path.pdf"
        )

    def test_api_validation_failure_response(self):
        """Test that Flask API returns HTTP 400 with structured validation payload on failure."""
        from backend.core.orchestrator import Orchestrator
        from backend.api.analysis import analysis_bp
        from flask import Flask
        
        app = Flask(__name__)
        app.register_blueprint(analysis_bp, url_prefix="/api")
        client = app.test_client()
        
        mock_validation_res = {
            "valid": False,
            "media_type": "image",
            "checks": {
                "file_exists": True,
                "non_empty": True,
                "signature_valid": False,
                "decodable": False,
                "dimensions_valid": False,
                "frame_validation": False
            },
            "errors": ["Signature validation failed"],
            "warnings": [],
            "summary": "Media validation failed."
        }
        val_error = ValueError("Media validation failed: Signature validation failed")
        val_error.validation_result = mock_validation_res
        
        mock_orch = MagicMock()
        mock_orch.run_orchestration.side_effect = val_error
        with patch('backend.api.analysis.get_orchestrator', return_value=mock_orch):
            response = client.post("/api/analyze", json={
                "file_id": "test_123",
                "media_type": "image"
            })
            
            self.assertEqual(response.status_code, 400)
            json_data = response.get_json()
            self.assertFalse(json_data["success"])
            self.assertEqual(json_data["error"], "MEDIA_VALIDATION_FAILED")
            self.assertEqual(json_data["validation"], mock_validation_res)

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_high_confidence_skips_artifact_agent(self, mock_gen_report, mock_save_analysis):
        """Test that high confidence and aligned metadata skips ArtifactAgent."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        # Mock primary agent output: High confidence REAL (95%)
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 95,
            "features": {"noise_inconsistency": 0.05, "edge_boundary_artifacts": 0.02},
            "summary": "High confidence authentic"
        })
        
        # Mock metadata: No anomalies
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomalies_detected": False,
            "summary": "Clean metadata"
        })
        
        # Mock artifact agent to see if it is called (should not be)
        orchestrator.artifact_agent.analyze = MagicMock()
        
        context = AnalysisContext(analysis_id="test_high_conf_skip", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # ArtifactAgent should be skipped
        orchestrator.artifact_agent.analyze.assert_not_called()
        self.assertIn("ArtifactAgent skipped", final_context.decision_history[0])

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_low_confidence_activates_artifact_agent(self, mock_gen_report, mock_save_analysis):
        """Test that low confidence triggers ArtifactAgent activation."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        # Mock primary agent output: Low confidence (65%)
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 65,
            "features": {"noise_inconsistency": 0.35, "edge_boundary_artifacts": 0.25},
            "summary": "Low confidence authentic"
        })
        
        # Mock metadata: No anomalies (aligned with REAL)
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomalies_detected": False,
            "summary": "Clean metadata"
        })
        
        # Mock artifact agent
        orchestrator.artifact_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomaly_score": 0.30,
            "verdict": "REAL",
            "summary": "Clean pixels"
        })
        
        context = AnalysisContext(analysis_id="test_low_conf_trigger", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # ArtifactAgent should be activated
        orchestrator.artifact_agent.analyze.assert_called_once()
        self.assertIn("ArtifactAgent activated: primary confidence", final_context.decision_history[0])

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_metadata_disagreement_activates_artifact_agent(self, mock_gen_report, mock_save_analysis):
        """Test that disagreement between primary verdict and metadata triggers ArtifactAgent."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        # Mock primary agent output: High confidence REAL (90%)
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 90,
            "features": {"noise_inconsistency": 0.05, "edge_boundary_artifacts": 0.02},
            "summary": "High confidence authentic"
        })
        
        # Mock metadata: Anomalies detected (photoshop tags present) -> Disagreement!
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomalies_detected": True,
            "software_tag": "Photoshop",
            "summary": "Photoshop trace found"
        })
        
        # Mock artifact agent
        orchestrator.artifact_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomaly_score": 0.60,
            "verdict": "DEEPFAKE",
            "summary": "Anomalies found"
        })
        
        context = AnalysisContext(analysis_id="test_disagree_trigger", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # ArtifactAgent should be activated
        orchestrator.artifact_agent.analyze.assert_called_once()
        self.assertIn("ArtifactAgent activated: conflict detected", final_context.decision_history[0])

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_aligned_specialists_avoid_unnecessary_deep_scan(self, mock_gen_report, mock_save_analysis):
        """Test that fully aligned specialists with high confidence skip ArtifactAgent."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "DEEPFAKE",
            "confidence_score": 85,
            "features": {"noise_inconsistency": 0.70, "edge_boundary_artifacts": 0.65},
            "summary": "High confidence fake"
        })
        
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomalies_detected": True,
            "software_tag": "Photoshop",
            "summary": "Photoshop trace found"
        })
        
        orchestrator.artifact_agent.analyze = MagicMock()
        
        context = AnalysisContext(analysis_id="test_aligned_skip", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        orchestrator.artifact_agent.analyze.assert_not_called()
        self.assertIn("ArtifactAgent skipped", final_context.decision_history[0])

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_metadata_agent_failure_is_recovered(self, mock_gen_report, mock_save_analysis):
        """Test that MetadataAgent failure is gracefully recovered and doesn't crash the pipeline."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 90,
            "features": {},
            "summary": "Authentic"
        })
        
        # Force MetadataAgent to raise exception
        orchestrator.metadata_agent.analyze = MagicMock(side_effect=Exception("Metadata read timeout error"))
        
        context = AnalysisContext(analysis_id="test_metadata_fail", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # Verify the pipeline completed successfully
        self.assertIsNotNone(final_context.fusion_result)
        # Verify failure was logged in context
        self.assertEqual(len(final_context.failures), 1)
        self.assertEqual(final_context.failures[0]["agent"], "MetadataAgent")
        self.assertIn("Metadata read timeout error", final_context.failures[0]["error"])
        # Ensure status logged as Failed (Recovered)
        failed_log = next(log for log in final_context.agent_logs if log["agent"] == "MetadataAgent")
        self.assertEqual(failed_log["status"], "Failed (Recovered)")

    @patch('backend.core.orchestrator.save_analysis')
    @patch('backend.utils.report_generator.ReportGenerator.generate_report')
    def test_artifact_agent_failure_is_recovered(self, mock_gen_report, mock_save_analysis):
        """Test that ArtifactConsistencyAgent failure is gracefully recovered and doesn't crash the pipeline."""
        orchestrator = Orchestrator()
        orchestrator.locate_file = MagicMock(return_value=self.image_path)
        
        # Force low confidence to trigger ArtifactAgent
        orchestrator.image_agent.analyze = MagicMock(return_value={
            "verdict": "REAL",
            "confidence_score": 60,
            "features": {},
            "summary": "Authentic"
        })
        
        # Metadata OK
        orchestrator.metadata_agent.analyze = MagicMock(return_value={
            "status": "Success",
            "anomalies_detected": False,
            "summary": "Clean metadata"
        })
        
        # Force ArtifactAgent to raise exception
        orchestrator.artifact_agent.analyze = MagicMock(side_effect=RuntimeError("GPU out of memory"))
        
        context = AnalysisContext(analysis_id="test_artifact_fail", media_type="image")
        final_context = orchestrator.run_orchestration_context(context)
        
        # Verify successful completion
        self.assertIsNotNone(final_context.fusion_result)
        # Verify failure recorded
        self.assertEqual(len(final_context.failures), 1)
        self.assertEqual(final_context.failures[0]["agent"], "ArtifactConsistencyAgent")
        self.assertIn("GPU out of memory", final_context.failures[0]["error"])
        # Check logs
        failed_log = next(log for log in final_context.agent_logs if log["agent"] == "ArtifactConsistencyAgent")
        self.assertEqual(failed_log["status"], "Failed (Recovered)")

    def test_fusion_agent_handles_missing_artifact_agent(self):
        """Test that FusionAgent performs weight normalization when ArtifactAgent is skipped."""
        from backend.agents.fusion_agent import FusionAgent
        from backend.core.context import AnalysisContext
        
        context = AnalysisContext(analysis_id="test_fusion_no_art", media_type="image")
        
        # ImageAgent result
        context.image_agent_result = {
            "verdict": "DEEPFAKE",
            "confidence_score": 80,
            "summary": "Image anomalies detected"
        }
        
        # MetadataAgent result
        context.metadata_agent_result = {
            "status": "Success",
            "anomalies_detected": False,
            "summary": "No metadata anomalies"
        }
        
        # ArtifactAgent skipped -> remains None
        context.artifact_agent_result = None
        
        fusion = FusionAgent()
        res = fusion.fuse(context)
        
        self.assertIn(res["verdict"], ["DEEPFAKE", "FAKE"])
        self.assertAlmostEqual(res["final_probability"], 0.6385, places=3)
        self.assertIn("ImageAgent", res["active_agents"])
        self.assertIn("MetadataAgent", res["active_agents"])
        self.assertNotIn("ArtifactConsistencyAgent", res["active_agents"])

    def test_fusion_agent_normalizes_weights_correctly(self):
        """Test weight normalizations when all agents are active."""
        from backend.agents.fusion_agent import FusionAgent
        from backend.core.context import AnalysisContext
        
        context = AnalysisContext(analysis_id="test_fusion_all", media_type="image")
        
        context.image_agent_result = {
            "verdict": "DEEPFAKE",
            "confidence_score": 80,
            "summary": "Primary fake"
        }
        
        context.metadata_agent_result = {
            "status": "Success",
            "anomalies_detected": True,
            "summary": "Metadata anomalies"
        }
        
        context.artifact_agent_result = {
            "status": "Success",
            "anomaly_score": 0.90,
            "verdict": "DEEPFAKE",
            "summary": "Artifact anomalies"
        }
        
        fusion = FusionAgent()
        res = fusion.fuse(context)
        
        self.assertIn(res["verdict"], ["DEEPFAKE", "FAKE"])
        self.assertAlmostEqual(res["final_probability"], 0.82, places=2)

    def test_fusion_agent_detects_disagreement(self):
        """Test that FusionAgent detects disagreement when specialists point to different classes."""
        from backend.agents.fusion_agent import FusionAgent
        from backend.core.context import AnalysisContext
        
        context = AnalysisContext(analysis_id="test_disagree", media_type="image")
        
        context.image_agent_result = {
            "verdict": "DEEPFAKE",
            "confidence_score": 80,
            "summary": "Primary fake"
        }
        
        context.metadata_agent_result = {
            "status": "Success",
            "anomalies_detected": False,
            "summary": "No metadata anomalies"
        }
        
        context.artifact_agent_result = {
            "status": "Success",
            "anomaly_score": 0.20,
            "verdict": "REAL",
            "summary": "No artifacts"
        }
        
        fusion = FusionAgent()
        res = fusion.fuse(context)
        
        self.assertTrue(res["disagreement_detected"])
        self.assertGreater(res["conflict_level"], 0.0)

    def test_final_confidence_decreases_under_disagreement(self):
        """Test that confidence is penalized when strong disagreement is detected."""
        from backend.agents.fusion_agent import FusionAgent
        from backend.core.context import AnalysisContext
        
        # Case A: Fully Aligned
        context_aligned_high = AnalysisContext(analysis_id="test_aligned_high", media_type="image")
        context_aligned_high.image_agent_result = {"verdict": "DEEPFAKE", "confidence_score": 90}
        context_aligned_high.metadata_agent_result = {"status": "Success", "anomalies_detected": True}
        context_aligned_high.artifact_agent_result = {"status": "Success", "anomaly_score": 0.90, "verdict": "DEEPFAKE"}
        
        fusion = FusionAgent()
        high_aligned_res = fusion.fuse(context_aligned_high)
        
        # Case B: Disaligned
        context_disagree_high = AnalysisContext(analysis_id="test_disagree_high", media_type="image")
        context_disagree_high.image_agent_result = {"verdict": "DEEPFAKE", "confidence_score": 90}
        context_disagree_high.metadata_agent_result = {"status": "Success", "anomalies_detected": True}
        context_disagree_high.artifact_agent_result = {"status": "Success", "anomaly_score": 0.10, "verdict": "REAL"}
        
        high_disagree_res = fusion.fuse(context_disagree_high)
        
        self.assertLess(high_disagree_res["confidence"], high_aligned_res["confidence"])

if __name__ == "__main__":
    unittest.main()
