import os
import unittest
import base64
import cv2
import numpy as np

from backend.app import create_app
from backend.config import Config


class TestRealTimeMonitoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.upload_dir = Config.UPLOAD_FOLDER

    def test_01_start_monitoring_session(self):
        """Test session initialization endpoint."""
        res = self.client.post('/api/monitor/start-session', json={
            "user_id": "test_user_id",
            "user_name": "Test Inspector"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertTrue(data.get("session_id", "").startswith("mon_"))

    def test_02_get_sample_videos(self):
        """Test sample videos retrieval for recorded video mode."""
        res = self.client.get('/api/monitor/samples')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("samples", data)
        samples = data["samples"]
        self.assertGreaterEqual(len(samples), 1)

        # Confirm each sample video file actually exists on disk
        for sample in samples:
            self.assertIn("id", sample)
            self.assertIn("filename", sample)
            self.assertIn("url", sample)
            self.assertIn("expected_verdict", sample)
            path = os.path.join(self.upload_dir, sample["filename"])
            self.assertTrue(os.path.exists(path), f"Sample video missing: {path}")

    def test_03_serve_sample_video(self):
        """Test video streaming endpoint with proper MIME headers."""
        res = self.client.get('/api/monitor/samples')
        sample = res.get_json()["samples"][0]
        video_res = self.client.get(sample["url"])
        self.assertEqual(video_res.status_code, 200)
        self.assertEqual(video_res.content_type, "video/mp4")
        self.assertGreater(len(video_res.data), 1000)

    def test_04_analyze_frame_genuine_face_tracking(self):
        """
        Test that genuine facial video produces aligned bounding box,
        REAL verdict, genuine model scores, and no false deepfake accusation.
        """
        video_name = "3cd9b7ec-16b3-467f-ade3-df1b246ca5dd_989_993.mp4"
        video_path = os.path.join(self.upload_dir, video_name)
        if not os.path.exists(video_path):
            self.skipTest(f"Sample video {video_name} not found")

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read test frame from video")

        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf).decode('utf-8')

        res = self.client.post('/api/monitor/analyze-frame', json={'frame': b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("verdict"), "REAL")
        self.assertEqual(data.get("risk_level"), "LOW")
        self.assertIn("faces", data)
        self.assertGreater(len(data["faces"]), 0, "Expected at least 1 face tracked")

        face = data["faces"][0]
        # Check bounding box format [x, y, w, h]
        self.assertEqual(len(face["box"]), 4)
        for coord in face["box"]:
            self.assertIsInstance(coord, int)
            self.assertGreaterEqual(coord, 0)

        # Check normalized box format [nx, ny, nw, nh]
        self.assertEqual(len(face["box_normalized"]), 4)
        for n_coord in face["box_normalized"]:
            self.assertTrue(0.0 <= n_coord <= 1.0)

        self.assertEqual(face["verdict"], "REAL")
        self.assertIn("REAL", face["label"])
        self.assertLess(face["detection_score"], 35.0, "Genuine face score should be below ambiguous threshold")
        self.assertIsNotNone(face["confidence"])

    def test_05_analyze_frame_inconclusive_face_tracking(self):
        """
        Test that ambiguous video frame produces INCONCLUSIVE / AMBER state
        and does NOT falsely fabricate a confirmed REAL or confirmed FAKE verdict.
        """
        video_name = "test_trace_vid_01_id1_0007.mp4"
        video_path = os.path.join(self.upload_dir, video_name)
        if not os.path.exists(video_path):
            self.skipTest(f"Sample video {video_name} not found")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret, "Failed to read test frame from video")

        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf).decode('utf-8')

        res = self.client.post('/api/monitor/analyze-frame', json={'frame': b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("verdict"), "INCONCLUSIVE")
        self.assertIn("faces", data)
        self.assertGreater(len(data["faces"]), 0)

        # Check that primary face is correctly recognized as INCONCLUSIVE (amber)
        face = data["faces"][0]
        self.assertEqual(face["verdict"], "INCONCLUSIVE")
        # Confidence must be None for uncertain predictions (no fabricated certainty)
        self.assertIsNone(face["confidence"])
        self.assertGreaterEqual(face["uncertainty"], 0.5)

    def test_06_analyze_frame_without_faces(self):
        """
        Test that an image with no faces does not crash or fabricate false face detections.
        """
        # Create a plain solid frame with a geometric shape (no human face)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 50, (0, 255, 0), -1)

        _, buf = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buf).decode('utf-8')

        res = self.client.post('/api/monitor/analyze-frame', json={'frame': b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data.get("status"), "success")
        self.assertEqual(data.get("verdict"), "NO_FACE")
        self.assertEqual(data.get("faces_count"), 0)
        self.assertEqual(len(data.get("faces", [])), 0)

    def test_07_invalid_frame_data(self):
        """Test error handling on missing or corrupt frame data."""
        res_empty = self.client.post('/api/monitor/analyze-frame', json={})
        self.assertEqual(res_empty.status_code, 400)

        # Corrupt / tiny image
        res_tiny = self.client.post('/api/monitor/analyze-frame', json={'frame': 'data:image/jpeg;base64,1234'})
        self.assertEqual(res_tiny.status_code, 200)
        self.assertEqual(res_tiny.get_json().get("verdict"), "INVALID_FRAME")

    def test_08_stop_monitoring_session(self):
        """Test session stop and stats finalization."""
        # Start session
        res_start = self.client.post('/api/monitor/start-session', json={
            "user_id": "test_auditor",
            "user_name": "Audit Inspector"
        })
        session_id = res_start.get_json().get("session_id")

        # Stop session with sample stats
        res_stop = self.client.post('/api/monitor/stop-session', json={
            "session_id": session_id,
            "stats": {
                "frames_received": 10,
                "frames_analyzed": 9,
                "frames_skipped": 1,
                "deepfake_detections": 0,
                "average_confidence": 88.5,
                "average_latency_ms": 42.0,
                "final_verdict": "REAL",
                "final_risk": "LOW"
            },
            "events": [
                {
                    "verdict": "REAL",
                    "detection_score": 12.5,
                    "risk_level": "LOW",
                    "latency_ms": 38.0,
                    "trigger_reason": "Baseline check"
                }
            ]
        })
        self.assertEqual(res_stop.status_code, 200)
        self.assertEqual(res_stop.get_json().get("status"), "success")


if __name__ == "__main__":
    unittest.main()
