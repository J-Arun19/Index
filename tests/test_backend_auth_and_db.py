import sys
import uuid
import unittest

sys.path.insert(0, r"d:\Download\satya-vision-guard-main\satya-vision-guard-main")

from backend.services.supabase_service import SupabaseService
from backend.services.database_service import DatabaseService
from backend.core.database import get_analyses_by_user, save_analysis, init_db

class TestBackendAuthAndDb(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connected = init_db()

    def test_uuid_user_history_lookup(self):
        """Verify get_analyses_by_user accepts UUID without ValueError"""
        test_uuid = str(uuid.uuid4())
        try:
            history = get_analyses_by_user(test_uuid)
            self.assertIsInstance(history, list)
        except ValueError as e:
            self.fail(f"get_analyses_by_user crashed on UUID with ValueError: {e}")

    def test_integer_user_history_lookup(self):
        """Verify get_analyses_by_user accepts legacy integer user ID"""
        history = get_analyses_by_user(1)
        self.assertIsInstance(history, list)

    def test_save_analysis_with_uuid(self):
        """Verify save_analysis routes UUID to owner_id without foreign key error"""
        if not getattr(self, "connected", False):
            self.skipTest("Supabase backend not reachable")
        test_analysis_id = "test_analysis_" + uuid.uuid4().hex[:8]
        test_uuid = str(uuid.uuid4())
        success = save_analysis(
            analysis_id=test_analysis_id,
            user_id=test_uuid,
            user_name="Test UUID User",
            file_name="test.jpg",
            media_type="image",
            prediction="REAL",
            confidence=95.0,
            report_path=None
        )
        self.assertTrue(success, "save_analysis with UUID user_id should succeed")

        # Verify it was saved successfully without crashing
        client = SupabaseService.get_client()
        res = client.table("analysis_history").select("*").eq("id", test_analysis_id).execute()
        self.assertTrue(res.data and len(res.data) > 0)

        # Clean up test row
        client.table("analysis_history").delete().eq("id", test_analysis_id).execute()

if __name__ == "__main__":
    unittest.main()
