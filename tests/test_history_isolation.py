import unittest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app

class TestHistoryAndReportIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def test_unauthenticated_history_returns_empty(self):
        """Unauthenticated /api/history query must return empty list, never leaking other users' records."""
        res = self.client.get('/api/history')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('history'), [])

    def test_invalid_user_id_returns_empty(self):
        """Query with invalid/null/undefined user_id must return empty list."""
        for param in ['null', 'undefined', 'None', '', '   ']:
            res = self.client.get(f'/api/history?user_id={param}')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get('status'), 'success')
            self.assertEqual(data.get('history'), [], f"Failed for user_id={param}")

    def test_download_report_without_analysis_returns_error(self):
        """Downloading report without valid analysis must explicitly return 'You can't download because no analysis is done'."""
        for invalid_id in ['null', 'undefined', 'nonexistent_id_12345']:
            res = self.client.get(f'/api/analysis/report/{invalid_id}')
            self.assertIn(res.status_code, [400, 404])
            data = res.get_json()
            self.assertEqual(data.get('error'), "You can't download because no analysis is done")

if __name__ == '__main__':
    unittest.main()
