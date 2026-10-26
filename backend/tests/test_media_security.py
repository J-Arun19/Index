import os
import unittest
from flask import Flask
from backend.config import Config
from backend.api.upload import upload_bp, set_upload_owner
from backend.api.analysis import analysis_bp
from backend.utils.security import generate_signed_media_token


class MediaSecurityTestCase(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        self.app.register_blueprint(upload_bp, url_prefix='/api')
        self.app.register_blueprint(analysis_bp, url_prefix='/api')
        self.client = self.app.test_client()

    def test_no_glob_by_original_filename(self):
        """
        SECURITY TEST 1: Requesting upload by naked original filename without unique file_id
        must be rejected with HTTP 400.
        """
        res = self.client.get('/api/analysis/uploads/photo.jpg')
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertTrue("disallowed" in data['error'].lower() or "unique file id" in data['error'].lower())

    def test_cross_user_same_named_upload_isolation(self):
        """
        SECURITY TEST 2:
        User A uploads 'evidence.jpg'.
        User B uploads 'evidence.jpg'.
        User B cannot fetch User A's upload; User A cannot fetch User B's upload.
        Both are isolated by unique file IDs and ownership.
        """
        upload_dir = Config.UPLOAD_FOLDER
        os.makedirs(upload_dir, exist_ok=True)

        file_a_id = "11111111-aaaa-4444-8888-111111111111"
        file_b_id = "22222222-bbbb-4444-8888-222222222222"

        filename_a = f"{file_a_id}_evidence.jpg"
        filename_b = f"{file_b_id}_evidence.jpg"

        content_a = b"USER_A_PRIVATE_IMAGE_DATA_12345"
        content_b = b"USER_B_PRIVATE_IMAGE_DATA_67890"

        with open(os.path.join(upload_dir, filename_a), 'wb') as f:
            f.write(content_a)
        with open(os.path.join(upload_dir, filename_b), 'wb') as f:
            f.write(content_b)

        set_upload_owner(file_a_id, "user_alpha")
        set_upload_owner(file_b_id, "user_beta")

        # 1. User B attempts to access User A's same-named upload using User B's identity
        res_b_attacks_a = self.client.get(
            f'/api/analysis/uploads/{filename_a}',
            headers={'X-User-ID': 'user_beta'}
        )
        self.assertEqual(res_b_attacks_a.status_code, 403)
        self.assertIn("Access denied", res_b_attacks_a.get_json()['error'])

        # 2. User A attempts to access User B's same-named upload using User A's identity
        res_a_attacks_b = self.client.get(
            f'/api/analysis/uploads/{filename_b}',
            headers={'X-User-ID': 'user_alpha'}
        )
        self.assertEqual(res_a_attacks_b.status_code, 403)
        self.assertIn("Access denied", res_a_attacks_b.get_json()['error'])

        # 3. Unauthenticated request to User A's upload without signed token is rejected
        res_anon_attacks_a = self.client.get(f'/api/analysis/uploads/{filename_a}')
        self.assertEqual(res_anon_attacks_a.status_code, 401)

        # 4. User A successfully accesses User A's file
        res_a_legit = self.client.get(
            f'/api/analysis/uploads/{filename_a}',
            headers={'X-User-ID': 'user_alpha'}
        )
        self.assertEqual(res_a_legit.status_code, 200)
        self.assertEqual(res_a_legit.data, content_a)

        # 5. User B successfully accesses User B's file
        res_b_legit = self.client.get(
            f'/api/analysis/uploads/{filename_b}',
            headers={'X-User-ID': 'user_beta'}
        )
        self.assertEqual(res_b_legit.status_code, 200)
        self.assertEqual(res_b_legit.data, content_b)

    def test_signed_token_access_and_cross_token_tampering(self):
        """
        SECURITY TEST 3:
        Signed URLs grant access only to the matching file ID.
        User B cannot use a token signed for file B to access file A.
        """
        upload_dir = Config.UPLOAD_FOLDER
        file_a_id = "33333333-aaaa-4444-8888-333333333333"
        file_b_id = "44444444-bbbb-4444-8888-444444444444"

        filename_a = f"{file_a_id}_selfie.jpg"
        filename_b = f"{file_b_id}_selfie.jpg"

        content_a = b"IMAGE_A_BYTES"
        content_b = b"IMAGE_B_BYTES"

        with open(os.path.join(upload_dir, filename_a), 'wb') as f:
            f.write(content_a)
        with open(os.path.join(upload_dir, filename_b), 'wb') as f:
            f.write(content_b)

        set_upload_owner(file_a_id, "user_alpha")
        set_upload_owner(file_b_id, "user_beta")

        token_a = generate_signed_media_token(file_a_id, user_id="user_alpha")
        token_b = generate_signed_media_token(file_b_id, user_id="user_beta")

        # Token A correctly serves file A
        res_a = self.client.get(f'/api/analysis/uploads/{filename_a}?token={token_a}')
        self.assertEqual(res_a.status_code, 200)
        self.assertEqual(res_a.data, content_a)

        # Token B cannot serve file A (signature mismatch)
        res_tamper = self.client.get(f'/api/analysis/uploads/{filename_a}?token={token_b}')
        self.assertIn(res_tamper.status_code, (401, 403))


if __name__ == '__main__':
    unittest.main()
