import os
import time
import hmac
import hashlib
import base64
from typing import Tuple, Optional
from backend.config import Config

SECRET_KEY = getattr(Config, 'SECRET_KEY', 'satyadrishti_secure_secret_key_2026').encode('utf-8')


def generate_signed_media_token(file_id: str, user_id: Optional[str] = None, expires_in: int = 86400) -> str:
    """
    Generates an HMAC-SHA256 signature token granting access to a specific file_id.
    """
    expires_at = int(time.time()) + expires_in
    uid = str(user_id or "anonymous")
    msg = f"{file_id}:{uid}:{expires_at}".encode('utf-8')
    sig = hmac.new(SECRET_KEY, msg, hashlib.sha256).hexdigest()
    token_str = f"{expires_at}.{uid}.{sig}"
    return base64.urlsafe_b64encode(token_str.encode('utf-8')).decode('utf-8')


def verify_signed_media_token(token: str, file_id: str) -> Tuple[bool, Optional[str]]:
    """
    Verifies that a signed token matches the given file_id and has not expired.
    Returns (is_valid, user_id).
    """
    if not token:
        return False, None
    try:
        decoded = base64.urlsafe_b64decode(token.encode('utf-8')).decode('utf-8')
        parts = decoded.split('.')
        if len(parts) != 3:
            return False, None
        expires_at_str, uid, sig = parts
        expires_at = int(expires_at_str)
        if time.time() > expires_at:
            return False, None
        msg = f"{file_id}:{uid}:{expires_at}".encode('utf-8')
        expected_sig = hmac.new(SECRET_KEY, msg, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected_sig):
            return True, uid
        return False, None
    except Exception:
        return False, None
