import os
import filetype
from backend.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger("MediaValidationAgent")

class MediaValidationAgent:
    def validate(self, file_path: str) -> dict:
        """
        Validates file path:
        1. Checks file existence & size (integrity).
        2. Inspects magic numbers / mime types.
        3. Confirms compatibility with the allowed image/video formats.
        """
        if not os.path.exists(file_path):
            return {
                "status": "invalid",
                "media_type": None,
                "issues": [f"File not found: {file_path}"]
            }

        size = os.path.getsize(file_path)
        if size == 0:
            return {
                "status": "invalid",
                "media_type": None,
                "issues": ["File is empty / corrupted upload"]
            }

        try:
            kind = filetype.guess(file_path)
            if kind is None:
                return {
                    "status": "invalid",
                    "media_type": None,
                    "issues": ["Could not determine file type from magic number signature"]
                }

            mime = kind.mime
            ext = kind.extension.lower()

            is_image = mime.startswith('image/')
            is_video = mime.startswith('video/')

            if not (is_image or is_video):
                return {
                    "status": "invalid",
                    "media_type": None,
                    "issues": [f"Unsupported media category. Detected mime: {mime}"]
                }

            detected_type = "image" if is_image else "video"

            # Normalize check extensions
            if is_image:
                allowed_exts = Config.ALLOWED_IMAGE_EXTENSIONS
                check_ext = ext if ext != 'jpeg' else 'jpg'
            else:
                allowed_exts = Config.ALLOWED_VIDEO_EXTENSIONS
                check_ext = ext if ext != 'quicktime' else 'mov'

            issues = []
            filename = os.path.basename(file_path)
            _, file_ext = os.path.splitext(filename)
            file_ext_clean = file_ext[1:].lower() if file_ext else ""

            # Check if format is allowed
            if check_ext not in allowed_exts and file_ext_clean not in allowed_exts:
                issues.append(f"Format extension '{ext}' is not supported. Supported: {', '.join(allowed_exts)}")

            # Check for extension masquerading
            if file_ext_clean and file_ext_clean != check_ext:
                if not (file_ext_clean in ['jpg', 'jpeg'] and check_ext in ['jpg', 'jpeg']):
                    issues.append(f"Filename extension '{file_ext_clean}' does not match file signature format '{ext}'")

            if issues:
                return {
                    "status": "invalid",
                    "media_type": detected_type,
                    "issues": issues
                }

            return {
                "status": "valid",
                "media_type": detected_type,
                "issues": []
            }

        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return {
                "status": "invalid",
                "media_type": None,
                "issues": [f"Validation exception: {str(e)}"]
            }
