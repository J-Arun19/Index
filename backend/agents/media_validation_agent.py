import os
import cv2
try:
    import filetype
except ImportError:
    filetype = None
from PIL import Image
import numpy as np
from backend.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger("MediaValidationAgent")

class MediaValidationAgent:
    def validate(self, file_path: str, expected_media_type: str = None) -> dict:
        """
        Validates file path:
        1. Checks file existence & size (integrity).
        2. Inspects magic numbers / mime types.
        3. Confirms compatibility with the allowed image/video formats.
        4. Decodes image/video to verify content decodability and dimensions.
        """
        result = {
            "valid": True,
            "media_type": None,
            "checks": {
                "file_exists": False,
                "non_empty": False,
                "signature_valid": False,
                "decodable": False,
                "dimensions_valid": False,
                "frame_validation": False
            },
            "errors": [],
            "warnings": [],
            "summary": ""
        }

        # 1. File existence check
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            result["valid"] = False
            result["errors"].append(f"File not found: {file_path}")
            result["summary"] = "Media validation failed."
            return result
        
        result["checks"]["file_exists"] = True

        # 2. File size check
        size = os.path.getsize(file_path)
        if size == 0:
            result["valid"] = False
            result["errors"].append("File is empty / corrupted upload")
            result["summary"] = "Media validation failed."
            return result

        result["checks"]["non_empty"] = True

        # 3. Magic number & Signature check
        try:
            from backend.core.validator import guess_mime_and_ext
            mime, ext = guess_mime_and_ext(file_path)

            if mime is None:
                result["valid"] = False
                result["errors"].append("Could not determine file type from magic number signature")
                result["summary"] = "Media validation failed."
                return result

            is_image = mime.startswith('image/')
            is_video = mime.startswith('video/')

            if not (is_image or is_video):
                result["valid"] = False
                result["errors"].append(f"Unsupported media category. Detected mime: {mime}")
                result["summary"] = "Media validation failed."
                return result

            detected_type = "image" if is_image else "video"
            result["media_type"] = detected_type

            # Check if signature matches expected media type
            if expected_media_type:
                expected_clean = expected_media_type.lower()
                if detected_type != expected_clean:
                    result["errors"].append(f"Expected media type '{expected_clean}', but detected '{detected_type}'")

            # Check supported extensions
            if is_image:
                allowed_exts = Config.ALLOWED_IMAGE_EXTENSIONS
                check_ext = ext if ext != 'jpeg' else 'jpg'
            else:
                allowed_exts = Config.ALLOWED_VIDEO_EXTENSIONS
                check_ext = ext if ext != 'quicktime' else 'mov'

            filename = os.path.basename(file_path)
            _, file_ext = os.path.splitext(filename)
            file_ext_clean = file_ext[1:].lower() if file_ext else ""

            # Check if format is allowed
            if check_ext not in allowed_exts and file_ext_clean not in allowed_exts:
                result["errors"].append(f"Format extension '{ext}' is not supported. Supported: {', '.join(allowed_exts)}")

            # Check for extension masquerading/spoofing
            if file_ext_clean and file_ext_clean != check_ext:
                # Allow jpg/jpeg mismatch
                if not (file_ext_clean in ['jpg', 'jpeg'] and check_ext in ['jpg', 'jpeg']):
                    result["errors"].append(f"Filename extension '{file_ext_clean}' does not match file signature format '{ext}'")

            if result["errors"]:
                result["valid"] = False
                result["summary"] = "Media validation failed."
                return result

            result["checks"]["signature_valid"] = True

        except Exception as e:
            logger.error(f"Error during signature validation: {str(e)}")
            result["valid"] = False
            result["errors"].append(f"Signature validation error: {str(e)}")
            result["summary"] = "Media validation failed."
            return result

        # 4. Content Decoding & Dimensions checks
        if detected_type == "image":
            try:
                # OpenCV decoding
                cv2_img = cv2.imread(file_path)
                if cv2_img is None:
                    result["errors"].append("OpenCV failed to decode image")
                else:
                    h, w, c = cv2_img.shape
                    if h <= 0 or w <= 0:
                        result["errors"].append(f"Invalid image dimensions detected: {w}x{h}")
                    else:
                        result["checks"]["dimensions_valid"] = True

                # PIL decoding
                pil_ok = False
                try:
                    with Image.open(file_path) as pil_img:
                        pil_img.verify()
                    # Reopen and load to check for truncation or pixel level corruption
                    with Image.open(file_path) as pil_img:
                        pil_img.load()
                    pil_ok = True
                except Exception as pil_err:
                    result["errors"].append(f"PIL failed to decode/load image: {str(pil_err)}")

                if not result["errors"]:
                    result["checks"]["decodable"] = True
                    result["checks"]["frame_validation"] = True

            except Exception as e:
                result["errors"].append(f"Image decoding exception: {str(e)}")

        elif detected_type == "video":
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    result["errors"].append("OpenCV VideoCapture failed to open video file")
                else:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if frame_count <= 0:
                        result["errors"].append(f"Video has invalid frame count: {frame_count}")
                    if fps <= 0:
                        result["errors"].append(f"Video has invalid FPS: {fps}")
                    if width <= 0 or height <= 0:
                        result["errors"].append(f"Video has invalid dimensions: {width}x{height}")

                    if not result["errors"]:
                        result["checks"]["dimensions_valid"] = True

                    # Try to decode multiple frames distributed across the video
                    readable_frames = 0
                    sample_indices = []
                    if frame_count > 0:
                        num_samples = min(5, frame_count)
                        if num_samples > 1:
                            sample_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int).tolist()
                        else:
                            sample_indices = [0]

                    decoded_at_least_one = False
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            readable_frames += 1
                            decoded_at_least_one = True

                    cap.release()

                    if decoded_at_least_one:
                        result["checks"]["decodable"] = True
                        if readable_frames < len(sample_indices):
                            result["warnings"].append(f"Only {readable_frames}/{len(sample_indices)} sampled frames could be decoded.")
                        result["checks"]["frame_validation"] = True
                    else:
                        result["errors"].append("Video contains no readable frames or is corrupted")

            except Exception as e:
                result["errors"].append(f"Video decoding exception: {str(e)}")

        # Final check status consolidation
        if result["errors"]:
            result["valid"] = False
            result["summary"] = "Media validation failed."
        else:
            result["valid"] = True
            result["summary"] = "Media passed validation."

        return result
