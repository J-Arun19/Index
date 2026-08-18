import os
from PIL import Image
from backend.utils.logger import setup_logger

logger = setup_logger("MetadataAgent")

class MetadataAgent:
    def analyze(self, file_path: str) -> dict:
        """
        Scans file metadata, container tags, and headers for editing software signatures.
        Returns a structured dict of findings.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        software_detected = None
        suspicious_indicators = []
        metadata_present = False
        
        # Basic size checks
        size = os.path.getsize(file_path)
        if size > 50 * 1024 * 1024:  # 50MB
            suspicious_indicators.append("Abnormally large file size for web media")

        # 1. Read first 64KB of the file to search for software headers/metadata strings
        try:
            with open(file_path, "rb") as f:
                head = f.read(65536)
                metadata_present = len(head) > 0
                
                # Search for signatures of known editing software in binary headers
                softwares = {
                    b"photoshop": "Adobe Photoshop",
                    b"adobe": "Adobe Creative Cloud",
                    b"gimp": "GIMP",
                    b"capcut": "CapCut",
                    b"premiere": "Adobe Premiere",
                    b"imovie": "iMovie",
                    b"ffmpeg": "FFmpeg",
                    b"canva": "Canva"
                }
                for sig, name in softwares.items():
                    if sig in head.lower():
                        software_detected = name
                        break
        except Exception as e:
            logger.warning(f"Failed to scan file headers: {e}")

        # 2. Image EXIF/Info inspection
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            try:
                with Image.open(file_path) as img:
                    info = img.info or {}
                    if info:
                        metadata_present = True
                    # Search in info dict keys/values
                    for k, v in info.items():
                        v_str = str(v).lower()
                        if "adobe" in v_str or "photoshop" in v_str:
                            software_detected = "Adobe Software Suite"
                        elif "gimp" in v_str:
                            software_detected = "GIMP"
                            
                    # EXIF data check
                    exif = img.getexif()
                    if exif:
                        metadata_present = True
                        for tag, val in exif.items():
                            val_str = str(val).lower()
                            if "adobe" in val_str or "photoshop" in val_str:
                                software_detected = "Adobe Software Suite"
                              # Line comments match indent
                            elif "gimp" in val_str:
                                software_detected = "GIMP"
            except Exception as e:
                logger.warning(f"PIL Image info scanning failed: {e}")

        # Determine anomalies based on signature presence
        anomalies_detected = False
        if software_detected:
            # Note: Editing software is NOT proof of deepfake, just an indicator
            anomalies_detected = True
            suspicious_indicators.append(f"Editing software signature detected: {software_detected}")

        # Summary message
        if software_detected:
            summary = f"Metadata indicates file was processed with {software_detected}."
        else:
            summary = "No editing software signatures or metadata anomalies detected."

        return {
            "status": "Success",
            "metadata_present": metadata_present,
            "anomalies_detected": anomalies_detected,
            "software_tag": software_detected,
            "suspicious_indicators": suspicious_indicators,
            "summary": summary
        }
