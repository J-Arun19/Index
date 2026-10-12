import os
from backend.config import Config

try:
    import filetype
except ImportError:
    filetype = None

def guess_mime_and_ext(file_storage_or_path):
    """
    Pure Python fallback magic-byte detector for image and video formats.
    """
    header = b""
    if isinstance(file_storage_or_path, str):
        if not os.path.exists(file_storage_or_path) or os.path.getsize(file_storage_or_path) == 0:
            return None, None
        with open(file_storage_or_path, "rb") as f:
            header = f.read(264)
    else:
        header = file_storage_or_path.read(264)
        file_storage_or_path.seek(0)

    if filetype is not None and header:
        kind = filetype.guess(header)
        if kind is not None:
            return kind.mime, kind.extension.lower()

    if len(header) < 4:
        return None, None

    # Magic byte checks
    if header.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg', 'jpg'
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png', 'png'
    if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'image/gif', 'gif'
    if header.startswith(b'RIFF') and len(header) >= 12 and header[8:12] == b'WEBP':
        return 'image/webp', 'webp'
    if header.startswith(b'BM'):
        return 'image/bmp', 'bmp'

    # Video magic bytes
    if len(header) >= 8 and (header[4:8] == b'ftyp' or header[4:8] == b'moov' or header[4:8] == b'wide'):
        if len(header) >= 12:
            brand = header[8:12]
            if brand in [b'qt  ', b'moov']:
                return 'video/quicktime', 'mov'
        return 'video/mp4', 'mp4'
    if header.startswith(b'\x1a\x45\xdf\xa3'):
        return 'video/webm', 'webm'
    if header.startswith(b'RIFF') and len(header) >= 12 and header[8:12] == b'AVI ':
        return 'video/x-msvideo', 'avi'

    return None, None

def detect_media_type(file_storage_or_path):
    """
    Single source of truth for media type detection.
    Inspects content signature/magic bytes and returns (media_type, mime, ext)
    where media_type is 'image', 'video', or None (unsupported/corrupted).
    """
    try:
        mime, ext = guess_mime_and_ext(file_storage_or_path)
    finally:
        if not isinstance(file_storage_or_path, str) and hasattr(file_storage_or_path, 'seek'):
            file_storage_or_path.seek(0)

    if not mime:
        return None, None, None

    if mime.startswith('image/'):
        clean_ext = 'jpg' if ext == 'jpeg' else (ext or 'jpg')
        return 'image', mime, clean_ext
    elif mime.startswith('video/'):
        clean_ext = 'mov' if ext == 'quicktime' else (ext or 'mp4')
        return 'video', mime, clean_ext

    return None, mime, ext

def get_file_extension(filename):
    if '.' not in filename:
        return ''
    return filename.rsplit('.', 1)[1].lower()

def is_allowed_extension(filename):
    ext = get_file_extension(filename)
    return ext in Config.ALLOWED_EXTENSIONS

def validate_media_file(file_storage):
    """
    Validates a Flask FileStorage object:
    1. Checks filename extension.
    2. Reads magic bytes to guess type via detect_media_type.
    3. Checks if the detected mime type is a supported image or video.
    4. Resets the stream pointer to 0.
    """
    filename = file_storage.filename
    if not filename:
        return False, None, "No file selected", None
        
    # 1. Check extension first
    if not is_allowed_extension(filename):
        return False, None, f"File extension not allowed. Supported formats: images ({', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}), videos ({', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)})", None

    # 2. Check content signature via detect_media_type
    try:
        media_type, mime, ext = detect_media_type(file_storage)
        if mime is None or media_type is None:
            # Check if file is empty
            file_storage.seek(0, os.SEEK_END)
            size = file_storage.tell()
            file_storage.seek(0)
            if size == 0:
                return False, None, "File is empty", None
            return False, None, "Could not determine file type from content signature", None
            
        # 3. Check allowed format lists
        if media_type == 'image':
            ext_to_check = ext.lower() if ext else ''
            if ext_to_check not in Config.ALLOWED_IMAGE_EXTENSIONS:
                filename_ext = get_file_extension(filename)
                if filename_ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
                    return False, None, f"Image format '{ext}' is not supported. Supported: {', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}", mime
            return True, 'image', None, mime
            
        elif media_type == 'video':
            ext_to_check = ext.lower() if ext else ''
            if ext_to_check not in Config.ALLOWED_VIDEO_EXTENSIONS:
                filename_ext = get_file_extension(filename)
                if filename_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
                    return False, None, f"Video format '{ext}' (filename extension '{filename_ext}') is not supported. Supported: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}", mime
            return True, 'video', None, mime

        return False, None, f"Unsupported media category. Content type detected: {mime}", mime

    except Exception as e:
        return False, None, f"Error validating file content: {str(e)}", None
