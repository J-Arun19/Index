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
    if len(header) >= 12 and header[4:8] == b'ftyp':
        brand = header[8:12]
        if brand in [b'qt  ', b'moov']:
            return 'video/quicktime', 'mov'
        return 'video/mp4', 'mp4'
    if header.startswith(b'\x1a\x45\xdf\xa3'):
        return 'video/webm', 'webm'
    if header.startswith(b'RIFF') and len(header) >= 12 and header[8:12] == b'AVI ':
        return 'video/x-msvideo', 'avi'

    return None, None

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
    2. Reads magic bytes to guess type.
    3. Checks if the detected mime type is a supported image or video.
    4. Resets the stream pointer to 0.
    """
    filename = file_storage.filename
    if not filename:
        return False, None, "No file selected", None
        
    # 1. Check extension first
    if not is_allowed_extension(filename):
        return False, None, f"File extension not allowed. Supported formats: images ({', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}), videos ({', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)})", None

    # 2. Check content signature (magic numbers)
    try:
        mime, ext = guess_mime_and_ext(file_storage)
        if mime is None:
            # Let's check if the file is empty
            file_storage.seek(0, os.SEEK_END)
            size = file_storage.tell()
            file_storage.seek(0)
            if size == 0:
                return False, None, "File is empty", None
            return False, None, "Could not determine file type from content signature", None
            
        # 3. Determine if it's a valid image or video
        is_image = mime.startswith('image/')
        is_video = mime.startswith('video/')
        
        if not (is_image or is_video):
            return False, None, f"Unsupported media category. Content type detected: {mime}", mime

        # Map to category and check allowed extensions/types
        if is_image:
            ext_to_check = ext.lower()
            if ext_to_check == 'jpeg':
                ext_to_check = 'jpg'
                
            if ext_to_check not in Config.ALLOWED_IMAGE_EXTENSIONS:
                filename_ext = get_file_extension(filename)
                if filename_ext not in Config.ALLOWED_IMAGE_EXTENSIONS:
                    return False, None, f"Image format '{ext}' is not supported. Supported: {', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}", mime
            return True, 'image', None, mime
            
        elif is_video:
            ext_to_check = ext.lower()
            if ext_to_check == 'quicktime':
                ext_to_check = 'mov'
            
            if ext_to_check not in Config.ALLOWED_VIDEO_EXTENSIONS:
                filename_ext = get_file_extension(filename)
                if filename_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
                    return False, None, f"Video format '{ext}' (filename extension '{filename_ext}') is not supported. Supported: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}", mime
            return True, 'video', None, mime

    except Exception as e:
        return False, None, f"Error validating file content: {str(e)}", None
