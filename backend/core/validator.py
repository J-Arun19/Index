import os
import filetype
from backend.config import Config

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
    2. Reads first 2048 bytes to guess type via magic numbers (filetype).
    3. Checks if the detected mime type is a supported image or video.
    4. Resets the stream pointer to 0.
    
    Returns:
        tuple: (is_valid, media_type, error_message, detected_mime)
        media_type will be 'image', 'video', or None
    """
    filename = file_storage.filename
    if not filename:
        return False, None, "No file selected", None
        
    # 1. Check extension first
    if not is_allowed_extension(filename):
        return False, None, f"File extension not allowed. Supported formats: images ({', '.join(Config.ALLOWED_IMAGE_EXTENSIONS)}), videos ({', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)})", None

    # 2. Check content signature (magic numbers)
    try:
        # Read header to guess file type
        header = file_storage.read(2048)
        # Reset file pointer immediately so subsequent reads get the full file
        file_storage.seek(0)
        
        kind = filetype.guess(header)
        if kind is None:
            # Let's check if the file is empty
            if len(header) == 0:
                return False, None, "File is empty", None
            return False, None, "Could not determine file type from content signature", None
            
        mime = kind.mime
        ext = kind.extension
        
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
                # If filetype reports jpeg/png but it's represented differently, double check
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
