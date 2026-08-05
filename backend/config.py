import os

class Config:
    # Get absolute path of backend directory
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Upload folder
    UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'uploads')
    
    # Max file size: 100MB (100 * 1024 * 1024 bytes)
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}
    
    # Combined allowed extensions
    ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS)
