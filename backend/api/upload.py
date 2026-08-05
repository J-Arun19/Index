import os
import uuid
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from backend.config import Config
from backend.core.validator import validate_media_file

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    # 1. Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'error': 'No file part in the request'
        }), 400
        
    file = request.files['file']
    
    # 2. Check if the user submitted an empty form / no file selected
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'error': 'No selected file'
        }), 400
        
    # 3. Validate file using core validator (extension + signature check)
    is_valid, media_type, error_msg, detected_mime = validate_media_file(file)
    
    if not is_valid:
        return jsonify({
            'status': 'error',
            'error': error_msg or 'Invalid file format'
        }), 400
        
    # 4. Generate unique file ID and save the file securely
    try:
        # Create UPLOAD_FOLDER if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        file_id = str(uuid.uuid4())
        original_name = secure_filename(file.filename)
        
        # If secure_filename returns empty (e.g. filename was only non-ascii or special chars)
        if not original_name:
            original_name = f"uploaded_file_{file_id}"
            
        saved_filename = f"{file_id}_{original_name}"
        save_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
        
        # Save file to disk
        file.save(save_path)
        
        # Return success JSON response
        return jsonify({
            'status': 'success',
            'id': file_id,
            'media_type': media_type,
            'filename': original_name,
            'mime_type': detected_mime
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Failed to save file: {str(e)}'
        }), 500
