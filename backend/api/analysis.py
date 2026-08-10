import os
from flask import Blueprint, request, jsonify, send_from_directory
from backend.core.orchestrator import Orchestrator
from backend.core.database import get_analyses_by_user
from backend.config import Config

analysis_bp = Blueprint('analysis', __name__)
orchestrator = Orchestrator()

@analysis_bp.route('/analyze', methods=['POST'])
def analyze_media():
    # 1. Parse request parameters
    data = request.get_json() or {}
    file_id = data.get('file_id')
    media_type = data.get('media_type')
    user = data.get('user')  # Optional user details: {"id": 1, "name": "...", "email": "..."}
    
    # 2. Check required parameters
    if not file_id:
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: file_id'
        }), 400
        
    if not media_type:
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: media_type'
        }), 400
        
    # 3. Invoke Orchestrator workflow
    try:
        result = orchestrator.run_orchestration(file_id, media_type, user=user)
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 404
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Orchestrator analysis failed: {str(e)}'
        }), 500

@analysis_bp.route('/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: user_id'
        }), 400
    try:
        history = get_analyses_by_user(user_id)
        return jsonify({
            'status': 'success',
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Failed to retrieve analysis history: {str(e)}'
        }), 500

@analysis_bp.route('/analysis/report/<analysis_id>', methods=['GET'])
def get_report(analysis_id):
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(backend_dir, 'reports')
    filename = f"report_{analysis_id}.pdf"
    if not os.path.exists(os.path.join(reports_dir, filename)):
        return jsonify({
            'status': 'error',
            'error': 'Report not found'
        }), 404
    return send_from_directory(reports_dir, filename, as_attachment=True)

@analysis_bp.route('/analysis/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    if not os.path.exists(os.path.join(Config.UPLOAD_FOLDER, filename)):
        return jsonify({
            'status': 'error',
            'error': 'File not found'
        }), 404
    return send_from_directory(Config.UPLOAD_FOLDER, filename)
