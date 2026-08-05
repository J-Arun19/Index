from flask import Blueprint, request, jsonify
from backend.core.orchestrator import Orchestrator

analysis_bp = Blueprint('analysis', __name__)
orchestrator = Orchestrator()

@analysis_bp.route('/analyze', methods=['POST'])
def analyze_media():
    # 1. Parse request parameters
    data = request.get_json() or {}
    file_id = data.get('file_id')
    media_type = data.get('media_type')
    
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
        result = orchestrator.run_orchestration(file_id, media_type)
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
