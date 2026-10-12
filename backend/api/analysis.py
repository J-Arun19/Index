import os
from flask import Blueprint, request, jsonify, send_from_directory
from backend.core.orchestrator import Orchestrator
from backend.core.database import get_analyses_by_user, get_analysis_by_id, get_agent_logs
from backend.config import Config

analysis_bp = Blueprint('analysis', __name__)

_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator

@analysis_bp.route('/analyze', methods=['POST'])
@analysis_bp.route('/analysis', methods=['POST'])
def analyze_media():
    # Handle direct multipart file upload if posted directly
    if request.files and 'file' in request.files:
        from werkzeug.utils import secure_filename
        import uuid
        import json
        from backend.core.validator import validate_media_file

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'error': 'No selected file'}), 400

        is_valid, media_type, error_msg, detected_mime = validate_media_file(file)
        if not is_valid:
            return jsonify({'status': 'error', 'error': error_msg or 'Invalid file format'}), 400

        try:
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            file_id = str(uuid.uuid4())
            original_name = secure_filename(file.filename) or f"uploaded_file_{file_id}"
            saved_filename = f"{file_id}_{original_name}"
            save_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
            file.save(save_path)

            raw_user = request.form.get('user')
            user = json.loads(raw_user) if raw_user else None
            question = request.form.get('question')

            result = get_orchestrator().run_orchestration(file_id, user=user)
            from backend.core.nlp_response import generate_nlp_response
            nlp_res = generate_nlp_response(result.get("fusion_result", {}), question=question)
            result["natural_language_answer"] = nlp_res
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Upload and analysis failed: {str(e)}'}), 500

    # Handle standard JSON request with file_id
    data = request.get_json() or {}
    file_id = data.get('file_id')
    media_type = data.get('media_type')  # Optional; Orchestrator auto-detects true media type
    user = data.get('user')  # Optional user details: {"id": 1, "name": "...", "email": "..."}
    
    # Check required parameter
    if not file_id:
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: file_id'
        }), 400
        
    question = data.get('question')

    # Invoke Orchestrator workflow with automatic media routing
    try:
        result = get_orchestrator().run_orchestration(file_id, media_type=media_type, user=user)
        from backend.core.nlp_response import generate_nlp_response
        nlp_res = generate_nlp_response(result.get("fusion_result", {}), question=question)
        result["natural_language_answer"] = nlp_res
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 404
        
    except ValueError as e:
        if hasattr(e, "validation_result"):
            return jsonify({
                "success": False,
                "error": "MEDIA_VALIDATION_FAILED",
                "message": str(e),
                "validation": e.validation_result
            }), 400
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Orchestrator analysis failed: {str(e)}'
        }), 500

@analysis_bp.route('/inquire', methods=['POST'])
def inquire_media():
    """
    Answers natural language / conversational questions grounded strictly
    in the SatyaDrishti forensic inference result.
    """
    data = request.get_json() or {}
    question = data.get('question', 'Is this a deepfake?')
    analysis_result = data.get('analysis_result')
    file_id = data.get('file_id')

    from backend.core.nlp_response import generate_nlp_response

    if analysis_result:
        res = generate_nlp_response(analysis_result, question=question)
        return jsonify(res), 200
    elif file_id:
        file_path = get_orchestrator().locate_file(file_id)
        if not file_path:
            return jsonify({'status': 'error', 'error': 'File not found'}), 404
        # Run image or video analysis
        from backend.agents.image_agent import ImageAgent
        img_agent = ImageAgent()
        result = img_agent.analyze(file_path)
        res = generate_nlp_response(result, question=question)
        return jsonify(res), 200
    else:
        return jsonify({'status': 'error', 'error': 'Missing analysis_result or file_id'}), 400

@analysis_bp.route('/history', methods=['GET'])
def get_history():
    user_id = request.args.get('user_id')
    try:
        if user_id and str(user_id).strip() and str(user_id).lower() not in ('null', 'undefined', 'none', ''):
            history = get_analyses_by_user(user_id)
        else:
            # Never return other users' confidential forensic records when unauthenticated or user_id is missing
            history = []
        return jsonify({
            'status': 'success',
            'history': history
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Failed to retrieve analysis history: {str(e)}'
        }), 500

@analysis_bp.route('/analysis/<analysis_id>', methods=['GET'])
def get_analysis_details(analysis_id):
    row = get_analysis_by_id(analysis_id)
    if not row:
        return jsonify({'status': 'error', 'error': 'Analysis record not found'}), 404
    
    logs = get_agent_logs(analysis_id)
    timeline = []
    agents = []
    for l in logs:
        timeline.append({
            'agent': l.get('agent_name', 'Agent'),
            'status': l.get('status', 'Completed'),
            'duration': float(l.get('duration', 0.0))
        })
        agents.append({
            'name': l.get('agent_name', 'Agent'),
            'summary': l.get('status', 'Completed')
        })
        
    if not agents:
        agents = [
            {'name': 'Validation Agent', 'summary': 'Completed'},
            {'name': f"{'Video' if row.get('media_type') == 'video' else 'Image'} Specialist Agent", 'summary': 'Completed'},
            {'name': 'Confidence Evaluator Agent', 'summary': 'Completed'},
            {'name': 'Fusion Agent', 'summary': 'Completed'}
        ]

    pred = row.get('prediction', 'REAL')
    conf = float(row.get('confidence', 0.0))
    conf_level = 'High' if conf >= 75 else 'Medium' if conf >= 50 else 'Low'
    is_fake = pred in ('DEEPFAKE', 'FAKE')

    return jsonify({
        'status': 'success',
        'analysis': {
            'id': row.get('id'),
            'fileName': row.get('file_name'),
            'fileType': row.get('media_type', 'image'),
            'mediaUrl': f"/api/analysis/uploads/{row.get('file_name')}",
            'verdict': pred,
            'prediction': pred,
            'confidence': round(conf),
            'confidenceLevel': conf_level,
            'fakeProbability': (conf / 100.0) if is_fake else ((100.0 - conf) / 100.0),
            'realProbability': ((100.0 - conf) / 100.0) if is_fake else (conf / 100.0),
            'explanation': f"Consensus was reached with a {pred.lower()} verdict.",
            'reason': f"Consensus was reached with a {pred.lower()} verdict based on multi-agent forensic evaluation.",
            'userFacingStatement': (
                "Deepfake detected / high likelihood of manipulation." if is_fake
                else "Insufficient evidence to determine authenticity reliably." if pred == "UNCERTAIN"
                else "No strong evidence of manipulation was detected."
            ),
            'evidence': [
                f"File analyzed: {row.get('file_name')}",
                f"Consensus verdict: {pred} ({round(conf)}% confidence)",
            ],
            'recommendation': "Forensic safety audit record permanently archived in verified database.",
            'agents': agents,
            'timeline': timeline,
            'createdAt': row.get('timestamp'),
            'reportUrl': f"/api/analysis/report/{analysis_id}.pdf"
        }
    }), 200

@analysis_bp.route('/analysis/report/<analysis_id>', methods=['GET'])
@analysis_bp.route('/analysis/report/<analysis_id>.pdf', methods=['GET'])
def get_report(analysis_id):
    if not analysis_id or str(analysis_id).strip().lower() in ('null', 'undefined', 'none', '', '.pdf'):
        return jsonify({
            'status': 'error',
            'error': "You can't download because no analysis is done"
        }), 400

    if analysis_id.endswith('.pdf'):
        analysis_id = analysis_id[:-4]

    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(backend_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"report_{analysis_id}.pdf"
    local_path = os.path.join(reports_dir, filename)

    if not os.path.exists(local_path):
        # 1. Try downloading from Supabase Storage
        try:
            from backend.services.supabase_service import SupabaseService
            downloaded = SupabaseService.download_report(filename, local_path)
            if not downloaded:
                SupabaseService.download_report(f"reports/{filename}", local_path)
        except Exception:
            pass

        # 2. If still not on disk, generate on the fly from database record
        if not os.path.exists(local_path):
            try:
                row = get_analysis_by_id(analysis_id)
                if row:
                    from backend.utils.report_generator import ReportGenerator
                    pred = row.get("prediction", "REAL")
                    conf = float(row.get("confidence", 85.0))
                    context_dict = {
                        "analysis_id": analysis_id,
                        "user": {"name": row.get("user_name", "Researcher")},
                        "uploaded_file": row.get("file_name", "analysis_media"),
                        "media_type": str(row.get("media_type", "IMAGE")).upper(),
                        "validation_result": {"status": "PASSED", "issues": []},
                        "confidence_result": {"confidence": conf, "level": "HIGH" if conf >= 75 else "MEDIUM"},
                        "fusion_result": {"verdict": pred},
                        "explanation": {
                            "evidence": [f"Archived verification completed for {row.get('file_name')}."],
                            "recommendation": "Media analysis record archived."
                        }
                    }
                    ReportGenerator.generate_report(context_dict)
            except Exception as e:
                pass

    if not os.path.exists(local_path):
        return jsonify({
            'status': 'error',
            'error': "You can't download because no analysis is done"
        }), 404

    as_attachment = request.args.get('download', 'false').lower() == 'true'
    disposition = "attachment" if as_attachment else "inline"
    pdf_download_name = f"SatyaDrishti_Report_{analysis_id}.pdf"
    response = send_from_directory(
        reports_dir,
        filename,
        as_attachment=as_attachment,
        download_name=pdf_download_name,
        mimetype="application/pdf"
    )
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f'{disposition}; filename="{pdf_download_name}"'
    response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response

@analysis_bp.route('/analysis/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    if not os.path.exists(os.path.join(Config.UPLOAD_FOLDER, filename)):
        return jsonify({
            'status': 'error',
            'error': 'File not found'
        }), 404
    return send_from_directory(Config.UPLOAD_FOLDER, filename)
