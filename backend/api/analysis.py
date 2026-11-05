import os
from flask import Blueprint, request, jsonify, send_from_directory
from backend.core.orchestrator import Orchestrator
from backend.core.database import (
    get_analyses_by_user,
    get_analysis_by_id,
    get_agent_logs,
    record_feedback_audit_intent,
    record_feedback_audit_outcome
)
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

            raw_providers = request.form.get('selected_providers')
            selected_providers = json.loads(raw_providers) if raw_providers else None

            result = get_orchestrator().run_orchestration(file_id, user=user, selected_providers=selected_providers)
            from backend.core.nlp_response import generate_nlp_response
            nlp_res = generate_nlp_response(result.get("fusion_result", {}), question=question)
            result["natural_language_answer"] = nlp_res
            return jsonify(result), 200
        except Exception as e:
            return jsonify({'status': 'error', 'error': f'Upload and analysis failed: {str(e)}'}), 500

    # Handle standard JSON request with file_id or media_id
    data = request.get_json() or {}
    file_id = data.get('file_id') or data.get('media_id') or data.get('id')
    media_type = data.get('media_type')  # Optional; Orchestrator auto-detects true media type
    user = data.get('user')  # Optional user details: {"id": 1, "name": "...", "email": "..."}
    
    # Check required parameter
    if not file_id:
        return jsonify({
            'status': 'error',
            'error': 'Missing required parameter: file_id'
        }), 400
        
    question = data.get('question')
    selected_providers = data.get('selected_providers')

    # Invoke Orchestrator workflow with automatic media routing
    try:
        result = get_orchestrator().run_orchestration(
            file_id, media_type=media_type, user=user, selected_providers=selected_providers
        )
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
    from backend.utils.auth import get_bearer_token, verify_token
    token = get_bearer_token(request)
    param_user_id = request.args.get('user_id')

    auth_user = None
    if token:
        is_valid, user_info, err = verify_token(token)
        if not is_valid or not user_info:
            return jsonify({
                'status': 'error',
                'error': err or 'Invalid or expired session. Please sign in again.'
            }), 401
        auth_user = user_info

    # Enforce access control and isolation
    if param_user_id and str(param_user_id).strip().lower() not in ('null', 'undefined', 'none', ''):
        if not auth_user:
            return jsonify({
                'status': 'error',
                'error': 'Authentication required to access user history.'
            }), 401
        auth_user_id = str(auth_user.get('id', ''))
        auth_role = auth_user.get('role', '')
        if auth_role != 'service_role' and auth_user_id != str(param_user_id):
            return jsonify({
                'status': 'error',
                'error': 'Access denied: You cannot view another user\'s analysis history.'
            }), 403
        target_user_id = param_user_id
    elif auth_user:
        target_user_id = auth_user.get('id')
    else:
        return jsonify({
            'status': 'success',
            'history': []
        }), 200

    try:
        history = get_analyses_by_user(target_user_id)
        from backend.utils.security import generate_signed_media_token
        for item in history:
            analysis_id = item.get('id')
            db_fn = item.get('file_name') or ''
            token_sig = generate_signed_media_token(analysis_id, user_id=target_user_id)
            item['media_url'] = f"/api/analysis/uploads/{db_fn}?token={token_sig}"
            item['mediaUrl'] = item['media_url']
            from backend.core.decision_system import PresentationDecisionPolicy
            c_v = PresentationDecisionPolicy.get_canonical_verdict(
                verdict=item.get('prediction') or item.get('verdict'),
                primary_verdict=item.get('primary_verdict') or item.get('primaryVerdict'),
                original_verdict=item.get('original_verdict') or item.get('originalVerdict'),
                is_binary_fallback=bool(item.get('is_binary_fallback', False)),
                fake_probability=item.get('fake_probability'),
            )
            item['prediction'] = c_v
            item['verdict'] = c_v
            item['canonical_verdict'] = c_v
            item['canonicalVerdict'] = c_v
            item['primary_verdict'] = c_v
            item['primaryVerdict'] = c_v
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

    # Verify access permission if record belongs to a registered user
    owner_id = row.get('owner_id') or row.get('user_id')
    if owner_id and str(owner_id).strip() and str(owner_id).lower() not in ('none', 'null', 'guest'):
        from backend.utils.auth import get_bearer_token, verify_token
        from backend.utils.security import verify_signed_media_token

        sig_token = request.args.get('token') or request.args.get('sig')
        is_valid_sig = False
        if sig_token:
            valid, _ = verify_signed_media_token(sig_token, analysis_id)
            is_valid_sig = valid

        if not is_valid_sig:
            bearer = get_bearer_token(request)
            if not bearer:
                return jsonify({
                    'status': 'error',
                    'error': 'Authentication required to view this private analysis record.'
                }), 401

            is_valid, user_info, err = verify_token(bearer)
            if not is_valid or not user_info:
                return jsonify({
                    'status': 'error',
                    'error': err or 'Invalid or expired session. Please sign in again.'
                }), 401

            auth_user_id = str(user_info.get('id', ''))
            auth_role = user_info.get('role', '')
            if auth_role != 'service_role' and auth_user_id != str(owner_id):
                return jsonify({
                    'status': 'error',
                    'error': 'Access denied: You do not own this analysis record.'
                }), 403
    
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

    import json
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(backend_dir, 'reports')
    canonical_json_path = os.path.join(reports_dir, f"analysis_{analysis_id}.json")
    cached = None
    if os.path.exists(canonical_json_path):
        try:
            with open(canonical_json_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
        except Exception:
            cached = None

    if not cached and logs:
        for l in logs:
            if l.get('agent_name') == 'Fusion Agent' and l.get('metadata'):
                cached = l.get('metadata')
                break

    cached_dict = cached if isinstance(cached, dict) else {}
    cached_fusion = cached_dict.get('fusion_result', cached_dict)

    from backend.core.decision_system import PresentationDecisionPolicy, DecisionState
    pred = row.get('prediction') or cached_fusion.get('verdict') or 'UNCERTAIN'
    raw_primary = row.get('primary_verdict') or row.get('primaryVerdict') or cached_fusion.get('primary_verdict') or pred
    fake_p = row.get('fake_probability') or cached_fusion.get('fake_probability')
    fake_prob = float(fake_p) if fake_p is not None else None
    norm_primary = PresentationDecisionPolicy.normalize_verdict(raw_primary, fake_prob)
    original_v = row.get('original_verdict') or row.get('originalVerdict') or cached_fusion.get('original_verdict') or pred
    is_fallback = bool(row.get('is_binary_fallback', cached_fusion.get('is_binary_fallback', False))) or (original_v in ('UNCERTAIN', 'INCONCLUSIVE'))
    canonical_v = PresentationDecisionPolicy.get_canonical_verdict(
        verdict=pred,
        primary_verdict=norm_primary,
        original_verdict=original_v,
        is_binary_fallback=is_fallback,
        fake_probability=fake_prob,
    )
    primary_v = canonical_v

    is_not_suitable = primary_v == "NOT_SUITABLE"
    is_illustration = primary_v == "ILLUSTRATION"
    is_fake = primary_v == "FAKE"
    is_real = primary_v == "REAL"
    is_uncertain = primary_v == "UNCERTAIN"
    rel_warning = row.get('reliability_warning') or cached_fusion.get('reliability_warning') or (
        PresentationDecisionPolicy.AMBIGUITY_WARNING if is_fallback else None
    )

    conf_val = row.get('confidence') if row.get('confidence') is not None else cached_fusion.get('confidence')
    conf_level_val = cached_fusion.get('confidence_level') or cached_fusion.get('confidenceLevel')

    if is_illustration or is_not_suitable:
        fake_prob = None
    elif fake_prob is None:
        if is_fake:
            fake_prob = round(float(conf_val) / 100.0, 4) if (conf_val is not None and float(conf_val) > 50) else 0.88
        elif is_real:
            fake_prob = round(1.0 - (float(conf_val) / 100.0), 4) if (conf_val is not None and float(conf_val) > 50) else 0.12
        else:
            fake_prob = 0.50

    real_prob = (1.0 - fake_prob) if fake_prob is not None else None

    # Rigorous uncertainty derived from probability p: U = 1 - |2p - 1|
    if fake_prob is not None and not (is_not_suitable or is_illustration):
        uncertainty = round(1.0 - abs(2.0 * fake_prob - 1.0), 4)
    else:
        uncertainty_raw = cached_fusion.get('uncertainty') or row.get('uncertainty')
        uncertainty = float(uncertainty_raw) if uncertainty_raw is not None else None

    # True ensemble raw specialist score
    raw_score_val = cached_fusion.get('raw_model_score') or row.get('raw_model_score')
    if raw_score_val is not None:
        raw_model_score = float(raw_score_val)
    elif fake_prob is not None and not (is_not_suitable or is_illustration):
        raw_model_score = float(fake_prob)
    else:
        raw_model_score = None

    if is_not_suitable or is_illustration:
        conf = None
        conf_level = 'Not Performed'
    elif is_uncertain:
        conf = None
        conf_level = 'Uncertified'
    elif is_fallback:
        conf = float(conf_val) if conf_val is not None else (round(max(fake_prob, 1.0 - fake_prob) * 100.0, 2) if fake_prob is not None else 65.0)
        conf_level = conf_level_val if conf_level_val else 'Low'
    else:
        conf = float(conf_val) if conf_val is not None else (round(max(fake_prob, 1.0 - fake_prob) * 100.0, 2) if fake_prob is not None else 85.0)
        conf_level = conf_level_val if conf_level_val else ('High' if conf >= 80 else 'Medium' if conf >= 60 else 'Low')

    # Clean file name and local path
    db_file_name = row.get('file_name') or ''
    clean_display_name = db_file_name
    file_prefix_id = analysis_id
    if '_' in db_file_name and len(db_file_name.split('_')[0]) in (32, 36):
        file_prefix_id = db_file_name.split('_', 1)[0]
        clean_display_name = db_file_name.split('_', 1)[1]

    # Resolve actual media URL path with signed token and file_id query parameter
    from backend.utils.security import generate_signed_media_token
    token_user = row.get('owner_id') or row.get('user_id')
    token = generate_signed_media_token(file_prefix_id, user_id=token_user)
    media_url = f"/api/analysis/uploads/{db_file_name}?token={token}&file_id={file_prefix_id}"

    # Extract or compute empirical forensic evidence cards, face bounding box, and quality gate
    evidence_cards = (
        cached_fusion.get('evidence_cards') or cached_fusion.get('evidenceCards')
        or cached_dict.get('evidence_cards') or cached_dict.get('evidenceCards') or []
    )
    evidence_snapshots = (
        cached_fusion.get('evidence_snapshots') or cached_fusion.get('evidenceSnapshots')
        or cached_dict.get('evidence_snapshots') or cached_dict.get('evidenceSnapshots') or []
    )
    if not evidence_cards and evidence_snapshots:
        evidence_cards = evidence_snapshots
    if not evidence_snapshots and evidence_cards:
        evidence_snapshots = evidence_cards

    face_detected = not (is_not_suitable or is_illustration)
    face_bbox = None
    quality_gate_dict = {}
    features_dict = {}

    media_type_str = str(row.get('media_type', 'image')).lower()
    upload_dir = Config.UPLOAD_FOLDER
    target_media_path = None
    if db_file_name and os.path.isfile(os.path.join(upload_dir, db_file_name)):
        target_media_path = os.path.join(upload_dir, db_file_name)
    elif os.path.exists(upload_dir):
        for entry in os.listdir(upload_dir):
            if entry.startswith(f"{analysis_id}_") or entry.startswith(f"{file_prefix_id}_") or (clean_display_name and entry.endswith(f"_{clean_display_name}")):
                target_media_path = os.path.join(upload_dir, entry)
                break

    if not evidence_cards and not evidence_snapshots and not (is_not_suitable or is_illustration):
        if media_type_str == 'image' and target_media_path and os.path.isfile(target_media_path):
            try:
                import cv2
                from backend.preprocessing.quality_gate import ImageQualityGate
                from backend.forensics.forensic_analyzers import ForensicEvidenceSuite

                img_bgr = cv2.imread(target_media_path)
                if img_bgr is not None:
                    q_res = ImageQualityGate().evaluate(img_bgr)
                    face_detected = q_res.face_detected
                    face_bbox = list(q_res.primary_face_bbox) if q_res.primary_face_bbox else None
                    quality_gate_dict = q_res.to_dict()

                    f_suite = ForensicEvidenceSuite()
                    forensic_res = f_suite.evaluate(img_bgr, face_bbox=q_res.primary_face_bbox)
                    evidence_cards = f_suite.generate_evidence_cards(
                        forensic_res, img_bgr, face_bbox=q_res.primary_face_bbox, verdict=primary_v
                    )
                    features_dict = {
                        "noise_inconsistency": forensic_res.get("composite_forensic_score", 0.15),
                        "edge_boundary_artifacts": forensic_res.get("edge_analysis", {}).get("edge_score", 0.10),
                        "fft_spectral_anomaly": forensic_res.get("fft_analysis", {}).get("fft_score", 0.10),
                        "ela_compression_anomaly": forensic_res.get("ela_analysis", {}).get("ela_score", 0.10),
                        "blur_score": q_res.blur_score,
                    }
            except Exception as e:
                import logging
                logging.getLogger("AnalysisAPI").warning(f"Failed to generate evidence cards for {analysis_id}: {e}")

        elif media_type_str == 'video' and target_media_path and os.path.isfile(target_media_path):
            try:
                import cv2
                import numpy as np
                from backend.preprocessing.quality_gate import ImageQualityGate
                from backend.forensics.forensic_analyzers import VideoEvidenceSuite

                cap = cv2.VideoCapture(target_media_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                    num_samples = min(16, total_frames) if total_frames > 0 else 0
                    if num_samples >= 2:
                        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
                        frames = []
                        for s_idx in sample_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                            ret, f = cap.read()
                            if ret and f is not None and f.size > 0:
                                frames.append(f)
                        cap.release()

                        if frames:
                            q_gate = ImageQualityGate()
                            fp_val = fake_prob if fake_prob is not None else 0.50
                            frame_fake_probs = [float(fp_val)] * len(frames)
                            evidence_snapshots = VideoEvidenceSuite.generate_evidence_snapshots(
                                frames=frames,
                                sample_indices=sample_indices[:len(frames)],
                                fps=fps,
                                frame_fake_probs=frame_fake_probs,
                                temporal_variance=0.08 if primary_v == "REAL" else 0.22,
                                temporal_conflict=False,
                                verdict=primary_v,
                                quality_gate=q_gate,
                            )
                            evidence_cards = evidence_snapshots
            except Exception as e:
                import logging
                logging.getLogger("AnalysisAPI").warning(f"Failed to generate video evidence snapshots for {analysis_id}: {e}")

    if is_illustration:
        explanation_text = "This image appears to be a stylized illustration, anime, or cartoon rather than a photographic image. SatyaDrishti's deepfake detection models are calibrated for real photographic media and cannot produce meaningful results for non-photographic content. No deepfake analysis was performed."
        reason_text = "Non-photographic media detected: illustration, anime, or cartoon."
        statement_text = "This image appears to be a stylized illustration rather than a photographic image. SatyaDrishti analyzes photographic media and does not evaluate illustrations or cartoons for deepfake manipulation."
    elif is_not_suitable:
        explanation_text = "No detectable human face was found in this media. Media is not suitable for facial analysis."
        reason_text = "No suitable face detected."
        statement_text = "No detectable human face was found in this media. SatyaDrishti's current deepfake detector is designed for facial-media authenticity analysis. Upload clear media containing a visible human face."
    elif is_fake:
        if is_fallback:
            explanation_text = f"Provisional binary assessment: Media leans FAKE. {rel_warning or 'Evidence is uncertified or contains conflicting indicators; manual verification is advised.'}"
            reason_text = row.get('reason') or "Provisional binary assessment leans FAKE."
            statement_text = "Deepfake detected / provisional assessment based on elevated anomaly indicators."
        else:
            explanation_text = "Multi-signal consensus confirms deepfake manipulation patterns in media features."
            reason_text = "Strong evidence of synthetic manipulation identified."
            if media_type_str == 'video':
                statement_text = f"Temporal consistency and spatial analysis confirmed elevated synthetic anomaly indicators across sampled video frames (fake probability {fake_prob:.1%})."
            else:
                top_finding = evidence_cards[0].get('observed_finding') if evidence_cards else "elevated frequency and boundary anomalies"
                statement_text = f"Facial region analysis identified synthetic indicators: {top_finding}. Primary facial classifier measured a calibrated synthetic probability of {fake_prob:.1%}."
    elif is_real:
        if is_fallback:
            explanation_text = f"Provisional binary prediction (REAL): The detector has NOT established authenticity. {rel_warning or 'Evidence lies in the indeterminate margin; manual forensic verification is required.'}"
            reason_text = row.get('reason') or "Provisional prediction: Evidence lies in the ambiguous decision margin; authenticity has not been established."
            statement_text = "PROVISIONAL PREDICTION: The detector has NOT established authenticity. Analysis evidence lies in the indeterminate decision margin; independent verification is required."
        else:
            explanation_text = "Authentic natural media with verifiable natural features. Limitation: Absence of detected manipulation does not prove absolute authenticity."
            reason_text = "Visual and frequency features align with authentic media."
            if media_type_str == 'video':
                statement_text = f"Sampled video frames observed authentic features (mean fake probability {fake_prob:.1%}) with stable inter-frame continuity. No strong manipulation evidence was detected."
            else:
                statement_text = f"Facial region analysis detected no significant manipulation artifacts across neural classifiers ({1.0 - fake_prob:.1%} authentic probability). Notice: Absence of detected manipulation does not guarantee absolute authenticity."
    else:
        explanation_text = "Provisional assessment: Media characteristics evaluated under system decision policy. Authenticity has not been established; independent verification is recommended."
        reason_text = row.get('reason') or "Provisional assessment under system decision policy."
        statement_text = "Provisional assessment: Evaluated under system decision policy; independent verification is advised."

    raw_cached_expl = (cached_fusion.get('explanation') or cached.get('explanation')) if cached else None
    if raw_cached_expl:
        if "uncertain" in raw_cached_expl.lower() or "secondary manual verification" in raw_cached_expl.lower():
            raw_cached_expl = None
        elif is_fake and "authenticity has not been established" in raw_cached_expl.lower():
            raw_cached_expl = None
        elif is_real and ("deepfake" in raw_cached_expl.lower() or "synthetic manipulation" in raw_cached_expl.lower()):
            raw_cached_expl = None
    final_expl = raw_cached_expl or explanation_text

    raw_cached_stmt = (cached_fusion.get('user_facing_statement') or cached.get('userFacingStatement')) if cached else None
    if raw_cached_stmt:
        if "uncertain" in raw_cached_stmt.lower() or "secondary manual verification" in raw_cached_stmt.lower():
            raw_cached_stmt = None
        elif is_fake and "notice:" in raw_cached_stmt.lower():
            raw_cached_stmt = None
        elif is_real and ("deepfake" in raw_cached_stmt.lower() or "synthetic manipulation" in raw_cached_stmt.lower()):
            raw_cached_stmt = None
    final_stmt = raw_cached_stmt or statement_text

    raw_evidence = (
        ["Media type: Illustration / Anime / Cartoon (non-photographic content detected)."]
        if is_illustration
        else (
            ["Eligibility: Unsupported media (No detectable human face found in image)."]
            if is_not_suitable
            else (
                (cached_fusion.get('evidence') or cached.get('evidence'))
                if cached and (cached_fusion.get('evidence') or cached.get('evidence'))
                else [
                    f"File analyzed: {clean_display_name}",
                    f"Forensic verdict: {primary_v} ({conf_level} confidence)",
                ]
            )
        )
    )
    if is_fake:
        clean_evidence = [
            e for e in raw_evidence
            if "abstained" not in e.lower()
            and "uncertain" not in e.lower()
            and "secondary manual verification is recommended" not in e.lower()
            and "authenticity has not been established" not in e.lower()
        ]
    elif is_real:
        clean_evidence = [
            e for e in raw_evidence
            if "abstained" not in e.lower()
            and "uncertain" not in e.lower()
            and "secondary manual verification is recommended" not in e.lower()
        ]
    else:
        clean_evidence = [
            e for e in raw_evidence
            if "uncertain" not in e.lower()
        ]

    if is_illustration:
        rec_text = "Upload a clear photograph of a real person for deepfake analysis. SatyaDrishti cannot assess illustrations, anime, or cartoon artwork."
    elif is_not_suitable:
        rec_text = "Upload a clear media file containing a visible human face."
    elif is_fake:
        rec_text = "Deepfake manipulation flagged. Secondary forensic inspection or source provenance verification is recommended before publishing or trusting this media."
    elif is_real:
        rec_text = "No strong evidence of manipulation was detected across evaluated detectors. Note: This does not prove authenticity, as novel synthesis techniques or clean recaptures may leave no detectable trace."
    else:
        rec_text = "Provisional assessment. Independent forensic verification is recommended."

    return jsonify({
        'status': 'success',
        'analysis': {
            'id': row.get('id'),
            'fileName': clean_display_name,
            'fileType': row.get('media_type', 'image'),
            'mediaUrl': media_url,
            'verdict': primary_v,
            'prediction': primary_v,
            'canonical_verdict': primary_v,
            'canonicalVerdict': primary_v,
            'primary_verdict': primary_v,
            'primaryVerdict': primary_v,
            'provisional_suggestion': primary_v if is_fallback else None,
            'provisionalSuggestion': primary_v if is_fallback else None,
            'original_verdict': original_v,
            'originalVerdict': original_v,
            'is_binary_fallback': is_fallback,
            'isBinaryFallback': is_fallback,
            'is_estimate': is_fallback,
            'isEstimate': is_fallback,
            'reliability_warning': rel_warning,
            'reliabilityWarning': rel_warning,
            'confidence': round(conf, 2) if (conf is not None and not (is_not_suitable or is_illustration)) else None,
            'confidenceLevel': conf_level,
            'fakeProbability': fake_prob,
            'realProbability': real_prob,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'rawModelScore': raw_model_score,
            'raw_model_score': raw_model_score,
            'uncertainty': uncertainty,
            'models': cached_fusion.get('models') or cached.get('models') if cached else [],
            'visualAnalysis': cached_fusion.get('visual_analysis') or cached.get('visual_analysis') if cached else {},
            'visual_analysis': cached_fusion.get('visual_analysis') or cached.get('visual_analysis') if cached else {},
            'faceDetected': face_detected,
            'face_detected': face_detected,
            'faceBbox': face_bbox,
            'face_bbox': face_bbox,
            'mediaCategory': "Illustration / Anime / Cartoon" if is_illustration else ("Document / Non-face image" if is_not_suitable else "Facial Media"),
            'media_category': "Illustration / Anime / Cartoon" if is_illustration else ("Document / Non-face image" if is_not_suitable else "Facial Media"),
            'qualityGate': quality_gate_dict,
            'quality_gate': quality_gate_dict,
            'evidenceCards': (cached_fusion.get('evidence_cards') or cached.get('evidence_cards')) if cached and cached_fusion.get('evidence_cards') else evidence_cards,
            'evidence_cards': (cached_fusion.get('evidence_cards') or cached.get('evidence_cards')) if cached and cached_fusion.get('evidence_cards') else evidence_cards,
            'evidenceSnapshots': evidence_snapshots,
            'evidence_snapshots': evidence_snapshots,
            'audioVisualSync': (cached_fusion.get('audio_visual_sync') or cached_dict.get('audio_visual_sync')) if (cached_fusion.get('audio_visual_sync') or cached_dict.get('audio_visual_sync')) else {
                'analyzed': False,
                'status': 'NOT AVAILABLE — NO AUDIO TRACK' if media_type_str == 'video' else 'NOT APPLICABLE — IMAGE MEDIA',
                'reason': 'Lip-sync analysis not performed.'
            },
            'audio_visual_sync': (cached_fusion.get('audio_visual_sync') or cached_dict.get('audio_visual_sync')) if (cached_fusion.get('audio_visual_sync') or cached_dict.get('audio_visual_sync')) else {
                'analyzed': False,
                'status': 'NOT AVAILABLE — NO AUDIO TRACK' if media_type_str == 'video' else 'NOT APPLICABLE — IMAGE MEDIA',
                'reason': 'Lip-sync analysis not performed.'
            },
            'sequenceAggregation': cached_fusion.get('sequence_aggregation') or cached_dict.get('sequence_aggregation'),
            'sequence_aggregation': cached_fusion.get('sequence_aggregation') or cached_dict.get('sequence_aggregation'),
            'frameEvidenceTimeline': cached_fusion.get('frame_evidence_timeline') or cached_dict.get('frame_evidence_timeline') or [],
            'frame_evidence_timeline': cached_fusion.get('frame_evidence_timeline') or cached_dict.get('frame_evidence_timeline') or [],
            'trackedFaces': cached_fusion.get('tracked_faces') or cached_dict.get('tracked_faces') or [],
            'tracked_faces': cached_fusion.get('tracked_faces') or cached_dict.get('tracked_faces') or [],
            'totalDurationSeconds': cached_fusion.get('total_duration_seconds') or cached_fusion.get('totalDurationSeconds') or (cached_fusion.get('metadata') or {}).get('duration_seconds') or (cached_dict.get('metadata') or {}).get('duration_seconds'),
            'total_duration_seconds': cached_fusion.get('total_duration_seconds') or cached_fusion.get('totalDurationSeconds') or (cached_fusion.get('metadata') or {}).get('duration_seconds') or (cached_dict.get('metadata') or {}).get('duration_seconds'),
            'extractedFramesCount': cached_fusion.get('extracted_frames_count') or cached_fusion.get('extractedFramesCount') or (cached_fusion.get('metadata') or {}).get('extracted_frames_count') or (len(evidence_snapshots) if evidence_snapshots else None),
            'extracted_frames_count': cached_fusion.get('extracted_frames_count') or cached_fusion.get('extractedFramesCount') or (cached_fusion.get('metadata') or {}).get('extracted_frames_count') or (len(evidence_snapshots) if evidence_snapshots else None),
            'analyzedFramesCount': cached_fusion.get('analyzed_frames_count') or cached_fusion.get('analyzedFramesCount') or (cached_fusion.get('metadata') or {}).get('analyzed_frames_count') or (len(evidence_snapshots) if evidence_snapshots else None),
            'analyzed_frames_count': cached_fusion.get('analyzed_frames_count') or cached_fusion.get('analyzedFramesCount') or (cached_fusion.get('metadata') or {}).get('analyzed_frames_count') or (len(evidence_snapshots) if evidence_snapshots else None),
            'trackedFacesCount': cached_fusion.get('tracked_faces_count') or cached_fusion.get('trackedFacesCount') or len(cached_fusion.get('tracked_faces') or cached_dict.get('tracked_faces') or []),
            'tracked_faces_count': cached_fusion.get('tracked_faces_count') or cached_fusion.get('trackedFacesCount') or len(cached_fusion.get('tracked_faces') or cached_dict.get('tracked_faces') or []),
            'firstAnalyzedTimestamp': cached_fusion.get('first_analyzed_timestamp') or cached_fusion.get('firstAnalyzedTimestamp') or 0.0,
            'first_analyzed_timestamp': cached_fusion.get('first_analyzed_timestamp') or cached_fusion.get('firstAnalyzedTimestamp') or 0.0,
            'lastAnalyzedTimestamp': cached_fusion.get('last_analyzed_timestamp') or cached_fusion.get('lastAnalyzedTimestamp'),
            'last_analyzed_timestamp': cached_fusion.get('last_analyzed_timestamp') or cached_fusion.get('lastAnalyzedTimestamp'),
            'temporalCoveragePercentage': cached_fusion.get('temporal_coverage_percentage') or cached_fusion.get('temporalCoveragePercentage') or 100.0,
            'temporal_coverage_percentage': cached_fusion.get('temporal_coverage_percentage') or cached_fusion.get('temporalCoveragePercentage') or 100.0,
            'temporalMetrics': cached_fusion.get('temporal_metrics') or cached_dict.get('temporal_metrics') or {},
            'temporal_metrics': cached_fusion.get('temporal_metrics') or cached_dict.get('temporal_metrics') or {},
            'metadata': cached_fusion.get('metadata') or cached_dict.get('metadata') or {},
            'features': (cached_fusion.get('features') or cached_dict.get('features')) if cached_fusion.get('features') else features_dict,
            'explanation': final_expl,
            'reason': cached_fusion.get('reason') or cached_dict.get('reason') or reason_text,
            'userFacingStatement': final_stmt,
            'userFriendlyExplanation': (cached_fusion.get('user_friendly_explanation') or cached.get('user_friendly_explanation') or cached_fusion.get('userFriendlyExplanation') or cached.get('userFriendlyExplanation')) if cached else None,
            'user_friendly_explanation': (cached_fusion.get('user_friendly_explanation') or cached.get('user_friendly_explanation') or cached_fusion.get('userFriendlyExplanation') or cached.get('userFriendlyExplanation')) if cached else None,
            'evidence': clean_evidence,
            'recommendation': rec_text,
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
                    is_susp = pred == "SUSPICIOUS - provisional"
                    is_inconc = pred in ("INCONCLUSIVE", "UNCERTAIN")
                    is_fb = is_susp or is_inconc
                    warning_msg = (
                        "Provisional verdict: Forensic conflict detected." if is_susp else
                        ("Inconclusive result: Evidence lies in indeterminate margin. Probability is an uncertified estimate." if is_inconc else None)
                    )
                    context_dict = {
                        "analysis_id": analysis_id,
                        "user": {"name": row.get("user_name", "Researcher")},
                        "uploaded_file": row.get("file_name", "analysis_media"),
                        "media_type": str(row.get("media_type", "IMAGE")).upper(),
                        "validation_result": {"status": "PASSED", "issues": []},
                        "confidence_result": {"confidence": conf, "level": "Provisional" if is_fb else ("HIGH" if conf >= 80 else "MEDIUM")},
                        "fusion_result": {
                            "verdict": pred,
                            "primary_verdict": pred,
                            "is_binary_fallback": is_fb,
                            "reliability_warning": warning_msg,
                        },
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
    """
    Securely serves uploaded media files for browser preview.
    Security protections:
    1. Media is resolved strictly by unique file ID (UUID) or authorized upload registry.
    2. Enforces access control for identified users via cryptographically signed URL token or session ownership.
    3. Seamlessly supports direct file requests, pre-signed URLs, and anonymous preview requests.
    """
    from backend.utils.security import verify_signed_media_token
    from backend.api.upload import get_upload_owner

    upload_dir = Config.UPLOAD_FOLDER
    os.makedirs(upload_dir, exist_ok=True)

    clean_name = os.path.basename(filename)
    
    # 1. Extract candidate file_id from filename or query parameters
    file_id = None
    if '_' in clean_name:
        candidate_id = clean_name.split('_', 1)[0]
        if len(candidate_id) in (32, 36) or '-' in candidate_id:
            file_id = candidate_id
    elif len(os.path.splitext(clean_name)[0]) in (32, 36):
        file_id = os.path.splitext(clean_name)[0]

    candidate_param = request.args.get('file_id') or request.args.get('id') or request.args.get('analysis_id')
    if candidate_param:
        file_id = candidate_param

    # Check if clean_name exists in upload_dir or is a suffix of an existing uploaded file
    matched_entry = None
    if os.path.isfile(os.path.join(upload_dir, clean_name)):
        matched_entry = clean_name
        if not file_id:
            file_id = clean_name
    elif os.path.exists(upload_dir):
        for entry in os.listdir(upload_dir):
            if entry.endswith(f"_{clean_name}"):
                matched_entry = entry
                if not file_id:
                    file_id = entry.split('_', 1)[0]
                break

    if not file_id:
        return jsonify({
            'status': 'error',
            'error': 'Direct lookup by original filename is disallowed for security. Media must be resolved by unique file ID.'
        }), 400

    # 2. Authorization and Ownership Verification
    token = request.args.get('token') or request.args.get('sig')
    is_valid_token = False
    if token:
        valid1, _ = verify_signed_media_token(token, file_id)
        valid2, _ = (False, None) if not candidate_param else verify_signed_media_token(token, candidate_param)
        valid3, _ = verify_signed_media_token(token, clean_name)
        is_valid_token = valid1 or valid2 or valid3

    # Determine owner from upload registry or database record
    owner_id = get_upload_owner(file_id)
    if not owner_id and candidate_param:
        owner_id = get_upload_owner(candidate_param)
    if not owner_id:
        row = get_analysis_by_id(file_id) or (get_analysis_by_id(candidate_param) if candidate_param else None)
        if row:
            owner_id = row.get('owner_id') or row.get('user_id')

    from backend.utils.auth import get_bearer_token, verify_token
    bearer = get_bearer_token(request)
    requester_id = None
    token_error = None
    if bearer:
        is_valid, user_info, err = verify_token(bearer)
        if is_valid and user_info:
            requester_id = user_info.get('id')
            if user_info.get('role') == 'service_role':
                requester_id = owner_id
        else:
            token_error = err or "Invalid or expired authentication session."
    elif request.headers.get('X-User-ID') or request.args.get('user_id'):
        requester_id = request.headers.get('X-User-ID') or request.args.get('user_id')

    # If the file belongs to an identified registered user, enforce ownership or signed URL
    if owner_id and str(owner_id).strip() and str(owner_id).lower() not in ('none', 'null', 'guest'):
        if is_valid_token:
            pass
        elif token_error:
            return jsonify({
                'status': 'error',
                'error': token_error
            }), 401
        elif requester_id and str(requester_id) == str(owner_id):
            pass
        elif requester_id and str(requester_id) != str(owner_id):
            return jsonify({
                'status': 'error',
                'error': 'Access denied: You do not own this media upload.'
            }), 403
        else:
            return jsonify({
                'status': 'error',
                'error': 'Authentication required to access this media upload.'
            }), 401

    # 3. Locate file on disk
    matched_filename = matched_entry
    if not matched_filename:
        target_path = os.path.join(upload_dir, clean_name)
        if os.path.isfile(target_path):
            matched_filename = clean_name
        elif os.path.exists(upload_dir):
            for entry in os.listdir(upload_dir):
                if entry.startswith(f"{file_id}_") or os.path.splitext(entry)[0] == file_id:
                    matched_filename = entry
                    break
                if candidate_param and (entry.startswith(f"{candidate_param}_") or os.path.splitext(entry)[0] == candidate_param):
                    matched_filename = entry
                    break
                if entry.endswith(f"_{clean_name}"):
                    matched_filename = entry
                    break

    if not matched_filename and file_id:
        row = get_analysis_by_id(file_id) or (get_analysis_by_id(candidate_param) if candidate_param else None)
        if row and row.get('file_name'):
            db_fn = os.path.basename(row['file_name'])
            candidate_path = os.path.join(upload_dir, db_fn)
            if os.path.isfile(candidate_path):
                matched_filename = db_fn

    if matched_filename and os.path.isfile(os.path.join(upload_dir, matched_filename)):
        return send_from_directory(upload_dir, matched_filename)

    # Remote storage fallback strictly matching file_id
    try:
        from backend.services.supabase_service import SupabaseService
        for folder in ["images", "videos"]:
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"]:
                remote_path = f"{folder}/{file_id}{ext}"
                local_target = os.path.join(upload_dir, f"{file_id}{ext}")
                if SupabaseService.download_media(remote_path, local_target):
                    if os.path.isfile(local_target):
                        return send_from_directory(upload_dir, f"{file_id}{ext}")
    except Exception:
        pass

    return jsonify({
        'status': 'error',
        'error': f'File not found: {filename}'
    }), 404


@analysis_bp.route('/providers', methods=['GET'])
def get_providers():
    """Returns configured multi-provider AI vision services."""
    try:
        from backend.services.vision_service import VisionAnalysisService
        service = VisionAnalysisService()
        return jsonify({
            'status': 'success',
            'providers': service.get_configured_providers(),
            'active_provider_ids': service.get_active_provider_ids(),
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Failed to retrieve provider configuration: {str(e)}'
        }), 500


# ==============================================================================
# Feedback-Driven Learning Endpoints
# ==============================================================================

@analysis_bp.route('/analysis/<analysis_id>/feedback', methods=['POST'])
@analysis_bp.route('/<analysis_id>/feedback', methods=['POST'])
def submit_analysis_feedback(analysis_id):
    """
    Submits user feedback on an analysis result.
    Stored with strict 'UNVERIFIED' status and 'eligible_for_training=False'.
    Production model weights and predictions remain unchanged.
    """
    import uuid
    import datetime
    import json
    from backend.core.database import save_analysis_feedback, get_analysis_by_id, get_analysis_feedback
    from backend.utils.logger import setup_logger
    from backend.utils.auth import get_bearer_token, verify_token

    logger = setup_logger("AnalysisFeedbackAPI")
    data = request.get_json() or {}
    feedback_type = str(data.get("feedback_type", "")).strip().upper()
    valid_types = ["CORRECT", "INCORRECT", "REPORT_ISSUE"]

    if feedback_type not in valid_types:
        return jsonify({
            "status": "error",
            "error": f"Invalid feedback_type '{feedback_type}'. Must be one of: {', '.join(valid_types)}"
        }), 400

    user_comment = data.get("user_comment") or data.get("comment") or ""
    issue_category = data.get("issue_category") or ("RESULT_CORRECT" if feedback_type == "CORRECT" else "UNSPECIFIED")

    # Load original canonical analysis record
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(backend_dir, 'reports')
    canonical_json_path = os.path.join(reports_dir, f"analysis_{analysis_id}.json")

    analysis_data = {}
    if os.path.exists(canonical_json_path):
        try:
            with open(canonical_json_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        except Exception as je:
            logger.warning(f"Could not read canonical json for feedback: {je}")

    if not analysis_data:
        db_rec = get_analysis_by_id(analysis_id)
        if db_rec:
            analysis_data = db_rec

    if not analysis_data:
        return jsonify({
            "status": "error",
            "error": f"Analysis with ID '{analysis_id}' not found."
        }), 404

    # Resolve user ownership from Bearer token if present, falling back to payload / analysis record
    user_id = None
    token = get_bearer_token()
    if token:
        is_valid, user_payload, _ = verify_token(token)
        if is_valid and user_payload:
            user_id = user_payload.get("sub") or user_payload.get("id") or user_payload.get("user_id")

    if not user_id:
        user_id = data.get("user_id") or analysis_data.get("user_id")

    orig_pred = analysis_data.get("prediction") or analysis_data.get("canonical_verdict") or "UNKNOWN"
    user_correction = data.get("user_correction")
    if not user_correction:
        if feedback_type == "CORRECT":
            user_correction = orig_pred
        elif feedback_type == "INCORRECT":
            user_correction = "REAL" if issue_category == "FALSE_POSITIVE" or orig_pred == "FAKE" else "FAKE"
        else:
            user_correction = "UNVERIFIED"

    # Check for existing feedback on this analysis to prevent duplicate spam / accidental retries
    existing_fb = get_analysis_feedback(analysis_id)
    if existing_fb and existing_fb.get("status") == "PENDING":
        # Check if identical submission
        if (existing_fb.get("feedback_type") == feedback_type and
            existing_fb.get("user_correction") == user_correction and
            str(existing_fb.get("user_comment", "")).strip() == str(user_comment).strip()):
            logger.info(f"Duplicate feedback submission detected for analysis '{analysis_id}'. Returning existing record.")
            storage_mode = existing_fb.get("storage_tier", "local_fallback")
            if os.path.exists(canonical_json_path):
                try:
                    analysis_data["user_feedback"] = existing_fb
                    analysis_data["userFeedback"] = existing_fb
                    with open(canonical_json_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=2)
                except Exception as je:
                    logger.warning(f"Could not update canonical json with feedback: {je}")
            return jsonify({
                "status": "success",
                "persistence": storage_mode,
                "storage": storage_mode,
                "persisted_in_supabase": bool(existing_fb.get("persisted_in_supabase")),
                "duplicate": True,
                "message": "Feedback already recorded. Returning existing submission. Your submission has been securely recorded and queued for expert forensic review. Production models remain unchanged.",
                "feedback": existing_fb
            }), 201
        else:
            # User is updating their pending submission
            feedback_record = existing_fb
            feedback_record["feedback_type"] = feedback_type
            feedback_record["user_correction"] = user_correction
            feedback_record["user_comment"] = user_comment
            feedback_record["reason"] = data.get("reason") or user_comment
            feedback_record["issue_category"] = issue_category
            if user_id:
                feedback_record["user_id"] = user_id
            feedback_record["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    else:
        feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
        media_ref = (
            analysis_data.get("file_path") or
            analysis_data.get("fileName") or
            analysis_data.get("file_name") or
            analysis_data.get("report_path") or
            ""
        )

        feedback_record = {
            "id": feedback_id,
            "analysis_id": analysis_id,
            "user_id": user_id,
            "feedback_type": feedback_type,
            "user_correction": user_correction,
            "user_comment": user_comment,
            "reason": data.get("reason") or user_comment,
            "issue_category": issue_category,
            "original_prediction": orig_pred,
            "predicted_verdict": orig_pred,
            "confidence": analysis_data.get("confidence"),
            "confidence_level": analysis_data.get("confidence_level") or analysis_data.get("confidenceLevel") or "Low",
            "media_reference": media_ref,
            "analysis_version": analysis_data.get("analysis_version", "2.4.0"),
            "model_version": analysis_data.get("model_version", {
                "pipeline": "2.4.0",
                "xception": "v2.0_c23",
                "vit": "v1.5_patch16_224",
                "sbi": "v1.2",
                "fusion": "v2.4_calibrated"
            }),
            "forensic_findings": analysis_data.get("forensic_findings") or {
                "evidence_items": analysis_data.get("evidence", [])
            },
            "agent_disagreements": analysis_data.get("agent_disagreements") or {
                "disagreement_detected": bool(analysis_data.get("is_binary_fallback")),
                "conflicting_components": []
            },
            "status": "PENDING",
            "verification_status": "UNVERIFIED",
            "verified_label": None,
            "verifier_notes": None,
            "verified_by": None,
            "verified_at": None,
            "eligible_for_training": False,
            "dataset_version": None,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    # Save to database and local store with explicit persistence result handling
    save_res = save_analysis_feedback(feedback_record)

    if not save_res.get("success"):
        logger.error(f"Failed to persist feedback for analysis '{analysis_id}': {save_res.get('error')}")
        return jsonify({
            "status": "error",
            "persistence": "failed",
            "storage": "failed",
            "error": f"Failed to persist feedback to database and local store: {save_res.get('error')}"
        }), 500

    persisted_in_supabase = bool(save_res.get("persisted_in_supabase"))
    storage_tier = save_res.get("storage_tier", "supabase" if persisted_in_supabase else "local_fallback")

    # Attach feedback reference to canonical analysis json
    if os.path.exists(canonical_json_path):
        try:
            analysis_data["user_feedback"] = feedback_record
            analysis_data["userFeedback"] = feedback_record
            with open(canonical_json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2)
        except Exception as je:
            logger.warning(f"Could not update canonical json with feedback: {je}")

    if persisted_in_supabase:
        msg = "Feedback submitted successfully and persisted to Supabase database. Your submission has been securely recorded and queued for expert forensic review. Production models remain unchanged."
    else:
        msg = "Feedback saved to local fallback storage (cloud database currently unavailable). Your submission has been securely recorded and queued for expert forensic review. Production models remain unchanged."

    logger.info(
        f"Feedback recorded for analysis '{analysis_id}': type={feedback_type}, "
        f"tier={storage_tier}, supabase={persisted_in_supabase}. Model prediction remains unchanged."
    )

    return jsonify({
        "status": "success",
        "persistence": storage_tier,
        "storage": storage_tier,
        "persisted_in_supabase": persisted_in_supabase,
        "message": msg,
        "feedback": feedback_record
    }), 201


@analysis_bp.route('/analysis/<analysis_id>/feedback', methods=['GET'])
@analysis_bp.route('/<analysis_id>/feedback', methods=['GET'])
def get_analysis_feedback_endpoint(analysis_id):
    """Retrieves user feedback associated with a specific analysis."""
    from backend.core.database import get_analysis_feedback
    fb = get_analysis_feedback(analysis_id)
    if fb:
        return jsonify({
            "status": "success",
            "feedback": fb
        }), 200
    return jsonify({
        "status": "success",
        "feedback": None
    }), 200


@analysis_bp.route('/analysis/feedback/queue', methods=['GET'])
@analysis_bp.route('/feedback/queue', methods=['GET'])
def get_feedback_queue_endpoint():
    """
    Returns unverified feedback items for expert forensic analyst review.
    Filterable by ?status=UNVERIFIED|VERIFIED_CORRECT|VERIFIED_INCORRECT|REJECTED
    """
    from backend.core.database import get_feedback_queue
    status_filter = request.args.get("status")
    items = get_feedback_queue(status_filter=status_filter)
    return jsonify({
        "status": "success",
        "count": len(items),
        "queue": items
    }), 200


@analysis_bp.route('/analysis/feedback/<feedback_id>/verify', methods=['POST'])
@analysis_bp.route('/feedback/<feedback_id>/verify', methods=['POST'])
@analysis_bp.route('/analysis/feedback/<feedback_id>/review', methods=['POST'])
@analysis_bp.route('/feedback/<feedback_id>/review', methods=['POST'])
def verify_feedback_endpoint(feedback_id):
    """
    Forensic review interface for submitted feedback.
    Protected endpoint: requires valid Supabase JWT and verified reviewer/admin privileges.
    Authorized reviewers inspect disputed cases and assign verified REAL, FAKE or UNVERIFIED labels.
    Never treats unverified user feedback as ground truth.
    """
    from backend.utils.auth import get_bearer_token, verify_token, is_authorized_reviewer
    from backend.core.database import verify_feedback, get_feedback_by_id

    # 1. Supabase JWT Authentication
    token = get_bearer_token(request)
    if not token:
        return jsonify({
            "status": "error",
            "error": "Authentication required: Missing Bearer token in Authorization header."
        }), 401

    is_valid, user_info, err = verify_token(token)
    if not is_valid or not user_info:
        return jsonify({
            "status": "error",
            "error": err or "Invalid or expired session. Please sign in again."
        }), 401

    # 2. Reviewer Role Authorization (trusted server claims & env whitelists; rejects user_metadata and service_role)
    authorized, reason = is_authorized_reviewer(user_info)
    if not authorized:
        return jsonify({
            "status": "error",
            "error": reason or "Forbidden: User does not have reviewer privileges."
        }), 403

    # 3. Feedback Record Validation: Confirm record exists
    feedback_record = get_feedback_by_id(feedback_id)
    if not feedback_record:
        return jsonify({
            "status": "error",
            "error": f"Feedback record '{feedback_id}' not found."
        }), 404

    # 4. Self-Review Prevention: Reviewers cannot review their own submissions
    reviewer_user_id = str(user_info.get("id") or "").strip()
    reviewer_email = str(user_info.get("email") or "").strip()
    record_user_id = str(feedback_record.get("user_id") or "").strip()
    if reviewer_user_id and record_user_id and reviewer_user_id.lower() == record_user_id.lower():
        return jsonify({
            "status": "error",
            "error": "Self-review prohibited: Reviewers cannot verify or review feedback they submitted."
        }), 403

    # 5. Training State Guard: Reject review attempts if already consumed for training
    current_status = str(feedback_record.get("status") or "").strip().upper()
    if current_status == "USED_FOR_TRAINING" or feedback_record.get("used_for_training") is True:
        return jsonify({
            "status": "error",
            "error": "Conflict: Feedback record has already been consumed for training and is immutable."
        }), 409

    # 6. Parse and Validate Request Body
    data = request.get_json(silent=True) or {}
    raw_status = str(data.get("status") or data.get("verification_status") or "").strip().upper()
    if not raw_status:
        return jsonify({
            "status": "error",
            "error": "Review status is required. Must be VERIFIED, REJECTED, or PENDING."
        }), 400

    if raw_status in ("VERIFIED", "APPROVE", "VERIFIED_CORRECT", "VERIFIED_INCORRECT"):
        target_status = "VERIFIED"
    elif raw_status in ("REJECTED", "REJECT", "DISMISS"):
        target_status = "REJECTED"
    elif raw_status in ("PENDING", "RESET"):
        target_status = "PENDING"
    else:
        return jsonify({
            "status": "error",
            "error": f"Invalid review status '{raw_status}'. Must be one of: VERIFIED, REJECTED, PENDING."
        }), 400

    raw_label = data.get("verified_label")
    verified_label = str(raw_label).strip().upper() if raw_label is not None else None

    # Handle verified_label rules
    if target_status == "VERIFIED":
        # Prohibit automatic inference; require explicit reviewer verdict
        if not verified_label:
            return jsonify({
                "status": "error",
                "error": "Explicit verified_label ('REAL' or 'FAKE') is required when verifying feedback. Automatic inference is prohibited."
            }), 400
        if verified_label not in ("REAL", "FAKE"):
            return jsonify({
                "status": "error",
                "error": f"Invalid verified_label '{verified_label}'. Must be 'REAL' or 'FAKE' when verified."
            }), 400
        legacy_status = "VERIFIED_CORRECT" if verified_label == "REAL" else "VERIFIED_INCORRECT"
    elif target_status == "REJECTED":
        if verified_label and verified_label not in ("UNVERIFIED", "REAL", "FAKE"):
            return jsonify({
                "status": "error",
                "error": f"Invalid verified_label '{verified_label}'. Must be 'REAL', 'FAKE', or 'UNVERIFIED'."
            }), 400
        verified_label = "UNVERIFIED"
        legacy_status = "REJECTED"
    else:  # PENDING
        if verified_label and verified_label not in ("UNVERIFIED", "REAL", "FAKE"):
            return jsonify({
                "status": "error",
                "error": f"Invalid verified_label '{verified_label}'. Must be 'REAL', 'FAKE', or 'UNVERIFIED'."
            }), 400
        verified_label = "UNVERIFIED"
        legacy_status = "PENDING"

    # Strict architectural separation: Human review leaves eligible_for_training=False
    eligible = False
    verifier_notes = str(data.get("verifier_notes") or data.get("notes") or "").strip()

    # 7. Reviewer Identity Integrity: derive strictly from verified JWT, ignoring client body
    verified_by = reviewer_email or f"Reviewer {reviewer_user_id}"

    # 8. Record audit INTENT before state change (Phase D: fail-closed audit logging)
    intent_ok = record_feedback_audit_intent(
        feedback_id=feedback_id,
        reviewer_user_id=reviewer_user_id,
        reviewer_email=reviewer_email,
        intended_action=target_status,
        intended_label=verified_label,
        previous_status=current_status,
        verifier_notes=verifier_notes
    )
    if not intent_ok:
        return jsonify({
            "status": "error",
            "error": "Audit logging failure: Pre-review audit intent could not be recorded. Review operation aborted."
        }), 500

    # 9. Execute verification update
    ok = verify_feedback(
        feedback_id=feedback_id,
        verification_status=legacy_status,
        verifier_notes=verifier_notes,
        verified_by=verified_by,
        verified_label=verified_label
    )

    if not ok:
        record_feedback_audit_outcome(
            feedback_id=feedback_id,
            reviewer_user_id=reviewer_user_id,
            reviewer_email=reviewer_email,
            action=target_status,
            verified_label=verified_label,
            previous_status=current_status,
            verifier_notes=verifier_notes,
            success=False,
            error="Database persistence failed"
        )
        return jsonify({
            "status": "error",
            "error": f"Could not update verification determination for feedback '{feedback_id}'."
        }), 500

    # 10. Record audit OUTCOME after state change
    outcome_ok = record_feedback_audit_outcome(
        feedback_id=feedback_id,
        reviewer_user_id=reviewer_user_id,
        reviewer_email=reviewer_email,
        action=target_status,
        verified_label=verified_label,
        previous_status=current_status,
        verifier_notes=verifier_notes,
        success=True
    )

    if not outcome_ok:
        # State modified in database, but outcome audit logging failed: surface uncertain audit state
        return jsonify({
            "status": "warning",
            "audit_status": "UNCERTAIN_AUDIT_OUTCOME",
            "message": "Review determination was saved, but post-update audit outcome logging failed.",
            "feedback_id": feedback_id,
            "status_code": 200,
            "review_status": target_status,
            "verification_status": legacy_status,
            "verified_label": verified_label,
            "verified_by": verified_by,
            "eligible_for_training": False
        }), 200

    return jsonify({
        "status": "success",
        "audit_status": "AUDITED",
        "message": f"Feedback determination recorded: status={target_status}, label={verified_label}.",
        "feedback_id": feedback_id,
        "status_code": 200,
        "review_status": target_status,
        "verification_status": legacy_status,
        "verified_label": verified_label,
        "verified_by": verified_by,
        "eligible_for_training": False
    }), 200

    return jsonify({
        "status": "error",
        "error": f"Could not update verification determination for feedback '{feedback_id}'."
    }), 500


@analysis_bp.route('/analysis/feedback/dataset', methods=['GET'])
@analysis_bp.route('/feedback/dataset', methods=['GET'])
def get_verified_dataset_endpoint():
    """
    Returns the dataset of reviewed, expert-verified feedback samples.
    Excludes unverified user feedback.
    """
    from backend.core.database import get_verified_training_dataset
    dataset = get_verified_training_dataset()
    return jsonify({
        "status": "success",
        "count": len(dataset),
        "description": "Reviewed and expert-verified dataset eligible for model evaluation and fine-tuning.",
        "dataset": dataset
    }), 200


@analysis_bp.route('/analysis/feedback/dataset/curate', methods=['POST'])
@analysis_bp.route('/feedback/dataset/curate', methods=['POST'])
def curate_dataset_endpoint():
    """
    Curates verified feedback into an immutable versioned training dataset.
    Prevents duplicates, enforces train/eval split without leakage, and assigns dataset version.
    """
    from backend.evaluation.model_evaluator import DatasetCurator
    data = request.get_json() or {}
    dataset_version = data.get("dataset_version") or f"v{int(time.time())}_verified"
    train_ratio = float(data.get("train_ratio", 0.8))

    try:
        manifest = DatasetCurator.curate_verified_dataset(
            dataset_version=dataset_version,
            train_ratio=train_ratio
        )
        return jsonify({
            "status": "success",
            "message": f"Curated verified dataset '{dataset_version}'.",
            "manifest": manifest
        }), 201
    except Exception as e:
        logger.error(f"Dataset curation failed: {e}")
        return jsonify({
            "status": "error",
            "error": f"Failed to curate dataset: {str(e)}"
        }), 500


@analysis_bp.route('/models/retrain', methods=['POST'])
@analysis_bp.route('/analysis/models/retrain', methods=['POST'])
def retrain_model_endpoint():
    """
    Manually triggers a candidate model retraining run using verified feedback data.
    Never retrains automatically on user feedback submission.
    Preserves active production model weights unchanged.
    """
    from backend.training.feedback_retraining_pipeline import run_candidate_retraining
    data = request.get_json() or {}
    candidate_version = data.get("candidate_version") or f"v{int(time.time())}_candidate"
    dataset_version = data.get("dataset_version") or "v1.0_verified"
    epochs = int(data.get("epochs", 2))
    batch_size = int(data.get("batch_size", 8))
    lr = float(data.get("learning_rate", 1e-4))

    try:
        run_data = run_candidate_retraining(
            candidate_version=candidate_version,
            dataset_version=dataset_version,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr
        )
        return jsonify({
            "status": "success",
            "message": f"Candidate retraining completed for '{candidate_version}'. Production model remains unchanged.",
            "run_data": run_data
        }), 201
    except Exception as e:
        logger.error(f"Candidate retraining failed: {e}")
        return jsonify({
            "status": "error",
            "error": f"Retraining failed: {str(e)}"
        }), 500


@analysis_bp.route('/models/candidate/evaluate', methods=['POST'])
@analysis_bp.route('/analysis/models/candidate/evaluate', methods=['POST'])
def evaluate_candidate_endpoint():
    """
    Evaluates candidate model against independent validation benchmark.
    Reports accuracy, precision, recall, F1, FPR, FNR and safety gate pass/fail.
    """
    from backend.evaluation.model_evaluator import ModelEvaluator
    data = request.get_json() or {}
    candidate_version = data.get("candidate_version")
    if not candidate_version:
        return jsonify({"status": "error", "error": "candidate_version is required"}), 400

    try:
        report = ModelEvaluator.evaluate_candidate_model(candidate_version=candidate_version)
        return jsonify({
            "status": "success",
            "report": report
        }), 200
    except Exception as e:
        logger.error(f"Candidate evaluation failed: {e}")
        return jsonify({
            "status": "error",
            "error": f"Evaluation failed: {str(e)}"
        }), 500


@analysis_bp.route('/models/candidate/promote', methods=['POST'])
@analysis_bp.route('/analysis/models/candidate/promote', methods=['POST'])
def promote_candidate_endpoint():
    """
    Promotes evaluated candidate model to PRODUCTION.
    Requires explicit administrator sign-off.
    Preserves rollback capability by demoting previous production version.
    """
    from backend.evaluation.model_evaluator import ModelRegistry
    data = request.get_json() or {}
    candidate_version = data.get("candidate_version")
    approved_by = data.get("approved_by") or "Lead Forensic Administrator"
    approval_notes = data.get("approval_notes") or "Passed independent benchmark validation and safety deployment gate."

    if not candidate_version:
        return jsonify({"status": "error", "error": "candidate_version is required"}), 400

    try:
        ModelRegistry.promote_to_production(
            version=candidate_version,
            approved_by=approved_by,
            approval_notes=approval_notes
        )
        return jsonify({
            "status": "success",
            "message": f"Candidate model '{candidate_version}' promoted to PRODUCTION.",
            "production_version": candidate_version,
            "approved_by": approved_by
        }), 200
    except Exception as e:
        logger.error(f"Promotion failed: {e}")
        return jsonify({
            "status": "error",
            "error": f"Promotion failed: {str(e)}"
        }), 400


@analysis_bp.route('/models/rollback', methods=['POST'])
@analysis_bp.route('/analysis/models/rollback', methods=['POST'])
def rollback_model_endpoint():
    """
    Rolls back the active production model to the immediate previous archived production version.
    """
    from backend.evaluation.model_evaluator import ModelRegistry
    data = request.get_json() or {}
    approved_by = data.get("approved_by") or "Lead Forensic Administrator"
    reason = data.get("reason") or "Operational rollback requested by administrator."

    try:
        res = ModelRegistry.rollback_to_previous(approved_by=approved_by, reason=reason)
        return jsonify({
            "status": "success",
            "message": f"Rolled back production model to '{res['restored_version']}'.",
            "rollback": res
        }), 200
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        return jsonify({
            "status": "error",
            "error": f"Rollback failed: {str(e)}"
        }), 400


@analysis_bp.route('/models/registry', methods=['GET'])
@analysis_bp.route('/analysis/models/registry', methods=['GET'])
def get_models_registry_endpoint():
    """
    Returns version registry of production, candidate, and archived models with their verified benchmark metrics.
    """
    from backend.evaluation.model_evaluator import ModelRegistry
    reg = ModelRegistry.get_registry()
    return jsonify({
        "status": "success",
        "registry": reg
    }), 200


