import os
import time
import uuid
import json
import base64
import datetime
import cv2
import numpy as np
import torch
from flask import Blueprint, request, jsonify, send_from_directory

from backend.core.database import (
    create_monitoring_session,
    update_monitoring_session,
    log_monitoring_event,
    get_user_monitoring_sessions
)
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.face_detector import FaceDetector
from backend.models.image_model import ImageModel
from backend.agents.media_validation_agent import MediaValidationAgent
from backend.agents.image_agent import ImageAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.artifact_agent import ArtifactConsistencyAgent
from backend.agents.fusion_agent import FusionAgent
from backend.core.decision_system import SafetyDecisionSystem, DecisionState

from backend.utils.config import PipelineConfig
from backend.config import Config
from backend.utils.logger import setup_logger

logger = setup_logger("MonitorAPI")
monitor_bp = Blueprint('monitor', __name__)

# Lazy component loader for fast server startup
_monitor_components = None

def get_monitor_components():
    global _monitor_components
    if _monitor_components is None:
        _monitor_components = {
            "preprocessor": ImagePreprocessor(),
            "image_agent": ImageAgent(),
            "artifact_agent": ArtifactConsistencyAgent(),
            "face_detector": FaceDetector(),
            "decision_system": SafetyDecisionSystem(),
        }
    return _monitor_components

# Persistent session-level face track state: session_id -> {track_id -> track_dict}
_session_face_tracks = {}

def get_or_update_track(session_id: str, bbox, img_w: int, img_h: int, score: float, raw_score: float, face_crop=None, claimed_track_ids=None):
    """
    Spatial IoU / Centroid matching to maintain consistent tracked-face identities,
    preserves separate per-face historical score buffers, inter-frame boundary seam flicker,
    and oral cavity texture variance across video frames.
    """
    global _session_face_tracks
    now = time.time()
    if session_id not in _session_face_tracks:
        _session_face_tracks[session_id] = {}

    session_tracks = _session_face_tracks[session_id]

    # Prune stale tracks older than 4 seconds to prevent ghost tracks or memory leakage
    stale_keys = [t_id for t_id, t_info in session_tracks.items() if now - t_info.get("last_seen", 0) > 4.0]
    for sk in stale_keys:
        session_tracks.pop(sk, None)

    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0

    best_track_id = None
    best_dist = float("inf")

    for t_id, t_info in session_tracks.items():
        if claimed_track_ids and t_id in claimed_track_ids:
            continue
        lx, ly, lw, lh = t_info["last_box"]
        lcx, lcy = lx + lw / 2.0, ly + lh / 2.0
        dist = np.hypot((cx - lcx) / max(1, img_w), (cy - lcy) / max(1, img_h))

        xA = max(x, lx)
        yA = max(y, ly)
        xB = min(x + w, lx + lw)
        yB = min(y + h, ly + lh)
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = w * h + lw * lh - inter
        iou_val = inter / float(union) if union > 0 else 0.0

        if (iou_val > 0.20 or dist < 0.20) and dist < best_dist:
            best_dist = dist
            best_track_id = t_id

    thumb = None
    mouth_var = 50.0
    eye_sharp = 50.0
    if face_crop is not None and face_crop.shape[0] >= 20 and face_crop.shape[1] >= 20:
        thumb = cv2.resize(face_crop, (128, 128))
        # Mouth region (lower-middle 1/3)
        mouth_crop = thumb[int(128*0.65):int(128*0.92), int(128*0.25):int(128*0.75)]
        mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
        mouth_var = float(cv2.Laplacian(mouth_gray, cv2.CV_64F).var())
        # Eye region (upper 1/3)
        eye_crop = thumb[int(128*0.20):int(128*0.48), int(128*0.15):int(128*0.85)]
        eye_gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        eye_sharp = float(cv2.Laplacian(eye_gray, cv2.CV_64F).var())

    if best_track_id is None:
        idx = len(session_tracks) + 1
        best_track_id = f"T{idx}"
        while best_track_id in session_tracks:
            idx += 1
            best_track_id = f"T{idx}"
        session_tracks[best_track_id] = {
            "track_id": best_track_id,
            "created_at": now,
            "last_box": bbox,
            "last_crop": thumb,
            "history_scores": [],
            "history_raw": [],
            "boundary_flickers": [],
            "mouth_textures": [],
            "eye_sharpnesses": [],
            "frames_count": 0,
        }

    if claimed_track_ids is not None:
        claimed_track_ids.add(best_track_id)

    t = session_tracks[best_track_id]

    # Inter-frame boundary seam flicker measurement
    boundary_flicker = 0.0
    if thumb is not None and t.get("last_crop") is not None:
        try:
            diff = cv2.absdiff(thumb, t["last_crop"])
            # Outer 12 pixels border seam
            mask = np.zeros((128, 128), dtype=bool)
            mask[:12, :] = True
            mask[-12:, :] = True
            mask[:, :12] = True
            mask[:, -12:] = True
            boundary_flicker = float(np.mean(diff[mask]))
            t["boundary_flickers"].append(boundary_flicker)
            if len(t["boundary_flickers"]) > 15:
                t["boundary_flickers"].pop(0)
        except Exception:
            pass

    t["last_crop"] = thumb
    t["last_box"] = bbox
    t["last_seen"] = now
    t["history_scores"].append(score)
    t["history_raw"].append(raw_score)
    t["mouth_textures"].append(mouth_var)
    t["eye_sharpnesses"].append(eye_sharp)
    t["frames_count"] += 1
    if len(t["history_scores"]) > 15:
        t["history_scores"].pop(0)
        t["history_raw"].pop(0)
        t["mouth_textures"].pop(0)
        t["eye_sharpnesses"].pop(0)

    track_avg = float(np.mean(t["history_scores"]))
    track_max = float(np.max(t["history_scores"]))
    mean_flicker = float(np.mean(t["boundary_flickers"])) if t["boundary_flickers"] else boundary_flicker
    mean_mouth_var = float(np.mean(t["mouth_textures"])) if t["mouth_textures"] else mouth_var

    return best_track_id, {
        "track_id": best_track_id,
        "track_avg_score": round(track_avg, 1),
        "track_max_score": round(track_max, 1),
        "frames_tracked": t["frames_count"],
        "boundary_flicker": round(mean_flicker, 1),
        "mouth_texture": round(mean_mouth_var, 1),
        "eye_sharpness": round(eye_sharp, 1),
    }

def update_track_verdict(session_id: str, track_id: str, verdict: str, risk: str, reason: str, score: float):
    """Stores the latest evaluated verdict and forensic evidence for a tracked face."""
    global _session_face_tracks
    sid = session_id or "default"
    if sid in _session_face_tracks and track_id in _session_face_tracks[sid]:
        t = _session_face_tracks[sid][track_id]
        t["last_verdict"] = verdict
        t["last_risk"] = risk
        t["last_reason"] = reason
        t["last_score"] = score

# Read optimized threshold from model registry
def get_optimized_threshold() -> float:
    reg_path = PipelineConfig.REGISTRY_PATH
    if os.path.exists(reg_path):
        try:
            with open(reg_path, 'r') as f:
                data = json.load(f)
                return data.get('image', {}).get('optimized_threshold', 0.04)
        except Exception:
            pass
    return 0.04

OPTIMIZED_THRESHOLD = get_optimized_threshold()

@monitor_bp.route('/start-session', methods=['POST'])
def start_session():
    """Initializes a new real-time monitoring session."""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    user_name = data.get('user_name', 'Anonymous User')
    
    session_id = f"mon_{uuid.uuid4().hex[:12]}"
    try:
        success = create_monitoring_session(session_id, user_id, user_name)
    except Exception as e:
        logger.warning(f"Remote monitoring session persistence skipped: {e}")
        success = False
    
    # Return 200 with session_id so monitoring proceeds reliably in both cloud and local modes
    return jsonify({
        "status": "success",
        "session_id": session_id,
        "message": "Monitoring session initialized" if success else "Monitoring session initialized (local mode)"
    }), 200

@monitor_bp.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Real-time video frame analysis endpoint.
    Accepts base64-encoded JPEG image and returns detected face bounding boxes,
    per-face deepfake verdicts, genuine calibrated scores, latency, and agent trace.
    """
    start_time = time.time()
    data = request.get_json() or {}
    
    frame_base64 = data.get('frame')
    session_id = data.get('session_id')
    client_timestamp = data.get('timestamp') or datetime.datetime.now().isoformat()
    
    if not frame_base64:
        return jsonify({
            "status": "error",
            "error": "No frame data provided"
        }), 400
        
    try:
        # Strip header if present
        if ',' in frame_base64:
            frame_base64 = frame_base64.split(',', 1)[1]
            
        img_bytes = base64.b64decode(frame_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None or img_bgr.shape[0] < 50 or img_bgr.shape[1] < 50:
            return jsonify({
                "status": "warning",
                "verdict": "INVALID_FRAME",
                "faces": [],
                "faces_count": 0,
                "detection_score": 0.0,
                "risk_level": "LOW",
                "latency_ms": int((time.time() - start_time) * 1000),
                "timestamp": client_timestamp,
                "agents": {
                    "validation": "Failed",
                    "face_detector": "Skipped",
                    "image": "Skipped",
                    "confidence": "Skipped",
                    "artifact": "Skipped",
                    "fusion": "Skipped"
                }
            }), 200

        img_h, img_w = img_bgr.shape[:2]

        # Unpack lazy-initialized components
        components = get_monitor_components()
        preprocessor = components["preprocessor"]
        image_agent = components["image_agent"]
        artifact_agent = components["artifact_agent"]
        face_detector = components["face_detector"]
        decision_system = components["decision_system"]

        agent_trace = {
            "validation": "Success",
            "face_detector": "Success",
            "image": "Success",
            "confidence": "Success",
            "artifact": "Skipped",
            "fusion": "Skipped"
        }

        # Step 1: Detect Faces in Frame
        faces = face_detector.detect_faces(img_bgr)
        
        face_results = []
        claimed_track_ids = set()
        if faces:
            # Process up to 4 faces to keep latency bounded and server responsive
            for idx, bbox in enumerate(faces[:4]):
                x, y, bw, bh = bbox
                crop = face_detector.crop_face(img_bgr, bbox)
                if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                try:
                    rgb_crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (224, 224))
                    tensor = preprocessor.apply_augmentation_and_normalization(rgb_crop, is_training=False).unsqueeze(0)
                    device = getattr(image_agent, "device", "cuda" if torch.cuda.is_available() else "cpu")
                    tensor = tensor.to(device)

                    with torch.no_grad():
                        logits = image_agent.model(tensor)
                        probs = torch.softmax(logits, dim=-1)[0]
                        fake_prob = float(probs[1].item())

                    # If ambiguous score, run forensic analysis with full frame context and bbox
                    is_ambiguous = (0.35 <= fake_prob <= 0.65)
                    artifact_res = None
                    if is_ambiguous:
                        agent_trace["artifact"] = "Success"
                        agent_trace["fusion"] = "Success"
                        artifact_res = artifact_agent.analyze(img_array=img_bgr, face_bbox=bbox)

                    decision = decision_system.decide(
                        raw_fake_prob=fake_prob,
                        forensic_result=artifact_res.get("forensic_details") if artifact_res else None,
                    )

                    raw_score = round(fake_prob * 100, 1)
                    calibrated_score = round(decision.fake_probability * 100, 1) if decision.fake_probability is not None else raw_score
                    conf_val = round(decision.confidence, 1) if decision.confidence is not None else None
                    unc_val = round(decision.uncertainty, 2) if decision.uncertainty is not None else 0.5

                    # Update persistent spatial track history with empirical face crop measurements
                    track_id, track_meta = get_or_update_track(
                        session_id or "default", bbox, img_w, img_h, calibrated_score, raw_score, face_crop=crop, claimed_track_ids=claimed_track_ids
                    )

                    # Temporal verdict stabilization across consecutive frames
                    avg_s = track_meta["track_avg_score"]
                    max_s = track_meta["track_max_score"]
                    frames_tr = track_meta["frames_tracked"]
                    flicker_val = track_meta["boundary_flicker"]
                    mouth_tex_val = track_meta["mouth_texture"]

                    # Decision logic strictly adhering to evidence-based forensic thresholds:
                    # 1. FAKE: Sufficient evidence of manipulation (neural classifier, boundary seam flicker, or artifact agent)
                    is_fake = (
                        avg_s >= 50.0
                        or max_s >= 55.0
                        or (avg_s >= 38.0 and flicker_val >= 18.0)
                        or (frames_tr >= 2 and avg_s >= 42.0 and flicker_val >= 16.0)
                        or (decision.prediction in [DecisionState.FAKE, DecisionState.DEEPFAKE])
                        or (artifact_res and artifact_res.get("forensic_details", {}).get("tampering_detected"))
                    )

                    # 2. REAL: Authenticity sufficiently supported
                    # Clean neural prediction (REAL/LIKELY_REAL), consistently low score, stable boundaries
                    is_real = (
                        not is_fake
                        and avg_s <= 26.0
                        and max_s <= 35.0
                        and flicker_val < 32.0
                        and decision.prediction in [DecisionState.REAL, DecisionState.LIKELY_REAL, "REAL", "LIKELY_REAL"]
                    )

                    if is_fake:
                        face_verdict = "FAKE"
                        face_risk = "CRITICAL" if (avg_s >= 75.0 or (decision.confidence and decision.confidence >= 80.0)) else "HIGH"
                        if flicker_val >= 18.0 and avg_s >= 38.0:
                            face_reason = f"Boundary seam flicker anomaly ({flicker_val:.1f}) & elevated synthetic score ({avg_s:.1f}%)"
                        elif max_s >= 55.0 or avg_s >= 50.0:
                            face_reason = f"Elevated neural deepfake probability ({avg_s:.1f}%, peak {max_s:.1f}%)"
                        else:
                            face_reason = f"Corroborated facial manipulation anomalies ({avg_s:.1f}%)"
                    elif is_real:
                        face_verdict = "REAL"
                        face_risk = "LOW"
                        face_reason = f"Authentic facial continuity (avg score {avg_s:.1f}%, conf {conf_val or 75.0}%) & natural boundary transitions"
                    else:
                        face_verdict = "INCONCLUSIVE"
                        face_risk = "MEDIUM"
                        if 26.0 < avg_s < 50.0:
                            face_reason = f"Score ({avg_s:.1f}%) in indeterminate margin [26% - 50%] without decisive corroboration"
                        else:
                            face_reason = "Evidence insufficient to certify as REAL or FAKE"

                    # Update persistent track status
                    update_track_verdict(session_id or "default", track_id, face_verdict, face_risk, face_reason, calibrated_score)

                    face_results.append({
                        "id": track_id,
                        "track_id": track_id,
                        "box": [int(x), int(y), int(bw), int(bh)],
                        "box_normalized": [
                            round(x / img_w, 4),
                            round(y / img_h, 4),
                            round(bw / img_w, 4),
                            round(bh / img_h, 4)
                        ],
                        "verdict": face_verdict,
                        "prediction": face_verdict,
                        "detection_score": calibrated_score,
                        "raw_score": raw_score,
                        "confidence": conf_val if face_verdict != "INCONCLUSIVE" else None,
                        "uncertainty": unc_val if face_verdict != "INCONCLUSIVE" else 0.75,
                        "risk_level": face_risk,
                        "track_avg_score": track_meta["track_avg_score"],
                        "track_max_score": track_meta["track_max_score"],
                        "track_frames": track_meta["frames_tracked"],
                        "boundary_flicker": track_meta["boundary_flicker"],
                        "mouth_texture": track_meta["mouth_texture"],
                        "eye_sharpness": track_meta["eye_sharpness"],
                        "label": f"[{track_id}: {face_verdict}] {calibrated_score:.1f}%",
                        "timestamp": client_timestamp,
                        "reason": face_reason,
                        "evidence": face_reason,
                    })
                except Exception as fe:
                    logger.warning(f"Error processing face {idx}: {fe}")
                    track_id, track_meta = get_or_update_track(
                        session_id or "default", bbox, img_w, img_h, 50.0, 50.0
                    )
                    face_results.append({
                        "id": track_id,
                        "track_id": track_id,
                        "box": [int(x), int(y), int(bw), int(bh)],
                        "box_normalized": [
                            round(x / img_w, 4),
                            round(y / img_h, 4),
                            round(bw / img_w, 4),
                            round(bh / img_h, 4)
                        ],
                        "verdict": "INCONCLUSIVE",
                        "prediction": "ERROR",
                        "detection_score": 50.0,
                        "raw_score": 50.0,
                        "confidence": None,
                        "uncertainty": 1.0,
                        "risk_level": "MEDIUM",
                        "track_avg_score": track_meta["track_avg_score"],
                        "track_max_score": track_meta["track_max_score"],
                        "track_frames": track_meta["frames_tracked"],
                        "label": f"[{track_id}: INCONCLUSIVE] Error",
                        "timestamp": client_timestamp,
                        "reason": f"Face processing exception: {str(fe)}",
                        "evidence": f"Face processing exception: {str(fe)}"
                    })

        # Step 2: Formulate Frame-Level Aggregated Verdict
        if not face_results:
            overall_verdict = "NO_FACE"
            overall_risk = "LOW"
            overall_score = 0.0
            overall_conf = None
            overall_uncertainty = 1.0
            overall_reason = "NO FACE DETECTED / ANALYSIS UNAVAILABLE: Facial bounding required for deepfake verification."
        else:
            # Evidence-driven multi-face aggregation across all active tracks in session (within 3.5s)
            session_tracks = _session_face_tracks.get(session_id or "default", {})
            now_ts = time.time()
            active_tracks = [t for t in session_tracks.values() if now_ts - t.get("last_seen", 0) <= 3.5]

            any_fake = any(f["verdict"] == "FAKE" for f in face_results) or any(t.get("last_verdict") == "FAKE" for t in active_tracks)
            any_inconclusive = any(f["verdict"] == "INCONCLUSIVE" for f in face_results) or any(t.get("last_verdict") == "INCONCLUSIVE" for t in active_tracks)

            max_score = max(f["detection_score"] for f in face_results)
            suspicious_face = max(face_results, key=lambda f: f["detection_score"])

            # Check if any active track in session was flagged as fake
            fake_tracks = [t for t in active_tracks if t.get("last_verdict") == "FAKE"]
            if fake_tracks:
                most_suspicious_track = max(fake_tracks, key=lambda t: t.get("last_score", 0.0))
                max_score = max(max_score, most_suspicious_track.get("last_score", 50.0))

            if any_fake:
                overall_verdict = "DEEPFAKE"
                overall_risk = "CRITICAL" if max_score >= 75.0 else "HIGH"
                fake_face = next((f for f in face_results if f["verdict"] == "FAKE"), None)
                if fake_face:
                    overall_reason = f"Face Track {fake_face['id']} FLAGGED AS FAKE: {fake_face.get('reason', 'Manipulation detected')}"
                elif fake_tracks:
                    overall_reason = f"Face Track {fake_tracks[0]['track_id']} FLAGGED AS FAKE: {fake_tracks[0].get('last_reason', 'Temporal manipulation detected')}"
                else:
                    overall_reason = "Synthetic manipulation detected across sequential face tracks."
            elif any_inconclusive:
                overall_verdict = "INCONCLUSIVE"
                overall_risk = "MEDIUM"
                overall_reason = f"Analysis Inconclusive ({suspicious_face['id']}): {suspicious_face.get('reason', 'Awaiting temporal continuity')}"
            else:
                overall_verdict = "REAL"
                overall_risk = "LOW"
                overall_reason = f"Authentic Verification ({suspicious_face['id']}): {suspicious_face.get('reason', 'Natural facial continuity certified')}"

            overall_score = max_score
            overall_conf = suspicious_face.get("confidence") if overall_verdict != "INCONCLUSIVE" else None
            overall_uncertainty = suspicious_face.get("uncertainty", 0.2) if overall_verdict != "INCONCLUSIVE" else 0.75

        latency_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "status": "success",
            "verdict": overall_verdict,
            "prediction": overall_verdict,
            "detection_score": overall_score,
            "confidence": overall_conf,
            "uncertainty": overall_uncertainty,
            "risk_level": overall_risk,
            "latency_ms": latency_ms,
            "timestamp": client_timestamp,
            "reason": overall_reason,
            "frame_size": [img_w, img_h],
            "faces_count": len(face_results),
            "faces": face_results,
            "agents": agent_trace,
            "providers": {
                "local_classifier": "AVAILABLE",
                "face_detector": "ACTIVE",
                "safety_engine": "ACTIVE"
            }
        }), 200

    except Exception as e:
        logger.error(f"Error processing real-time frame: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "Frame processing error",
            "verdict": "ERROR",
            "faces": [],
            "faces_count": 0,
            "detection_score": 0.0,
            "confidence": None,
            "uncertainty": 1.0,
            "risk_level": "LOW",
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": client_timestamp,
            "agents": {
                "validation": "Failed",
                "face_detector": "Failed",
                "image": "Failed",
                "confidence": "Skipped",
                "artifact": "Skipped",
                "fusion": "Skipped"
            }
        }), 200

@monitor_bp.route('/samples', methods=['GET'])
def get_sample_videos():
    """Returns verified sample videos available for real-time recorded video testing."""
    upload_dir = Config.UPLOAD_FOLDER
    samples = []
    
    candidates = [
        {
            "id": "genuine_interview",
            "title": "Genuine Video Sample (Broadcast/Interview)",
            "filename": "3cd9b7ec-16b3-467f-ade3-df1b246ca5dd_989_993.mp4",
            "expected_verdict": "REAL",
            "description": "Authentic video recording with natural facial features and consistent temporal continuity."
        },
        {
            "id": "fake_face_swap",
            "title": "Deepfake Video Sample (FaceForensics++ C23)",
            "filename": "ff_deepfake_sample_01.mp4",
            "expected_verdict": "FAKE",
            "description": "Synthetically manipulated video clip exhibiting temporal boundary seam flicker, oral texture blurring, and face-swap manipulation."
        },
        {
            "id": "genuine_clip",
            "title": "Genuine Video Clip (Standard Validation)",
            "filename": "test_valid_video_clip.mp4",
            "expected_verdict": "REAL",
            "description": "Standard benchmark authentic video sequence."
        }
    ]
    
    for c in candidates:
        filepath = os.path.join(upload_dir, c["filename"])
        if os.path.exists(filepath):
            samples.append({
                "id": c["id"],
                "title": c["title"],
                "filename": c["filename"],
                "expected_verdict": c["expected_verdict"],
                "description": c["description"],
                "url": f"/api/monitor/sample/{c['filename']}"
            })
            
    return jsonify({
        "status": "success",
        "samples": samples
    }), 200

@monitor_bp.route('/sample/<path:filename>', methods=['GET'])
def serve_sample_video(filename):
    """Safely serves sample video file with range-request streaming support."""
    upload_dir = Config.UPLOAD_FOLDER
    safe_name = os.path.basename(filename)
    file_path = os.path.join(upload_dir, safe_name)
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "error": "Sample video not found"}), 404
        
    return send_from_directory(upload_dir, safe_name, mimetype='video/mp4')

@monitor_bp.route('/reset-session', methods=['POST'])
def reset_session():
    """Resets tracking histories to prevent stale predictions leaking between videos."""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    if session_id:
        _session_face_tracks.pop(session_id, None)
    else:
        _session_face_tracks.clear()
    _session_face_tracks.pop("default", None)
    return jsonify({"status": "success", "message": "Face tracking histories reset"}), 200

@monitor_bp.route('/stop-session', methods=['POST'])
def stop_session():
    """Finalizes a monitoring session and saves aggregated statistics."""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    stats = data.get('stats') or {}
    events = data.get('events') or []
    
    if not session_id:
        return jsonify({"status": "error", "error": "session_id required"}), 400
        
    _session_face_tracks.pop(session_id, None)
    _session_face_tracks.pop("default", None)
    update_success = update_monitoring_session(session_id, stats)
    
    # Save suspicious detection events if any
    for evt in events:
        evt_id = f"evt_{uuid.uuid4().hex[:10]}"
        log_monitoring_event(
            session_id=session_id,
            event_id=evt_id,
            verdict=evt.get('verdict', 'SUSPICIOUS'),
            confidence=evt.get('detection_score', 0.0),
            risk_level=evt.get('risk_level', 'MEDIUM'),
            latency_ms=evt.get('latency_ms', 0.0),
            trigger_reason=evt.get('trigger_reason', 'Temporal risk escalation')
        )
        
    return jsonify({
        "status": "success",
        "message": "Monitoring session finalized successfully"
    }), 200

@monitor_bp.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    """Fetches past monitoring sessions for a user."""
    sessions = get_user_monitoring_sessions(user_id)
    return jsonify({
        "status": "success",
        "sessions": sessions
    }), 200

