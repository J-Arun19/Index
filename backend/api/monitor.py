import os
import time
import uuid
import json
import base64
import datetime
import cv2
import numpy as np
import torch
from flask import Blueprint, request, jsonify

from backend.core.database import (
    create_monitoring_session,
    update_monitoring_session,
    log_monitoring_event,
    get_user_monitoring_sessions
)
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.models.image_model import ImageModel
from backend.agents.media_validation_agent import MediaValidationAgent
from backend.agents.image_agent import ImageAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.artifact_agent import ArtifactConsistencyAgent
from backend.agents.fusion_agent import FusionAgent

from backend.utils.config import PipelineConfig
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
        }
    return _monitor_components

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
    success = create_monitoring_session(session_id, user_id, user_name)
    
    if success:
        return jsonify({
            "status": "success",
            "session_id": session_id,
            "message": "Monitoring session initialized"
        }), 200
    else:
        return jsonify({
            "status": "error",
            "error": "Failed to create monitoring session"
        }), 500

@monitor_bp.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Lightweight real-time camera frame analysis endpoint.
    Accepts base64-encoded JPEG image and returns detection score, risk level,
    latency, and agent execution trace.
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
                "detection_score": 0.0,
                "risk_level": "LOW",
                "latency_ms": int((time.time() - start_time) * 1000),
                "timestamp": client_timestamp,
                "agents": {
                    "validation": "Failed",
                    "image": "Skipped",
                    "confidence": "Skipped",
                    "artifact": "Skipped",
                    "fusion": "Skipped"
                }
            }), 200

        # Step 1: Media Validation
        agent_trace = {
            "validation": "Success",
            "image": "Success",
            "confidence": "Success",
            "artifact": "Skipped",
            "fusion": "Skipped"
        }

        # Unpack lazy-initialized components
        components = get_monitor_components()
        preprocessor = components["preprocessor"]
        image_agent = components["image_agent"]
        artifact_agent = components["artifact_agent"]

        # Preprocess frame tensor (Face crop or resize 224x224 RGB ImageNet normalized)
        rgb_img = preprocessor.get_cropped_resized_rgb(img_bgr)
        tensor = preprocessor.apply_augmentation_and_normalization(rgb_img, is_training=False).unsqueeze(0)

        # Step 2: Image Model Inference
        with torch.no_grad():
            logits = image_agent.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            fake_prob = float(probs[1].item())
            real_prob = float(probs[0].item())

        # Step 3: Confidence Evaluation & Safety Decision System
        from backend.core.decision_system import SafetyDecisionSystem, DecisionState
        decision_system = SafetyDecisionSystem()
        
        # Check if score is ambiguous (between 0.35 and 0.65) to trigger forensic inspection
        is_ambiguous = (0.35 <= fake_prob <= 0.65)
        artifact_res = None
        if is_ambiguous:
            agent_trace["artifact"] = "Success"
            agent_trace["fusion"] = "Success"
            artifact_res = artifact_agent.analyze(img_array=img_bgr)

        # Execute three-state safety decision
        decision = decision_system.decide(
            raw_fake_prob=fake_prob,
            forensic_result=artifact_res.get("forensic_details") if artifact_res else None,
        )

        if decision.prediction == DecisionState.FAKE:
            verdict = "DEEPFAKE"
            risk_level = "CRITICAL" if decision.confidence >= 80.0 else "HIGH"
        elif decision.prediction == DecisionState.UNCERTAIN:
            verdict = "UNCERTAIN"
            risk_level = "MEDIUM"
        else:
            verdict = "LIKELY_REAL"
            risk_level = "LOW"

        detection_score = round(decision.fake_probability * 100, 2)
        latency_ms = int((time.time() - start_time) * 1000)

        return jsonify({
            "status": "success",
            "verdict": verdict,
            "prediction": decision.prediction,
            "detection_score": detection_score,
            "confidence": decision.confidence,
            "uncertainty": decision.uncertainty,
            "risk_level": risk_level,
            "latency_ms": latency_ms,
            "timestamp": client_timestamp,
            "reason": decision.reason,
            "user_facing_statement": decision.user_facing_statement,
            "agents": agent_trace
        }), 200

    except Exception as e:
        logger.error(f"Error processing real-time frame: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "Frame processing error",
            "verdict": "ERROR",
            "detection_score": 0.0,
            "risk_level": "LOW",
            "latency_ms": int((time.time() - start_time) * 1000),
            "timestamp": client_timestamp,
            "agents": {
                "validation": "Failed",
                "image": "Failed",
                "confidence": "Skipped",
                "artifact": "Skipped",
                "fusion": "Skipped"
            }
        }), 200

@monitor_bp.route('/stop-session', methods=['POST'])
def stop_session():
    """Finalizes a monitoring session and saves aggregated statistics."""
    data = request.get_json() or {}
    session_id = data.get('session_id')
    stats = data.get('stats') or {}
    events = data.get('events') or []
    
    if not session_id:
        return jsonify({"status": "error", "error": "session_id required"}), 400
        
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
