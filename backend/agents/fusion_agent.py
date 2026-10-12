import os
import numpy as np
from typing import Dict, Any, Optional
from backend.utils.config import PipelineConfig
from backend.core.decision_system import SafetyDecisionSystem, DecisionState, USER_FACING_STATEMENTS


class FusionAgent:
    """
    Fuses multi-agent forensic evidence using a transparent three-state decision architecture.
    Prevents false negatives by refusing to collapse ambiguous or contradictory
    evidence into confident REAL verdicts.
    """

    def __init__(self):
        self.decision_system = SafetyDecisionSystem()

    def fuse(self, context_or_agent_name, agent_output=None, analysis_id=None) -> Dict[str, Any]:
        """
        Fuses outputs from the specialized analysis agents.
        Applies calibrated evidence fusion and abstention rules.
        """
        if not isinstance(context_or_agent_name, str):
            # Context-driven execution
            context = context_or_agent_name
            media_type = context.media_type
            primary_name = "ImageAgent" if media_type.lower() == "image" else "VideoAgent"
            primary_res = context.image_agent_result if media_type.lower() == "image" else context.video_agent_result
            metadata_res = context.metadata_agent_result
            artifact_res = context.artifact_agent_result
            analysis_id = context.analysis_id
        else:
            # Linear execution fallback
            primary_name = context_or_agent_name
            primary_res = agent_output or {}
            metadata_res = None
            artifact_res = None

        # 1. Primary specialist signals
        primary_pred = primary_res.get("prediction") or primary_res.get("verdict", DecisionState.UNCERTAIN)
        # Normalize legacy names
        if primary_pred == "DEEPFAKE":
            primary_pred = DecisionState.FAKE
        elif primary_pred == "REAL":
            primary_pred = DecisionState.LIKELY_REAL

        # Media Suitability bypass: If media is unsupported, do not perform authenticity fusion
        if primary_pred in (DecisionState.NOT_SUITABLE, "NOT_SUITABLE"):
            final_verdict = DecisionState.NOT_SUITABLE
            user_statement = USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE]
            final_reason = primary_res.get("reason", "No suitable face detected")
            recommendation = primary_res.get(
                "recommendation",
                "Upload a clear image containing a visible human face."
            )
            media_category = primary_res.get("media_category", "Document / Non-face image")

            if analysis_id:
                try:
                    from backend.services.database_service import DatabaseService
                    DatabaseService.update_fusion_prediction(analysis_id, final_verdict, 0.0)
                except Exception as e:
                    logger.warning(f"FusionAgent: Failed to save final prediction to DB: {e}")

            agents_trace = [
                {"name": primary_name, "summary": "Media suitability evaluation: Unsupported (no face detected)."},
                {"name": "Fusion Agent", "summary": "Media is not suitable for facial analysis. Analysis not performed."}
            ]

            return {
                "prediction": final_verdict,
                "verdict": final_verdict,
                "final_verdict": final_verdict,
                "confidence": None,
                "confidence_score": None,
                "confidence_level": "Not Performed",
                "fake_probability": None,
                "real_probability": None,
                "final_probability": None,
                "raw_model_score": None,
                "uncertainty": None,
                "evidence": primary_res.get("evidence", ["Eligibility: Unsupported media (No detectable human face found in image)."]),
                "reason": final_reason,
                "disagreement_detected": False,
                "explanation": f"NOT SUITABLE FOR FACE ANALYSIS: {user_statement}",
                "user_facing_statement": user_statement,
                "recommendation": recommendation,
                "analysis_status": "Not performed",
                "face_detected": False,
                "media_category": media_category,
                "active_agents": [primary_name, "Fusion Agent"],
                "evidence_breakdown": {
                    "eligibility": {
                        "name": "Media Suitability Gate",
                        "status": "UNSUPPORTED",
                        "reason": final_reason,
                    }
                },
                "agents": agents_trace,
            }

        primary_fake_p = float(primary_res.get("fake_probability", primary_res.get("raw_model_score", 0.50)) or 0.50)
        primary_conf = float(primary_res.get("confidence", primary_res.get("confidence_score", 50.0)) or 50.0)

        evidence_breakdown = {
            "primary_specialist": {
                "name": primary_name,
                "verdict": primary_pred,
                "confidence": primary_conf,
                "fake_probability": primary_fake_p,
            }
        }

        weights = {"primary": PipelineConfig.WEIGHT_PRIMARY}
        probabilities = {"primary": primary_fake_p}
        evidence_items = list(primary_res.get("evidence", []))

        # 2. Metadata signals
        metadata_anomalies = False
        if metadata_res and metadata_res.get("status") != "Failed":
            metadata_anomalies = bool(metadata_res.get("anomalies_detected", False))
            p_meta = 0.70 if metadata_anomalies else 0.15
            weights["metadata"] = PipelineConfig.WEIGHT_METADATA
            probabilities["metadata"] = p_meta
            evidence_breakdown["metadata_specialist"] = {
                "name": "MetadataAgent",
                "anomalies_detected": metadata_anomalies,
                "probability": p_meta,
                "software_tag": metadata_res.get("software_tag"),
            }
            if metadata_anomalies:
                evidence_items.append(
                    f"Metadata: Editing signature detected ({metadata_res.get('software_tag') or 'Edited'})."
                )

        # 3. Artifact & Genuine Forensic signals
        artifact_anomalies = False
        artifact_score = 0.15
        if artifact_res and artifact_res.get("status") != "Failed":
            artifact_score = float(artifact_res.get("anomaly_score", 0.15))
            artifact_anomalies = artifact_score > 0.45
            weights["artifact"] = PipelineConfig.WEIGHT_ARTIFACT
            probabilities["artifact"] = artifact_score
            evidence_breakdown["artifact_specialist"] = {
                "name": "ArtifactConsistencyAgent",
                "anomaly_score": artifact_score,
                "probability": artifact_score,
            }
            for art in artifact_res.get("detected_artifacts", []):
                if art not in evidence_items:
                    evidence_items.append(art)

        # 4. Compute weighted fused probability
        sum_weights = sum(weights.values())
        norm_weights = {k: v / sum_weights for k, v in weights.items()}
        fused_fake_p = sum(probabilities[k] * norm_weights[k] for k in weights.keys())
        fused_fake_p = float(np.clip(fused_fake_p, 0.0, 1.0))

        # 5. Evaluate Disagreements & Conflicts (CRITICAL SAFETY LAYER)
        disagreement_detected = False
        conflict_reason = ""

        # Check if primary says LIKELY_REAL, but forensic or metadata says anomalous
        if primary_pred in (DecisionState.LIKELY_REAL, "REAL") and (artifact_anomalies or metadata_anomalies):
            disagreement_detected = True
            conflict_reason = (
                "Contradictory evidence: Primary classifier indicated authentic features, "
                "but independent forensic/metadata analysis detected manipulation signatures."
            )
        elif primary_pred == DecisionState.FAKE and not artifact_anomalies and not metadata_anomalies and primary_fake_p < 0.65:
            disagreement_detected = True
            conflict_reason = "Weak synthetic signals without corroboration from independent forensic analyzers."

        # 6. Determine Three-State Final Verdict
        tau_real = self.decision_system.calibrator.tau_real
        tau_fake = self.decision_system.calibrator.tau_fake

        # Rule 1: Quality or Primary Abstention must be respected
        if primary_pred == DecisionState.UNCERTAIN:
            final_verdict = DecisionState.UNCERTAIN
            final_reason = primary_res.get("reason", "Primary specialist abstained due to inconclusive evidence.")
        # Rule 2: Escalation to UNCERTAIN if contradictory evidence detected
        elif disagreement_detected:
            final_verdict = DecisionState.UNCERTAIN
            final_reason = f"Abstained due to evidence conflict: {conflict_reason}"
        # Rule 3: Boundary zone abstention
        elif tau_real <= fused_fake_p <= tau_fake:
            final_verdict = DecisionState.UNCERTAIN
            final_reason = (
                f"Fused probability ({fused_fake_p:.2f}) lies in the ambiguous decision margin [{tau_real:.2f} - {tau_fake:.2f}]. "
                "Evidence is inconclusive."
            )
        # Rule 4: Decisive FAKE
        elif fused_fake_p > tau_fake:
            final_verdict = DecisionState.FAKE
            final_reason = f"Multi-agent consensus confirms synthetic manipulation (fused probability: {fused_fake_p:.2f})."
        # Rule 5: Decisive LIKELY_REAL
        else:
            final_verdict = DecisionState.LIKELY_REAL
            final_reason = (
                "No strong evidence of manipulation was detected across classifier and forensic checks. "
                "Note: This does not prove authenticity."
            )

        # 7. Confidence & Uncertainty Computation
        conf_ratio = max(fused_fake_p, 1.0 - fused_fake_p)
        final_confidence = round(conf_ratio * 100.0, 2)
        if disagreement_detected:
            final_confidence = max(50.0, final_confidence - 15.0)

        final_uncertainty = round(1.0 - abs(2.0 * fused_fake_p - 1.0), 4)
        if final_verdict == DecisionState.UNCERTAIN:
            final_uncertainty = max(0.60, final_uncertainty)

        # Confidence level tier
        if final_confidence >= 80.0:
            conf_level = "High"
        elif final_confidence >= 60.0:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        # 8. Database Update
        if analysis_id:
            try:
                from backend.services.database_service import DatabaseService
                DatabaseService.update_fusion_prediction(analysis_id, final_verdict, final_confidence)
            except Exception as e:
                logger.warning(f"FusionAgent: Failed to save final prediction to DB: {e}")

        # 9. Format Agent Timeline Trace
        agents_trace = [
            {"name": primary_name, "summary": primary_res.get("summary", "Specialist analysis completed.")}
        ]
        if metadata_res:
            agents_trace.append({"name": "Metadata Agent", "summary": metadata_res.get("summary", "Metadata inspected.")})
        if artifact_res:
            agents_trace.append({"name": "Artifact Agent", "summary": artifact_res.get("summary", "Forensics analyzed.")})
        agents_trace.append({
            "name": "Fusion Agent",
            "summary": f"Final Verdict: {final_verdict}. {final_reason}"
        })

        user_statement = USER_FACING_STATEMENTS.get(final_verdict, USER_FACING_STATEMENTS[DecisionState.UNCERTAIN])

        return {
            "prediction": final_verdict,
            "verdict": final_verdict,
            "final_verdict": final_verdict,
            "confidence": final_confidence,
            "confidence_score": final_confidence,
            "confidence_level": conf_level,
            "fake_probability": round(fused_fake_p, 4),
            "real_probability": round(1.0 - fused_fake_p, 4),
            "final_probability": round(fused_fake_p, 4),
            "raw_model_score": primary_res.get("raw_model_score", fused_fake_p),
            "uncertainty": final_uncertainty,
            "evidence": evidence_items,
            "reason": final_reason,
            "disagreement_detected": disagreement_detected,
            "explanation": f"{final_verdict}: {user_statement} {final_reason}",
            "user_facing_statement": user_statement,
            "recommendation": primary_res.get("recommendation", "Secondary verification recommended."),
            "analysis_status": primary_res.get("analysis_status", "Completed"),
            "face_detected": primary_res.get("face_detected", True),
            "media_category": primary_res.get("media_category", "Facial Media"),
            "active_agents": [evidence_breakdown[k]["name"] for k in evidence_breakdown.keys()],
            "evidence_breakdown": evidence_breakdown,
            "agents": agents_trace,
        }
