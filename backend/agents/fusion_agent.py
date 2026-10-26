import os
import logging
import numpy as np
from typing import Dict, Any, Optional
from backend.utils.config import PipelineConfig
from backend.core.decision_system import SafetyDecisionSystem, DecisionState, USER_FACING_STATEMENTS, PresentationDecisionPolicy

logger = logging.getLogger(__name__)


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
        if primary_pred in ("DEEPFAKE", "FAKE"):
            primary_pred = DecisionState.FAKE
        elif primary_pred in (DecisionState.REAL, "REAL"):
            primary_pred = DecisionState.REAL
        elif primary_pred in (DecisionState.LIKELY_REAL, "LIKELY_REAL", "LIKELY REAL"):
            primary_pred = DecisionState.LIKELY_REAL

        # Media Suitability bypass: If media is unsupported, do not perform authenticity fusion
        if primary_pred in (DecisionState.NOT_SUITABLE, "NOT_SUITABLE"):
            final_verdict = DecisionState.NOT_SUITABLE
            primary_verdict, original_verdict, is_binary_fallback, reliability_warning = (
                PresentationDecisionPolicy.apply_presentation_policy(
                    verdict=final_verdict,
                    fake_probability=None,
                    disagreement_detected=False,
                    quality_issues=False,
                )
            )
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
                    DatabaseService.update_fusion_prediction(analysis_id, primary_verdict, 0.0)
                except Exception as e:
                    logger.warning(f"FusionAgent: Failed to save final prediction to DB: {e}")

            agents_trace = [
                {"name": primary_name, "summary": "Media suitability evaluation: Unsupported (no face detected)."},
                {"name": "Fusion Agent", "summary": "Media is not suitable for facial analysis. Analysis not performed."}
            ]

            return {
                "prediction": primary_verdict,
                "verdict": primary_verdict,
                "final_verdict": final_verdict,
                "primary_verdict": primary_verdict,
                "original_verdict": original_verdict,
                "is_binary_fallback": is_binary_fallback,
                "reliability_warning": reliability_warning,
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

        primary_conf = float(primary_res.get("confidence", primary_res.get("confidence_score", 50.0)) or 50.0)

        def coerce_probability(value):
            if value is None:
                return None
            try:
                p = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(p):
                return None
            if 0.0 <= p <= 1.0:
                return p
            if 0.0 <= p <= 100.0:
                return p / 100.0
            return None

        primary_fake_p_val = primary_res.get("fake_probability")
        if primary_fake_p_val is None:
            primary_fake_p_val = primary_res.get("fakeProbability")
        if primary_fake_p_val is None:
            primary_fake_p_val = primary_res.get("raw_model_score")
        if primary_fake_p_val is None and primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE"):
            primary_fake_p_val = primary_res.get("confidence_score")
        if primary_fake_p_val is None and primary_pred in (DecisionState.LIKELY_REAL, "REAL"):
            primary_fake_p_val = 1.0 - (primary_res.get("confidence") or primary_res.get("confidence_score") or 0.0) / 100.0

        primary_fake_p = coerce_probability(primary_fake_p_val)
        if primary_fake_p is None and primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE"):
            primary_fake_p = 0.50
        if primary_fake_p is None and primary_pred in (DecisionState.LIKELY_REAL, "REAL"):
            primary_fake_p = 0.50

        evidence_breakdown = {
            "primary_specialist": {
                "name": primary_name,
                "verdict": primary_pred,
                "confidence": primary_conf,
                "fake_probability": primary_fake_p,
            }
        }

        evidence_items = list(primary_res.get("evidence", []))

        # 2. Metadata signals (Corroborating qualitative evidence; does NOT distort neural probability)
        metadata_anomalies = False
        if metadata_res and metadata_res.get("status") != "Failed":
            metadata_anomalies = bool(metadata_res.get("anomalies_detected", False))
            if metadata_anomalies:
                evidence_breakdown["metadata_specialist"] = {
                    "name": "MetadataAgent",
                    "anomalies_detected": True,
                    "software_tag": metadata_res.get("software_tag"),
                }
                evidence_items.append(
                    f"Metadata: Editing signature detected ({metadata_res.get('software_tag') or 'Edited'})."
                )
            elif str(media_type).lower() == "image":
                evidence_breakdown["metadata_specialist"] = {
                    "name": "MetadataAgent",
                    "anomalies_detected": False,
                    "software_tag": metadata_res.get("software_tag"),
                }

        # 3. Artifact & Genuine Forensic signals (Corroborating qualitative evidence)
        artifact_anomalies = False
        if artifact_res and artifact_res.get("status") not in ("Failed", "Skipped") and artifact_res.get("anomaly_score") is not None:
            artifact_score = float(artifact_res.get("anomaly_score", 0.15))
            artifact_anomalies = artifact_score > 0.45
            evidence_breakdown["artifact_specialist"] = {
                "name": "ArtifactConsistencyAgent",
                "anomaly_score": artifact_score,
            }
            for art in artifact_res.get("detected_artifacts", []):
                if art not in evidence_items:
                    evidence_items.append(art)

        # 4. Multi-Provider AI Vision Signals
        visual_analysis = primary_res.get("visual_analysis", {})
        if visual_analysis:
            evidence_breakdown["multi_provider_vision"] = {
                "name": "MultiProviderVisionForensics",
                "consensus_assessment": visual_analysis.get("consensus_assessment", "INCONCLUSIVE"),
                "providers_attempted": visual_analysis.get("providers_attempted", []),
                "providers_succeeded": visual_analysis.get("providers_succeeded", []),
                "providers_failed": visual_analysis.get("providers_failed", []),
                "provider_results": visual_analysis.get("provider_results", {}),
                "observable_anomalies": visual_analysis.get("observable_anomalies", []),
                "approx_cost_usd": visual_analysis.get("approx_cost_usd", 0.0),
                "external_ai_configured": visual_analysis.get("external_ai_configured", False),
            }

        # Multi-agent weighted evidence normalization across active agents
        weights = {"primary": PipelineConfig.WEIGHT_PRIMARY}
        probabilities = {"primary": primary_fake_p if primary_fake_p is not None else 0.50}

        if metadata_res and metadata_res.get("status") != "Failed":
            weights["metadata"] = PipelineConfig.WEIGHT_METADATA
            probabilities["metadata"] = 0.70 if metadata_anomalies else 0.10

        if artifact_res and artifact_res.get("status") not in ("Failed", "Skipped") and artifact_res.get("anomaly_score") is not None:
            weights["artifact"] = PipelineConfig.WEIGHT_ARTIFACT
            probabilities["artifact"] = float(artifact_res.get("anomaly_score", 0.15))

        sum_w = sum(weights.values()) if weights else 1.0
        norm_w = {k: v / sum_w for k, v in weights.items()}
        multi_agent_p = round(float(sum(probabilities[k] * norm_w[k] for k in weights.keys())), 4)

        # Compute fused fake probability strictly anchored to neural specialist inference
        if primary_fake_p is None:
            fused_fake_p = None
        elif visual_analysis.get("external_ai_configured") and visual_analysis.get("providers_succeeded") and visual_analysis.get("visual_synthetic_score") is not None:
            # When external AI vision successfully executes, blend with calibrated neural probability
            v_score = float(visual_analysis.get("visual_synthetic_score"))
            fused_fake_p = float(np.clip(0.65 * primary_fake_p + 0.35 * v_score, 0.0, 1.0))
        else:
            # True calibrated neural probability from the primary specialist agent
            fused_fake_p = float(np.clip(primary_fake_p, 0.0, 1.0))


        # 5. Evaluate Disagreements & Conflicts (CRITICAL SAFETY LAYER)
        disagreement_detected = False
        conflict_reason = ""

        is_primary_fake = primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE")
        is_primary_real = primary_pred in (DecisionState.LIKELY_REAL, "REAL", "NO MANIPULATION DETECTED")

        # Disagreement occurs when classifier indicates authentic features but independent forensic analyzers flag tampering,
        # or when classifier indicates synthetic manipulation but independent checks confirm authentic natural media.
        primary_conflict = (
            primary_res.get("disagreement_detected", False)
            or ("contradictory" in str(primary_res.get("reason", "")).lower())
            or ("disagreement" in str(primary_res.get("reason", "")).lower())
            or ("evidence conflict" in str(primary_res.get("reason", "")).lower())
            or (is_primary_real and "tampering" in str(primary_res.get("reason", "")).lower())
        )

        has_vision_tampering = (
            visual_analysis.get("consensus_assessment") == "FAKE"
            or (visual_analysis.get("visual_synthetic_score", 0.0) > 0.65)
            or (len(visual_analysis.get("observable_anomalies", [])) > 0 and visual_analysis.get("visual_synthetic_score", 0.0) >= 0.50)
        )
        artifact_verdict = artifact_res.get("verdict") if artifact_res else None
        specialist_disagreement = (
            (is_primary_real and (artifact_anomalies or metadata_anomalies or has_vision_tampering or artifact_verdict in (DecisionState.FAKE, "DEEPFAKE", "FAKE")))
            or (is_primary_fake and artifact_verdict in (DecisionState.LIKELY_REAL, "REAL") and not has_vision_tampering)
        )

        if specialist_disagreement or primary_conflict:
            disagreement_detected = True
            conflict_reason = (
                primary_res.get("reason") if primary_conflict else
                "Contradictory evidence: Primary classifier indicated authentic features, "
                "but independent forensic/vision/metadata analysis detected manipulation signatures."
                if is_primary_real else
                "Contradictory evidence: Primary classifier flagged synthetic manipulation, "
                "but independent artifact consistency agent found no manipulation artifacts."
            )
        elif primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE") and not artifact_anomalies and not metadata_anomalies and primary_fake_p < 0.65 and not has_vision_tampering:
            disagreement_detected = True
            conflict_reason = "Weak synthetic signals without corroboration from independent forensic analyzers."

        # 6. Determine Three-State Final Verdict
        tau_real = self.decision_system.calibrator.tau_real
        tau_fake = self.decision_system.calibrator.tau_fake
        abstention_code = None

        # Rule 1: quality or primary abstention is respected only when the underlying evidence remains genuinely ambiguous
        if primary_pred == DecisionState.UNCERTAIN:
            if primary_fake_p is not None and not disagreement_detected and primary_fake_p >= 0.65:
                final_verdict = DecisionState.FAKE
                final_reason = f"Primary specialist evidence is ambiguous but still points to synthetic manipulation (fake probability: {primary_fake_p:.2f})."
            elif primary_fake_p is not None and not disagreement_detected and primary_fake_p <= 0.35:
                final_verdict = DecisionState.REAL
                final_reason = f"Primary specialist evidence is ambiguous but still points to authentic natural media (fake probability: {primary_fake_p:.2f})."
            else:
                final_verdict = DecisionState.UNCERTAIN
                final_reason = primary_res.get("reason", "Primary specialist abstained due to inconclusive evidence.")
                abstention_code = primary_res.get("abstention_code") or primary_res.get("abstentionCode") or "INSUFFICIENT_EVIDENCE"
        # Rule 2: Escalation to UNCERTAIN if contradictory evidence detected
        elif disagreement_detected:
            final_verdict = DecisionState.UNCERTAIN
            final_reason = f"Abstained due to evidence conflict: {conflict_reason}"
            abstention_code = "MODEL_DISAGREEMENT"
        # Rule 3: Boundary zone resolution / abstention
        elif tau_real <= fused_fake_p <= tau_fake:
            if primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE") and fused_fake_p >= PipelineConfig.CLASSIFICATION_THRESHOLD and not disagreement_detected and primary_conf >= 80.0:
                final_verdict = DecisionState.FAKE
                final_reason = f"Multi-agent consensus confirms synthetic manipulation (fused probability: {fused_fake_p:.2f})."
            elif primary_pred in (DecisionState.LIKELY_REAL, DecisionState.REAL, "REAL") and not disagreement_detected and primary_conf >= 75.0:
                final_verdict = DecisionState.REAL
                final_reason = primary_res.get("reason", "Multi-agent consensus confirms authentic natural features.")
            else:
                final_verdict = DecisionState.UNCERTAIN
                final_reason = (
                    f"Fused probability ({fused_fake_p:.2f}) lies in the ambiguous decision margin [{tau_real:.2f} - {tau_fake:.2f}]. "
                    "Evidence is inconclusive."
                )
                abstention_code = "INSUFFICIENT_EVIDENCE"
        # Rule 4: Decisive FAKE
        elif fused_fake_p > tau_fake or primary_pred in (DecisionState.FAKE, "DEEPFAKE", "FAKE") or (visual_analysis.get("consensus_assessment") == "FAKE" and has_vision_tampering):
            final_verdict = DecisionState.FAKE
            final_reason = f"Multi-agent consensus confirms synthetic manipulation (fused probability: {fused_fake_p:.2f})."
        # Rule 5: Explicitly certified REAL vs uncertified LIKELY_REAL
        elif primary_pred in (DecisionState.REAL, "REAL") and not disagreement_detected:
            final_verdict = DecisionState.REAL
            final_reason = primary_res.get("reason", "Multi-agent consensus explicitly confirms authentic natural features.")
        else:
            final_verdict = DecisionState.REAL
            final_reason = (
                "No strong evidence of manipulation was detected across classifier and forensic checks."
            )

        # 7. Confidence & Uncertainty Computation
        if fused_fake_p is not None:
            conf_ratio = max(fused_fake_p, 1.0 - fused_fake_p)
            final_confidence = round(conf_ratio * 100.0, 2)
            if disagreement_detected:
                final_confidence = max(50.0, final_confidence - 15.0)

            final_uncertainty = round(1.0 - abs(2.0 * fused_fake_p - 1.0), 4)
            if final_verdict == DecisionState.UNCERTAIN:
                final_uncertainty = max(0.60, final_uncertainty)
        else:
            final_confidence = None
            final_uncertainty = 1.0

        # Apply Presentation Fallback Policy
        quality_issues = bool(
            primary_res.get("quality_issues", False) or 
            primary_res.get("face_quality_score", 1.0) < 0.5
        )
        primary_verdict, original_verdict, is_binary_fallback, reliability_warning = (
            PresentationDecisionPolicy.apply_presentation_policy(
                verdict=final_verdict,
                fake_probability=fused_fake_p,
                has_conflict=disagreement_detected,
                disagreement_detected=disagreement_detected,
                quality_issues=quality_issues,
                reason=final_reason,
            )
        )

        # Confidence level tier: provisional, suspicious, or inconclusive results must NEVER receive High Confidence
        if is_binary_fallback or primary_verdict in ("SUSPICIOUS - provisional", "INCONCLUSIVE", "UNCERTAIN") or final_verdict == DecisionState.UNCERTAIN:
            conf_level = "Provisional" if is_binary_fallback else "Uncertified"
        elif final_confidence >= 80.0:
            conf_level = "High"
        elif final_confidence >= 60.0:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        # 8. Database Update
        if analysis_id:
            try:
                from backend.services.database_service import DatabaseService
                db_conf = float(final_confidence) if final_confidence is not None else 0.0
                DatabaseService.update_fusion_prediction(analysis_id, primary_verdict, db_conf)
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
        
        fusion_summary = f"Final Verdict: {primary_verdict}. {final_reason}"
        if is_binary_fallback:
            fusion_summary += " (Provisional prediction; treat cautiously)."
        agents_trace.append({
            "name": "Fusion Agent",
            "summary": fusion_summary
        })

        # Prefer evidence-driven user statement from primary specialist if available
        primary_statement = primary_res.get("user_facing_statement") or primary_res.get("summary")
        if primary_statement and any(k in primary_statement for k in ("At ", "Across ", "Facial region", "measured", "observed")):
            user_statement = primary_statement
        else:
            user_statement = USER_FACING_STATEMENTS.get(
                primary_verdict, USER_FACING_STATEMENTS.get(DecisionState.UNCERTAIN)
            )

        canonical_state = (
            DecisionState.NOT_SUITABLE if final_verdict in (DecisionState.NOT_SUITABLE, "NOT_SUITABLE") else
            DecisionState.UNCERTAIN if (is_binary_fallback or final_verdict == DecisionState.UNCERTAIN or original_verdict == DecisionState.UNCERTAIN) else
            DecisionState.FAKE if primary_verdict in (DecisionState.FAKE, DecisionState.DEEPFAKE, "DEEPFAKE", "FAKE") else
            DecisionState.REAL if primary_verdict in (DecisionState.REAL, "REAL") else
            DecisionState.UNCERTAIN
        )

        conf_out = None if (is_binary_fallback or canonical_state in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE)) else final_confidence
        conf_level_out = (
            "Not Performed" if canonical_state == DecisionState.NOT_SUITABLE else
            ("Provisional" if is_binary_fallback else ("Uncertified" if canonical_state == DecisionState.UNCERTAIN else conf_level))
        )

        return {
            "prediction": canonical_state,
            "verdict": canonical_state,
            "final_verdict": canonical_state,
            "canonical_verdict": canonical_state,
            "canonicalVerdict": canonical_state,
            "primary_verdict": primary_verdict,
            "primaryVerdict": primary_verdict,
            "provisional_suggestion": primary_verdict if is_binary_fallback else None,
            "provisionalSuggestion": primary_verdict if is_binary_fallback else None,
            "abstention_code": abstention_code if canonical_state in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE) else None,
            "abstentionCode": abstention_code if canonical_state in (DecisionState.UNCERTAIN, DecisionState.NOT_SUITABLE) else None,
            "original_verdict": original_verdict,
            "originalVerdict": original_verdict,
            "is_binary_fallback": is_binary_fallback,
            "isBinaryFallback": is_binary_fallback,
            "is_estimate": is_binary_fallback or (canonical_state == DecisionState.UNCERTAIN),
            "isEstimate": is_binary_fallback or (canonical_state == DecisionState.UNCERTAIN),
            "reliability_warning": reliability_warning,
            "reliabilityWarning": reliability_warning,
            "confidence": final_confidence,
            "confidence_score": final_confidence,
            "certified_confidence": conf_out,
            "confidence_level": conf_level_out,
            "fake_probability": round(fused_fake_p, 4) if fused_fake_p is not None else None,
            "real_probability": round(1.0 - fused_fake_p, 4) if fused_fake_p is not None else None,
            "final_probability": multi_agent_p if (metadata_res or artifact_res) else (round(fused_fake_p, 4) if fused_fake_p is not None else None),
            "raw_model_score": primary_res.get("raw_model_score", fused_fake_p),
            "uncertainty": final_uncertainty,
            "evidence": evidence_items,
            "influencing_evidence": primary_res.get("influencing_evidence", []),
            "reason": final_reason,
            "disagreement_detected": disagreement_detected,
            "conflict_level": 0.50 if disagreement_detected else 0.0,
            "explanation": f"{primary_verdict}: {user_statement} {final_reason}",
            "user_facing_statement": user_statement,
            "summary": user_statement,
            "recommendation": primary_res.get("recommendation", "Secondary verification recommended."),
            "analysis_status": primary_res.get("analysis_status", "Completed"),
            "face_detected": primary_res.get("face_detected", True),
            "face_bbox": primary_res.get("face_bbox"),
            "evidence_snapshots": primary_res.get("evidence_snapshots", []),
            "evidence_cards": primary_res.get("evidence_cards", []),
            "forensic_evidence": primary_res.get("forensic_evidence", {}),
            "quality_gate": primary_res.get("quality_gate", {}),
            "features": primary_res.get("features", {}),
            "media_category": primary_res.get("media_category", "Facial Media"),
            "visual_analysis": primary_res.get("visual_analysis", {}),
            "models": primary_res.get("models", []),
            "measurements": primary_res.get("measurements", {}),
            "supporting_evidence": primary_res.get("supporting_evidence", []),
            "contradicting_evidence": primary_res.get("contradicting_evidence", []),
            "unavailable_checks": primary_res.get("unavailable_checks", []),
            "decision": primary_res.get("decision", {
                "method": "Multi-agent forensic evidence fusion",
                "score": round(fused_fake_p, 4) if fused_fake_p is not None else None,
                "score_type": "calibrated_synthetic_probability",
                "confidence": conf_out,
                "confidence_type": "Certified" if conf_out is not None else "Uncertified",
                "criteria_met": bool(conf_out is not None and canonical_state in (DecisionState.REAL, DecisionState.FAKE)),
                "reason": final_reason,
            }),
            "limitations": primary_res.get("limitations", []),
            "is_provisional": is_binary_fallback,
            "isProvisional": is_binary_fallback,
            "media_type": primary_res.get("media_type", "image"),
            "task_type": primary_res.get("task_type", "multimodal_facial_deepfake_detection"),
            "audio_visual_sync": primary_res.get("audio_visual_sync", {
                "analyzed": False,
                "reason": "Lip-sync analysis not performed (audio-visual temporal model not configured)."
            }),
            "active_agents": [evidence_breakdown[k]["name"] for k in evidence_breakdown.keys()],
            "evidence_breakdown": evidence_breakdown,
            "approx_cost_usd": visual_analysis.get("approx_cost_usd", 0.0),
            "providers_ran": visual_analysis.get("providers_succeeded", visual_analysis.get("providers_attempted", [])),
            "agents": agents_trace,
        }

