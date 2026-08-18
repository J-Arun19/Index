import os
from backend.utils.config import PipelineConfig

class FusionAgent:
    def fuse(self, context_or_agent_name, agent_output=None, analysis_id=None):
        """
        Fuses outputs from the specialized analysis agents.
        Applies a weighted evidence fusion algorithm across active agents.
        Handles both context-driven multi-agent execution and legacy linear execution.
        """
        # Check if we are running in context-driven mode
        if not isinstance(context_or_agent_name, str):
            # Context-driven mode
            context = context_or_agent_name
            media_type = context.media_type
            
            # Determine which primary agent was run
            if media_type.lower() == "image":
                primary_res = context.image_agent_result
                primary_name = "ImageAgent"
            else:
                primary_res = context.video_agent_result
                primary_name = "VideoAgent"
                
            metadata_res = context.metadata_agent_result
            artifact_res = context.artifact_agent_result
            analysis_id = context.analysis_id
        else:
            # Legacy linear execution fallback
            primary_name = context_or_agent_name
            primary_res = agent_output
            metadata_res = None
            artifact_res = None

        # 1. Extract primary agent details
        primary_verdict = primary_res.get("verdict", "REAL") if primary_res else "REAL"
        primary_score = float(primary_res.get("confidence_score", 50))
        p_primary = primary_score / 100.0 if primary_verdict == "DEEPFAKE" else (100.0 - primary_score) / 100.0

        # Define active structures
        weights = {"primary": PipelineConfig.WEIGHT_PRIMARY}
        probabilities = {"primary": p_primary}
        verdicts = {"primary": primary_verdict}
        evidence_breakdown = {
            "primary_specialist": {
                "name": primary_name,
                "verdict": primary_verdict,
                "confidence": primary_score,
                "probability": p_primary
            }
        }

        # 2. Extract Metadata details if available
        if metadata_res:
            metadata_anomalies = metadata_res.get("anomalies_detected", False)
            # Metadata anomalies indicate editing signatures but not automatic deepfake proof
            p_metadata = 0.70 if metadata_anomalies else 0.10
            metadata_verdict = "DEEPFAKE" if metadata_anomalies else "REAL"
            
            weights["metadata"] = PipelineConfig.WEIGHT_METADATA
            probabilities["metadata"] = p_metadata
            verdicts["metadata"] = metadata_verdict
            evidence_breakdown["metadata_specialist"] = {
                "name": "MetadataAgent",
                "verdict": metadata_verdict,
                "anomalies_detected": metadata_anomalies,
                "probability": p_metadata,
                "software_tag": metadata_res.get("software_tag")
            }

        # 3. Extract Artifact details if available
        if artifact_res and artifact_res.get("status") != "Failed":
            artifact_score = float(artifact_res.get("anomaly_score", 0.20))
            artifact_verdict = artifact_res.get("verdict", "REAL")
            
            weights["artifact"] = PipelineConfig.WEIGHT_ARTIFACT
            probabilities["artifact"] = artifact_score
            verdicts["artifact"] = artifact_verdict
            evidence_breakdown["artifact_specialist"] = {
                "name": "ArtifactConsistencyAgent",
                "verdict": artifact_verdict,
                "anomaly_score": artifact_score,
                "probability": artifact_score
            }

        # 4. Normalize weights of active agents
        sum_weights = sum(weights.values())
        normalized_weights = {k: v / sum_weights for k, v in weights.items()}

        # 5. Compute fused probability of FAKE
        final_probability = sum(probabilities[k] * normalized_weights[k] for k in weights.keys())
        final_verdict = "DEEPFAKE" if final_probability > 0.5 else "REAL"

        # 6. Detect disagreements and apply confidence penalty
        disagreement_detected = False
        conflict_level = 0.0
        
        unique_verdicts = set(verdicts.values())
        if len(unique_verdicts) > 1:
            disagreement_detected = True
            conflict_level = round(max(probabilities.values()) - min(probabilities.values()), 2)

        # Base confidence calculation
        base_confidence = final_probability * 100.0 if final_verdict == "DEEPFAKE" else (1.0 - final_probability) * 100.0
        
        if disagreement_detected:
            # Penalty decreases final confidence based on severity of conflict
            penalty = 15.0 * conflict_level
            final_confidence = max(50.0, base_confidence - penalty)
        else:
            final_confidence = base_confidence

        final_confidence = round(final_confidence, 2)
        final_probability = round(final_probability, 4)

        # 7. Database Update
        if analysis_id:
            try:
                from backend.services.database_service import DatabaseService
                DatabaseService.update_fusion_prediction(analysis_id, final_verdict, final_confidence)
            except Exception as e:
                print(f"FusionAgent: Failed to save final prediction: {str(e)}")

        # 8. Format Explanation
        if final_verdict == "DEEPFAKE":
            explanation = f"Fused decision: {final_verdict} (fused confidence {final_confidence}%). "
            if disagreement_detected:
                explanation += "Disagreement was resolved by prioritizing model probability weights."
            else:
                explanation += "All active agents are in alignment."
        else:
            explanation = f"Fused decision: {final_verdict} (fused confidence {final_confidence}%). "
            if disagreement_detected:
                explanation += "Disagreement detected but final probability leans towards authentic."
            else:
                explanation += "All active agents verified the media as authentic."

        # 9. Format Agent trace for the frontend timeline UI
        agents_trace = []
        # Add primary agent trace
        agents_trace.append({
            "name": primary_name,
            "summary": primary_res.get("summary") if primary_res else "Primary analysis completed."
        })
        
        # Add metadata agent trace
        if metadata_res:
            agents_trace.append({
                "name": "Metadata Agent",
                "summary": metadata_res.get("summary")
            })
            
        # Add artifact agent trace
        if artifact_res:
            agents_trace.append({
                "name": "Artifact Agent",
                "summary": artifact_res.get("summary")
            })

        # Add Fusion trace
        agents_trace.append({
            "name": "Fusion Agent",
            "summary": f"Verdict: {final_verdict}. Conflict level: {conflict_level}."
        })

        # Re-map active agents list
        active_agents_list = [evidence_breakdown[k]["name"] for k in evidence_breakdown.keys()]

        # Generate final fusion reason string
        weight_str = ", ".join([f"{k}: {round(v, 2)}" for k, v in normalized_weights.items()])
        fusion_reason = f"Weighted fusion applied ({weight_str}). Final probability of fake: {final_probability}."

        return {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "confidence_score": final_confidence,
            "final_probability": final_probability,
            "final_verdict": final_verdict,
            "active_agents": active_agents_list,
            "disagreement_detected": disagreement_detected,
            "conflict_level": conflict_level,
            "evidence_breakdown": evidence_breakdown,
            "fusion_reason": fusion_reason,
            "explanation": explanation,
            "agents": agents_trace
        }
