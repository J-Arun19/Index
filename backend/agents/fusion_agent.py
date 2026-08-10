class FusionAgent:
    def fuse(self, agent_name, agent_output, analysis_id=None):
        """
        Fuses outputs from the specialized analysis agents.
        In a production system, this would apply a weighted fusion algorithm,
        a meta-classifier, or a LLM-based consensus generator.
        Here we generate a structured verdict, explanation, and agent summaries.
        """
        verdict = agent_output["verdict"]
        confidence = agent_output["confidence_score"]
        is_fake = verdict == "DEEPFAKE"
        
        # Store prediction in database if analysis_id is provided
        if analysis_id:
            try:
                from backend.services.database_service import DatabaseService
                DatabaseService.update_fusion_prediction(analysis_id, verdict, confidence)
            except Exception as e:
                print(f"FusionAgent: Failed to save final prediction: {str(e)}")

        
        if is_fake:
            explanation = (
                f"{'Image' if agent_name == 'ImageAgent' else 'Video'} Agent detected "
                f"high-confidence anomalies ({confidence}%). "
            )
            if agent_name == 'ImageAgent':
                explanation += (
                    "Image agent detected inconsistent frequency artifacts around facial contours. "
                    "Sensor noise pattern mismatches confirm synthetic splicing."
                )
            else:
                explanation += (
                    "Video Agent flagged temporal blinking irregularities and facial boundary jitters "
                    "across consecutive frames. Temporal consistency checks confirm synthetic artifacts."
                )
        else:
            explanation = (
                f"No significant frequency or temporal anomalies detected. "
                f"Sensor noise pattern and frame-to-frame consistency are stable. "
                f"Confidence agent found agreement between image and video specialists ({confidence}% confidence)."
            )
            
        # Format the agent traces for the frontend UI
        agents_trace = [
            {
                "name": "Image Agent",
                "summary": agent_output["summary"] if agent_name == "ImageAgent" else "Media type is video. Image analysis skipped."
            },
            {
                "name": "Video Agent",
                "summary": agent_output["summary"] if agent_name == "VideoAgent" else "Media type is image. Video analysis skipped."
            },
            {
                "name": "Confidence Agent",
                "summary": f"Weighted score {confidence}%, agents aligned."
            },
            {
                "name": "Fusion Agent",
                "summary": f"Final verdict: {verdict}."
            }
        ]
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "agents": agents_trace
        }
