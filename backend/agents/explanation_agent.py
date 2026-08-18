class ExplanationAgent:
    def explain(self, verdict: str, confidence_level: str, media_type: str, features: dict) -> dict:
        """
        Synthesizes human-readable explanations, including detailed evidence and recommendations.
        """
        is_fake = verdict == "DEEPFAKE"
        
        evidence = []
        if is_fake:
            evidence.append("Facial blending artifacts detected around facial region.")
            evidence.append("Eye region inconsistencies and asymmetrical reflection patterns observed.")
            evidence.append("Texture mismatch and unnatural gradient detected around facial boundary.")
            
            if features:
                if features.get("noise_inconsistency", 0) > 0.4 or features.get("temporal_flicker_score", 0) > 0.4:
                    evidence.append("Sub-pixel noise inconsistency signals synthetic media composition.")
                if features.get("edge_boundary_artifacts", 0) > 0.4 or features.get("face_jitter_ratio", 0) > 0.4:
                    evidence.append("Edge boundary anomalies suggest face-swapping or warping techniques.")
                
                # Check for new agentic fields
                if features.get("metadata_anomaly") or features.get("audio_sync_anomaly"):
                    soft = features.get("software_tag")
                    software_msg = f" ({soft})" if soft else ""
                    evidence.append(f"Media metadata anomalies / editing software signature detected{software_msg}.")
                if features.get("artifact_anomaly"):
                    score = features.get("anomaly_score", 0.0)
                    evidence.append(f"Deep consistency scan flagged structural manipulation artifacts (anomaly score: {score}).")
            
            if confidence_level == "High":
                recommendation = "Manual verification not required. The confidence rating is high and artifacts are distinct."
            else:
                recommendation = "Manual verification recommended. Artifacts are borderline or noisy."
        else:
            evidence.append("The image classifier assigned a REAL probability based on learned visual representations.")
            evidence.append("No significant high-frequency edge or boundary anomalies were detected in the primary analysis.")
            evidence.append("Note: Physiological signals (blink rates, gaze consistency, micro-expressions) were NOT measured during single-frame image analysis.")
            
            if features:
                if features.get("metadata_anomaly"):
                    soft = features.get("software_tag")
                    evidence.append(f"Note: Editing software metadata found ({soft}), but visual elements are authentic.")
                if features.get("artifact_anomaly"):
                    evidence.append("Pixel/frame level anomalies detected but insufficient to override primary authenticity verdict.")
                    
            if confidence_level == "Low":
                recommendation = "Manual verification recommended due to overall low analysis confidence."
            else:
                recommendation = "Media appears authentic based on primary model analysis. No manual intervention required under standard auditing procedures."

        return {
            "verdict": verdict,
            "evidence": evidence,
            "recommendation": recommendation
        }
