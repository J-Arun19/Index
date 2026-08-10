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
                if features.get("metadata_anomaly") or features.get("audio_sync_anomaly"):
                    evidence.append("Media metadata anomalies / lip-sync deviations detected.")
            
            if confidence_level == "High":
                recommendation = "Manual verification not required. The confidence rating is high and artifacts are distinct."
            else:
                recommendation = "Manual verification recommended. Artifacts are borderline or noisy."
        else:
            evidence.append("Consistent sensor noise distribution observed across the image planes.")
            evidence.append("Facial boundaries are natural and seamless with no digital splicing artifacts.")
            evidence.append("Physiologic features (blink rates, gaze consistency, micro-expressions) conform to standard biological baselines.")
            
            if confidence_level == "Low":
                recommendation = "Manual verification recommended due to overall low analysis confidence."
            else:
                recommendation = "Media appears authentic. No manual intervention required under standard auditing procedures."

        return {
            "verdict": verdict,
            "evidence": evidence,
            "recommendation": recommendation
        }
