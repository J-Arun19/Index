class ConfidenceEvaluationAgent:
    def evaluate(self, agent_output: dict) -> dict:
        """
        Evaluates the confidence score of the analysis output and assigns a tier and recommendation.
        """
        verdict = agent_output.get("verdict") or agent_output.get("prediction")
        if verdict == "NOT_SUITABLE":
            return {
                "confidence": None,
                "level": "Not Performed",
                "recommendation": "Upload a clear image containing a visible human face."
            }

        confidence = agent_output.get("confidence_score", 0)
        if confidence is None:
            return {
                "confidence": None,
                "level": "Not Performed",
                "recommendation": "Upload a clear image containing a visible human face."
            }
        
        if confidence >= 80:
            level = "High"
            recommendation = "Prediction considered reliable. Facial landmarks and noise distributions match the classification with high certainty."
        elif confidence >= 50:
            level = "Medium"
            recommendation = "Prediction moderately confident. Some edge discrepancies present. Secondary verification recommended."
        else:
            level = "Low"
            recommendation = "Low confidence. Recommend manual verification of facial features and compression artifacts."

        return {
            "confidence": confidence,
            "level": level,
            "recommendation": recommendation
        }
