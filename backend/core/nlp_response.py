from typing import Dict, Any, Optional
from backend.core.decision_system import DecisionResult, DecisionState


def generate_nlp_response(decision_data: Any, question: Optional[str] = None) -> Dict[str, str]:
    """
    Answers natural-language / conversational inquiries based strictly on the
    empirical SatyaDrishti inference result.
    Never invents conclusions or converts uncertainty into REAL.
    """
    # Normalize input to dictionary or DecisionResult
    if hasattr(decision_data, "to_dict"):
        d = decision_data.to_dict()
    elif isinstance(decision_data, dict):
        d = decision_data
    else:
        d = {}

    prediction = d.get("prediction") or d.get("verdict", DecisionState.UNCERTAIN)
    confidence = d.get("confidence", 0.0)
    fake_prob = d.get("fake_probability", 0.0)
    reason = d.get("reason", "No specific reason provided.")
    evidence = d.get("evidence", [])
    question_clean = (question or "Is this a deepfake?").strip()

    # Build answer grounded strictly in forensic verdict
    if prediction == DecisionState.NOT_SUITABLE:
        answer = (
            "This media is not suitable for facial deepfake analysis because no human face was detected. "
            "SatyaDrishti's image deepfake detector is designed for facial-media authenticity analysis. "
            "Please upload a clear image containing a visible human face."
        )
    elif prediction == DecisionState.UNCERTAIN:
        answer = (
            f"SatyaDrishti cannot determine this reliably from the available evidence. "
            f"The result is UNCERTAIN. Reason: {reason} "
            f"Additional verification or higher-quality source media is recommended."
        )
    elif prediction == DecisionState.FAKE:
        ev_summary = "; ".join(evidence[:2]) if evidence else "Learned synthetic artifacts detected."
        answer = (
            f"Based on SatyaDrishti forensic analysis, this media is classified as FAKE. "
            f"Deepfake detected with {confidence}% confidence (calibrated manipulation probability: {fake_prob:.2f}). "
            f"Key findings: {ev_summary}"
        )
    elif prediction in (DecisionState.LIKELY_REAL, "REAL"):
        answer = (
            f"Based on SatyaDrishti forensic analysis, this media is classified as LIKELY REAL. "
            f"No strong evidence of manipulation was detected across classifier and forensic checks "
            f"(confidence: {confidence}%). "
            f"Important: This assessment does not prove authenticity, as sophisticated or novel manipulations may leave no detectable trace."
        )
    else:
        answer = (
            f"SatyaDrishti inference status is {prediction}. "
            f"Authenticity cannot be confirmed without conclusive forensic evidence."
        )

    return {
        "question": question_clean,
        "answer": answer,
        "prediction": prediction,
        "grounded": True,
        "abstention": (prediction == DecisionState.UNCERTAIN),
    }
