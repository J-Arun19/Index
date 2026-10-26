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

    from backend.core.decision_system import PresentationDecisionPolicy
    confidence = d.get("confidence", 0.0)
    fake_prob = d.get("fake_probability")
    if fake_prob is not None:
        try:
            fake_prob = float(fake_prob)
        except (ValueError, TypeError):
            fake_prob = None
    original_verdict = d.get("original_verdict") or d.get("originalVerdict")
    is_binary_fallback = bool(d.get("is_binary_fallback") or d.get("isBinaryFallback"))
    raw_primary = d.get("primary_verdict") or d.get("prediction") or d.get("verdict", DecisionState.UNCERTAIN)
    norm_primary = PresentationDecisionPolicy.normalize_verdict(raw_primary, fake_prob)
    canonical_verdict = PresentationDecisionPolicy.get_canonical_verdict(
        verdict=d.get("verdict"),
        primary_verdict=norm_primary,
        original_verdict=original_verdict,
        is_binary_fallback=is_binary_fallback,
        fake_probability=fake_prob,
    )
    primary_verdict = canonical_verdict
    reliability_warning = d.get("reliability_warning")
    reason = d.get("reason", "No specific reason provided.")
    evidence = d.get("evidence", [])
    question_clean = (question or "Is this a deepfake?").strip()

    fake_prob_pct = f"{fake_prob * 100:.2f}%" if fake_prob is not None else "N/A"
    conf_str = f" (confidence: {confidence}%)" if (confidence is not None and not is_binary_fallback) else ""

    # Build answer grounded strictly in forensic verdict: REAL, FAKE, NOT_SUITABLE, UNCERTAIN
    if primary_verdict == DecisionState.NOT_SUITABLE:
        answer = (
            "This media is not suitable for facial deepfake analysis because no human face was detected. "
            "SatyaDrishti's image deepfake detector is designed for facial-media authenticity analysis. "
            "Please upload a clear image containing a visible human face."
        )
    elif primary_verdict in (DecisionState.FAKE, "FAKE", "DEEPFAKE"):
        answer = (
            f"Based on SatyaDrishti forensic analysis, this media is classified as FAKE{conf_str}. "
            f"Deepfake detected / high likelihood of manipulation based on forensic anomaly patterns. "
            f"Estimated fake probability: {fake_prob_pct}. "
            f"Reason: {reason}"
        )
    elif primary_verdict in (DecisionState.REAL, "REAL"):
        answer = (
            f"Based on SatyaDrishti forensic analysis, this media is classified as REAL{conf_str}. "
            f"No strong evidence of manipulation was detected across classifier and forensic checks. "
            f"Notice: Absence of detected manipulation does not guarantee absolute authenticity. This does not prove authenticity and does not prove absolute authenticity."
        )
    else:  # UNCERTAIN
        if is_binary_fallback:
            answer = (
                f"SatyaDrishti has classified this media as UNCERTAIN (uncertified). "
                f"Specialist detectors produced ambiguous or conflicting indicators. "
                f"Provisional directional indicator: tends toward {raw_primary} (estimated fake probability: {fake_prob_pct}). "
                f"Notice: {reliability_warning or 'Authenticity has NOT been established; secondary manual verification is advised.'} "
                f"Reason: {reason}"
            )
        else:
            answer = (
                f"SatyaDrishti cannot determine this reliably from the available evidence. "
                f"The result is UNCERTAIN. Reason: {reason} "
                f"Additional verification or higher-quality source media is recommended."
            )


    return {
        "question": question_clean,
        "answer": answer,
        "prediction": canonical_verdict,
        "verdict": canonical_verdict,
        "canonical_verdict": canonical_verdict,
        "primary_verdict": norm_primary,
        "original_verdict": original_verdict,
        "is_binary_fallback": is_binary_fallback,
        "reliability_warning": reliability_warning,
        "grounded": True,
        "abstention": (canonical_verdict == DecisionState.UNCERTAIN),
    }

