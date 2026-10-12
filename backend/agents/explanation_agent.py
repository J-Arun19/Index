from typing import Dict, Any, List
from backend.core.decision_system import DecisionState, USER_FACING_STATEMENTS


class ExplanationAgent:
    """
    Synthesizes human-readable explanations based strictly on factual evidence.
    Enforces forensic safety disclaimers and prevents fabricated claims.
    """

    def explain(
        self,
        verdict: str,
        confidence_level: str,
        media_type: str,
        features: dict,
        evidence: list = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes factual forensic explanations.
        """
        # Check for media suitability
        if verdict in ("NOT_SUITABLE", getattr(DecisionState, "NOT_SUITABLE", "NOT_SUITABLE")):
            user_statement = USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE]
            return {
                "verdict": DecisionState.NOT_SUITABLE,
                "prediction": DecisionState.NOT_SUITABLE,
                "evidence": evidence if evidence else ["Eligibility: Unsupported media (No detectable human face found in image)."],
                "recommendation": "Upload a clear image containing a visible human face.",
                "user_facing_statement": user_statement,
            }

        # Normalize verdict
        if verdict in ("DEEPFAKE", "FAKE"):
            state = DecisionState.FAKE
        elif verdict in ("LIKELY_REAL", "REAL"):
            state = DecisionState.LIKELY_REAL
        else:
            state = DecisionState.UNCERTAIN

        evidence_items: List[str] = list(evidence) if evidence else []

        # Genuine feature checks
        if features:
            if features.get("fft_spectral_anomaly", 0.0) > 0.45:
                item = "Frequency domain: Anomalous high-frequency spectral roll-off detected."
                if item not in evidence_items:
                    evidence_items.append(item)
            if features.get("ela_compression_anomaly", 0.0) > 0.45:
                item = "Compression ELA: Localized compression variance indicates potential splicing."
                if item not in evidence_items:
                    evidence_items.append(item)
            if features.get("edge_boundary_artifacts", 0.0) > 0.45:
                item = "Edge Analysis: High-frequency gradient discontinuity around facial perimeter."
                if item not in evidence_items:
                    evidence_items.append(item)
            if features.get("metadata_anomaly"):
                soft = features.get("software_tag")
                item = f"Metadata: Editing signature detected{f' ({soft})' if soft else ''}."
                if item not in evidence_items:
                    evidence_items.append(item)
            if features.get("temporal_flicker_score", 0.0) > 0.22:
                item = "Temporal: Inter-frame inconsistency and motion flicker detected."
                if item not in evidence_items:
                    evidence_items.append(item)

        # Standardized recommendation and statements
        if state == DecisionState.FAKE:
            if not evidence_items:
                evidence_items.append("Neural feature extraction identified deepfake manipulation patterns.")
            recommendation = (
                "Deepfake manipulation flagged. Secondary forensic inspection or source provenance "
                "verification is recommended before publishing or trusting this media."
            )
        elif state == DecisionState.LIKELY_REAL:
            if not evidence_items:
                evidence_items.append("No significant high-frequency or boundary anomalies detected in primary inspection.")
            evidence_items.append("Notice: Physiological signals and deep temporal flows were not measured.")
            recommendation = (
                "No strong evidence of manipulation was detected. Note: This assessment does not prove "
                "absolute authenticity, as novel synthesis techniques may bypass standard detectors."
            )
        else:  # UNCERTAIN
            if reason and reason not in evidence_items:
                evidence_items.insert(0, f"Abstention Reason: {reason}")
            if not evidence_items:
                evidence_items.append("Evidence is ambiguous, noisy, or conflicting.")
            recommendation = (
                "Insufficient evidence to determine authenticity reliably. Additional verification, "
                "original uncompressed file inspection, or multi-frame video analysis is strongly recommended."
            )

        user_statement = USER_FACING_STATEMENTS.get(state, USER_FACING_STATEMENTS[DecisionState.UNCERTAIN])

        return {
            "verdict": state,
            "prediction": state,
            "evidence": evidence_items,
            "recommendation": recommendation,
            "user_facing_statement": user_statement,
        }
