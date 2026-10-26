from typing import Dict, Any, List
from backend.core.decision_system import DecisionState, USER_FACING_STATEMENTS


class ExplanationAgent:
    """
    Synthesizes human-readable explanations based strictly on factual evidence.
    Enforces forensic safety disclaimers and prevents fabricated claims.
    Guarantees agreement with the unified DecisionState: REAL, FAKE, UNCERTAIN, NOT_SUITABLE.
    """

    def explain(
        self,
        verdict: str,
        confidence_level: str,
        media_type: str,
        features: dict,
        evidence: list = None,
        reason: str = None,
        user_facing_statement: str = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes factual forensic explanations aligned with the canonical verdict.
        """
        # Canonical normalization
        if verdict in ("NOT_SUITABLE", getattr(DecisionState, "NOT_SUITABLE", "NOT_SUITABLE")):
            state = DecisionState.NOT_SUITABLE
            user_statement = USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE]
            return {
                "verdict": state,
                "prediction": state,
                "evidence": evidence if evidence else ["Eligibility: Unsupported media (No detectable human face found in media)."],
                "recommendation": "Upload a clear media file containing a visible human face.",
                "user_facing_statement": user_statement,
            }

        from backend.core.decision_system import PresentationDecisionPolicy
        state = PresentationDecisionPolicy.normalize_verdict(verdict)

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
                item = "Temporal: Inter-frame inconsistency and appearance variance detected."
                if item not in evidence_items:
                    evidence_items.append(item)

        # Unified recommendations and statements
        if state == DecisionState.FAKE:
            if not evidence_items:
                evidence_items.append("Neural feature extraction identified deepfake manipulation patterns.")
            recommendation = (
                "Deepfake manipulation flagged. Secondary forensic inspection or source provenance "
                "verification is recommended before publishing or trusting this media."
            )
            user_statement = user_facing_statement or USER_FACING_STATEMENTS[DecisionState.FAKE]
        elif state == DecisionState.REAL:
            if not evidence_items:
                evidence_items.append("No significant high-frequency or boundary anomalies detected in primary inspection.")
            evidence_items.append("Notice: Absence of detected manipulation does not prove absolute authenticity.")
            recommendation = (
                "No strong evidence of manipulation was detected across evaluated detectors. "
                "Note: This does not prove authenticity, as novel synthesis techniques or clean recaptures may leave no detectable trace."
            )
            user_statement = user_facing_statement or USER_FACING_STATEMENTS[DecisionState.REAL]
        else:  # UNCERTAIN
            if reason and reason not in evidence_items:
                evidence_items.insert(0, f"Abstention Reason: {reason}")
            if not evidence_items:
                evidence_items.append("Evidence lies in indeterminate boundary margin or contains conflicting indicators.")
            recommendation = (
                "Insufficient evidence to determine authenticity reliably. "
                "Additional verification, original uncompressed file inspection, or source provenance verification is recommended."
            )
            user_statement = user_facing_statement or USER_FACING_STATEMENTS[DecisionState.UNCERTAIN]

        return {
            "verdict": state,
            "prediction": state,
            "evidence": evidence_items,
            "recommendation": recommendation,
            "user_facing_statement": user_statement,
        }
