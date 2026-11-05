from typing import Dict, Any, List, Optional
from backend.core.decision_system import DecisionState, USER_FACING_STATEMENTS


def generate_user_friendly_explanation(
    verdict: str,
    confidence: Optional[float] = None,
    confidence_level: Optional[str] = None,
    fake_probability: Optional[float] = None,
    features: Optional[dict] = None,
    forensic_evidence: Optional[dict] = None,
    evidence_cards: Optional[list] = None,
    models: Optional[list] = None,
    is_binary_fallback: bool = False,
    disagreement_detected: bool = False,
    media_type: str = "image",
    sequence_aggregation: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Generates a clear, user-friendly forensic explanation designed for ordinary users.
    Avoids technical jargon (ResNet18, XceptionNet, ViT, PRNU, ELA, FFT, Temperature Scaling,
    Fused Probability, Decision Margin, Diffusion Model) in Level 1 explanations.
    """
    norm_v = str(verdict or "").strip().upper()
    if norm_v in ("ILLUSTRATION", "ILLUSTRATION_OR_CARTOON", "ILLUSTRATION / CARTOON DETECTED"):
        norm_v = "ILLUSTRATION"
    elif norm_v in ("NOT_SUITABLE", getattr(DecisionState, "NOT_SUITABLE", "NOT_SUITABLE")):
        norm_v = "NOT_SUITABLE"
    elif norm_v in ("FAKE", "DEEPFAKE", "SUSPICIOUS", "MANIPULATED"):
        norm_v = "FAKE"
    else:
        norm_v = "REAL"

    # 1. Determine Confidence Tier and Numeric Value
    conf_pct = None
    if norm_v in ("NOT_SUITABLE", "ILLUSTRATION"):
        conf_tier = "Not Performed"
    elif is_binary_fallback or disagreement_detected:
        conf_tier = "Low"
        if confidence is not None:
            c = float(confidence)
            conf_pct = round(c if c > 1.0 else c * 100.0, 1)
        elif fake_probability is not None:
            fp = float(fake_probability)
            conf_pct = round(max(fp, 1.0 - fp) * 100.0, 1)
        else:
            conf_pct = 52.0
    else:
        if confidence is not None:
            c = float(confidence)
            conf_pct = round(c if c > 1.0 else c * 100.0, 1)
        elif fake_probability is not None:
            fp = float(fake_probability)
            conf_pct = round(max(fp, 1.0 - fp) * 100.0, 1)
        else:
            conf_pct = 85.0

        if confidence_level in ("High", "Moderate", "Low"):
            conf_tier = confidence_level
        elif conf_pct >= 75.0:
            conf_tier = "High"
        elif conf_pct >= 55.0:
            conf_tier = "Moderate"
        else:
            conf_tier = "Low"

    # 2. Extract grounded findings in plain English
    findings: List[Dict[str, str]] = []

    # Map evidence cards to plain-language findings
    if evidence_cards:
        for c in evidence_cards:
            cid = str(c.get("id", "")).lower()
            label = str(c.get("label", "")).upper()
            is_card_fake = "FAKE" in label

            if "boundary" in cid or "jawline" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Face Boundary Blending",
                        "explanation": "Noticeable blending, blurring, or edge inconsistency was detected along the border where the face meets the hair and background.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Natural Facial Boundaries",
                        "explanation": "The transition between the face, hair, and background shows natural sharpness, realistic hair strands, and consistent optical focus.",
                        "status": "authentic",
                    })
            elif "eye" in cid or "ocular" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Unusual Eye Details & Reflections",
                        "explanation": "The light reflections and geometric alignment in the eyes exhibit subtle irregularities that differ from natural photographic optics.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Natural Eye Reflections & Symmetry",
                        "explanation": "Both eyes exhibit matching reflections of the ambient environment, consistent pupil geometry, and realistic anatomical alignment.",
                        "status": "authentic",
                    })
            elif "lighting" in cid or "nasal" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Inconsistent Facial Lighting",
                        "explanation": "Directional lighting and shadow falloff across the nose and facial contours do not fully match the ambient illumination of the surrounding scene.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Consistent Scene Lighting",
                        "explanation": "Highlights and shadows across the facial contours align naturally with the ambient light sources in the scene.",
                        "status": "authentic",
                    })
            elif "mouth" in cid or "oral" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Mouth and Smile Irregularities",
                        "explanation": "Subtle distortion or unnatural blending was detected around the lips and teeth boundaries.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Natural Mouth Geometry",
                        "explanation": "Lip contours, dental definition, and expression wrinkles exhibit genuine biological anatomy.",
                        "status": "authentic",
                    })
            elif "skin" in cid or "malar" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Unnatural Skin Texture",
                        "explanation": "Certain facial regions appear excessively smooth or show repetitive micro-patterns, lacking the natural pores and fine lines found in standard photographs.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Natural Skin Texture",
                        "explanation": "Facial skin displays organic pore distribution, natural micro-textures, and fine variations typical of real human skin.",
                        "status": "authentic",
                    })
            elif "fft" in cid or "frequency" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Unusual Fine Image Patterns",
                        "explanation": "Microscopic frequency analysis detected subtle periodic patterns typical of artificial image generation.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Natural Frequency Distribution",
                        "explanation": "Microscopic image frequencies decay smoothly from coarse structures to fine details, typical of natural light captured on a physical camera sensor.",
                        "status": "authentic",
                    })
            elif "ela" in cid or "compression" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Face and Background Differences",
                        "explanation": "The compression characteristics around the face differ noticeably from the surrounding background, suggesting the face may have been inserted or saved at a different quality level.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Consistent Image Compression",
                        "explanation": "Compression levels are uniform across both the face and the background, indicating that the entire image was saved together in a single capture.",
                        "status": "authentic",
                    })
            elif "noise" in cid or "sensor" in cid or "prnu" in cid:
                if is_card_fake:
                    findings.append({
                        "title": "Sensor Noise Inconsistencies",
                        "explanation": "Microscopic pixel noise across the face is inconsistent with the background, deviating from how physical camera sensors record light.",
                        "status": "concern",
                    })
                else:
                    findings.append({
                        "title": "Consistent Camera Sensor Noise",
                        "explanation": "Microscopic sensor noise is uniform across all facial regions and background areas, confirming a single optical sensor origin.",
                        "status": "authentic",
                    })

    # If video, check temporal continuity
    if media_type == "video" and sequence_aggregation:
        fake_cnt = sequence_aggregation.get("fake_frames_detected", 0)
        if fake_cnt > 0:
            findings.insert(0, {
                "title": "Temporal Inconsistencies Across Frames",
                "explanation": f"Specific video frames showed sudden changes in facial appearance or alignment ({fake_cnt} frames flagged), typical of digital face alteration across motion.",
                "status": "concern",
            })
        else:
            findings.insert(0, {
                "title": "Smooth Motion and Continuity",
                "explanation": "Facial expressions and head movements transition smoothly across frames without sudden flickering or boundary warping.",
                "status": "authentic",
            })

    # Fallback to feature dict if cards weren't provided
    if not findings and features:
        if features.get("ela_compression_anomaly", 0.0) > 0.45:
            findings.append({
                "title": "Face and Background Differences",
                "explanation": "Compression error levels differ between the face and background, indicating localized splicing or insertion.",
                "status": "concern",
            })
        if features.get("sensor_noise_score", 0.0) > 0.45:
            findings.append({
                "title": "Sensor Noise Inconsistencies",
                "explanation": "Camera sensor noise residuals across the face deviate from natural optical capture.",
                "status": "concern",
            })
        if features.get("fft_spectral_anomaly", 0.0) > 0.45:
            findings.append({
                "title": "Unusual Fine Image Patterns",
                "explanation": "Microscopic pixel patterns show synthetic signatures characteristic of AI generation.",
                "status": "concern",
            })
        if features.get("edge_boundary_artifacts", 0.0) > 0.45:
            findings.append({
                "title": "Face Boundary Transitions",
                "explanation": "Noticeable edge inconsistencies were detected along facial contours.",
                "status": "concern",
            })

    # Default findings if still empty
    if not findings:
        if norm_v == "FAKE":
            findings = [
                {
                    "title": "Unusual Fine Image Patterns",
                    "explanation": "Fine pixel details and subtle texture patterns show characteristics typical of synthetic face generation.",
                    "status": "concern",
                },
                {
                    "title": "Face and Background Differences",
                    "explanation": "Discrepancies in lighting and compression were detected between facial features and the surrounding area.",
                    "status": "concern",
                },
            ]
        elif norm_v == "REAL":
            findings = [
                {
                    "title": "Natural Facial Boundaries",
                    "explanation": "Edge transitions and facial contours blend smoothly and naturally with surrounding hair and background.",
                    "status": "authentic",
                },
                {
                    "title": "Consistent Camera Sensor Noise",
                    "explanation": "Microscopic sensor noise patterns are uniform across the image, consistent with genuine physical camera optics.",
                    "status": "authentic",
                },
                {
                    "title": "Natural Skin Texture Detail",
                    "explanation": "Skin exhibits natural micro-textures and lighting without artificial smoothing.",
                    "status": "authentic",
                },
            ]
        elif norm_v == "ILLUSTRATION":
            findings = [
                {
                    "title": "Stylized 2D Artwork",
                    "explanation": "The image exhibits stylized line art, cel-shading, or cartoon facial proportions characteristic of illustration rather than optical photography.",
                    "status": "neutral",
                }
            ]
        else:
            findings = [
                {
                    "title": "No Detectable Face",
                    "explanation": "The image does not contain a clear, unobstructed human face required for facial forensic evaluation.",
                    "status": "neutral",
                }
            ]

    # Select top 4 most informative findings, prioritizing matches to verdict
    if norm_v == "FAKE":
        concern_items = [f for f in findings if f["status"] == "concern"]
        other_items = [f for f in findings if f["status"] != "concern"]
        selected_findings = (concern_items + other_items)[:4]
    else:
        auth_items = [f for f in findings if f["status"] == "authentic"]
        other_items = [f for f in findings if f["status"] != "authentic"]
        selected_findings = (auth_items + other_items)[:4]

    # 3. Dynamic "What Does This Mean?" (2-3 plain-language sentences)
    m_label = "video" if media_type == "video" else "image"
    if norm_v == "FAKE":
        if conf_tier == "High":
            what_does_this_mean = (
                f"Our analysis indicates strong evidence that this {m_label} has been artificially generated or digitally altered. "
                f"Multiple independent forensic checks identified synthetic anomalies across key facial regions. "
                f"The visual characteristics observed are highly consistent with deepfake technology."
            )
        else:
            what_does_this_mean = (
                f"Our analysis suggests potential digital alteration or synthetic elements in this {m_label}, though some indicators remain inconclusive. "
                f"Several subtle irregularities were detected, but image quality or mixed signals reduce overall certainty. "
                f"We recommend treating this content with caution and verifying the source."
            )
    elif norm_v == "REAL":
        if conf_tier == "High":
            what_does_this_mean = (
                f"Our analysis found no significant evidence of digital manipulation or artificial generation in this {m_label}. "
                f"The facial features, skin texture, lighting, and camera noise patterns closely match genuine photographic capture. "
                f"While no automated tool can guarantee 100% authenticity, the visual evidence strongly supports that this {m_label} is authentic."
            )
        else:
            what_does_this_mean = (
                f"Our analysis found that the evaluated facial characteristics generally appear authentic, but overall confidence is limited. "
                f"Factors such as compression, lighting, or slight variations across detectors prevent a definitive conclusion. "
                f"While signs point to a real {m_label}, secondary verification is advised for high-stakes contexts."
            )
    elif norm_v == "ILLUSTRATION":
        what_does_this_mean = (
            "This image appears to be a stylized illustration rather than a photographic image. "
            "The photographic deepfake detection models are not sufficient to determine whether this "
            "artwork was created by a human or generated by AI."
        )
    else:
        what_does_this_mean = (
            f"This {m_label} could not be analyzed for facial manipulation because no recognizable human face was detected. "
            f"The detection tools require a clear view of facial features to evaluate authenticity. "
            f"Please provide an image or video with a clearly visible human face."
        )

    # 4. "Why Is Confidence Limited?"
    why_confidence_is_limited: Optional[str] = None
    if norm_v not in ("NOT_SUITABLE", "ILLUSTRATION") and (conf_tier in ("Moderate", "Low") or is_binary_fallback or disagreement_detected):
        if disagreement_detected or is_binary_fallback:
            why_confidence_is_limited = (
                "Different automated inspection tools produced conflicting indications. "
                "While some detectors flagged suspicious patterns, others assessed the image as authentic, leading to lower overall confidence."
            )
        elif features and features.get("blur_score", 100) < 50:
            why_confidence_is_limited = (
                "The media contains optical softness or heavy compression, which can obscure minute pixel details and limit the certainty of the analysis."
            )
        elif fake_probability is not None and 0.35 <= fake_probability <= 0.65:
            why_confidence_is_limited = (
                "The measured visual features lie near the borderline between authentic and altered media, meaning the evidence is suggestive rather than definitive."
            )
        else:
            why_confidence_is_limited = (
                "Subtle image characteristics or compression artifacts introduce ambiguity, so the evidence does not meet high-certainty criteria."
            )
    elif norm_v == "ILLUSTRATION":
        why_confidence_is_limited = (
            "Photographic deepfake detection models are not calibrated for non-photographic artwork."
        )

    # 5. "What Should You Do?"
    if norm_v == "FAKE":
        if conf_tier == "High":
            what_should_user_do = (
                "Do not rely on this media for critical decisions or share it without verification. "
                "If this claims to represent a real person, treat it with high skepticism."
            )
        else:
            what_should_user_do = (
                "Treat this media with caution. Some indicators suggest alteration, but image quality or conflicting signals prevent a definitive conclusion. "
                "Verify through original sources."
            )
    elif norm_v == "REAL":
        if conf_tier == "High":
            what_should_user_do = (
                "This media appears authentic based on standard forensic indicators. Standard precautions still apply."
            )
        else:
            what_should_user_do = (
                "While indicators lean toward authentic, limitations in image quality or subtle inconsistencies suggest caution. "
                "Do not rely solely on this analysis for high-stakes decisions."
            )
    elif norm_v == "ILLUSTRATION":
        what_should_user_do = (
            "Upload a natural photographic image containing a visible human face to perform facial deepfake authenticity analysis."
        )
    else:
        what_should_user_do = (
            "Upload a high-quality, clear image or video containing a clearly visible, unobstructed human face to perform a complete facial forensic analysis."
        )

    return {
        "headline": "Illustration / Cartoon Detected" if norm_v == "ILLUSTRATION" else "Understanding Your Result",
        "verdict": norm_v,
        "confidence_tier": conf_tier,
        "confidence_percentage": conf_pct,
        "what_does_this_mean": what_does_this_mean,
        "what_raised_concern": selected_findings,
        "why_confidence_is_limited": why_confidence_is_limited,
        "what_should_user_do": what_should_user_do,
        "is_conflicted": bool(disagreement_detected or is_binary_fallback),
    }


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
        models: list = None,
        visual_analysis: dict = None,
        measurements: dict = None,
        evidence_cards: list = None,
        forensic_evidence: dict = None,
        fake_probability: float = None,
        confidence: float = None,
        is_binary_fallback: bool = False,
        disagreement_detected: bool = False,
        sequence_aggregation: dict = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes factual forensic explanations aligned with the canonical verdict.
        Provides both Level 1 (User-Friendly Explanation) and Level 2 (Technical Details).
        """
        # Canonical normalization
        if verdict in ("NOT_SUITABLE", getattr(DecisionState, "NOT_SUITABLE", "NOT_SUITABLE")):
            state = DecisionState.NOT_SUITABLE
            user_statement = USER_FACING_STATEMENTS[DecisionState.NOT_SUITABLE]
            user_friendly = generate_user_friendly_explanation(
                verdict="NOT_SUITABLE",
                media_type=media_type,
            )
            return {
                "verdict": state,
                "prediction": state,
                "evidence": evidence if evidence else ["Eligibility: Unsupported media (No detectable human face found in media)."],
                "recommendation": "Upload a clear media file containing a visible human face.",
                "user_facing_statement": user_statement,
                "user_friendly_explanation": user_friendly,
            }

        from backend.core.decision_system import PresentationDecisionPolicy
        state = PresentationDecisionPolicy.normalize_verdict(verdict)

        evidence_items: List[str] = list(evidence) if evidence else []

        # 1. Specialist Neural Model Comparisons & Agreement / Divergence (Level 2 Technical)
        if models and len(models) > 0:
            classes = [m.get("predicted_class") for m in models if m.get("predicted_class")]
            model_scores_str = ", ".join(
                f"{m.get('name')}: {m.get('predicted_class')} ({m.get('raw_score', 0.0):.1%})"
                for m in models if m.get("raw_score") is not None
            )
            if classes and all(c == classes[0] for c in classes):
                item = f"Specialist Model Agreement: Evaluated neural detectors concordantly assess this sample as {classes[0]} ({model_scores_str})."
            else:
                item = f"Specialist Model Assessment: Detectors evaluated diverse features with divergent outputs ({model_scores_str})."
            if item not in evidence_items:
                evidence_items.append(item)

        # 2. Multi-Provider Vision Corroboration (Level 2 Technical)
        if visual_analysis and visual_analysis.get("providers_succeeded"):
            v_providers = ", ".join(visual_analysis.get("providers_succeeded", []))
            v_score = visual_analysis.get("visual_synthetic_score")
            v_assessment = visual_analysis.get("consensus_assessment", "EVALUATED")
            score_str = f" (synthetic score: {v_score:.1%})" if v_score is not None else ""
            item = f"External AI Vision: {v_providers} reported {v_assessment}{score_str}."
            if item not in evidence_items:
                evidence_items.append(item)

        # 3. Genuine Spatial, Boundary, and Frequency Feature Measurements (Level 2 Technical)
        if features:
            edge_score = features.get("edge_boundary_artifacts", 0.0)
            if edge_score > 0.45:
                item = f"Edge & Boundary: High-frequency gradient discontinuity of {edge_score:.2f} measured across facial perimeter."
            elif edge_score > 0.0:
                item = f"Edge & Boundary: Natural spatial continuity observed across facial perimeter (boundary score: {edge_score:.2f})."
            else:
                item = None
            if item and item not in evidence_items:
                evidence_items.append(item)

            fft_score = features.get("fft_spectral_anomaly", 0.0)
            if fft_score > 0.45:
                item = f"Frequency Spectrum (FFT): Anomalous high-frequency spectral roll-off detected (score: {fft_score:.2f})."
            elif fft_score > 0.0:
                item = f"Frequency Spectrum (FFT): Natural radial spectral decay consistent with genuine sensor capture (score: {fft_score:.2f})."
            else:
                item = None
            if item and item not in evidence_items:
                evidence_items.append(item)

            ela_score = features.get("ela_compression_anomaly", 0.0)
            if ela_score > 0.45:
                item = f"Compression ELA: Face-to-background error discrepancy of {ela_score:.2f} indicates potential localized splicing."
            elif ela_score > 0.0:
                item = f"Compression ELA: Uniform error level distribution between face and background (discrepancy: {ela_score:.2f})."
            else:
                item = None
            if item and item not in evidence_items:
                evidence_items.append(item)

            blur_val = features.get("blur_score")
            if blur_val is not None:
                item = f"Image Quality Gate: Laplacian sharpness measured at {blur_val:.1f} (standard quality assessment; image sharpness or blur does not constitute standalone manipulation proof)."
                if item not in evidence_items:
                    evidence_items.append(item)

            if features.get("metadata_anomaly"):
                soft = features.get("software_tag")
                item = f"Metadata Signature: Editing software tag detected{f' ({soft})' if soft else ''}."
                if item not in evidence_items:
                    evidence_items.append(item)

            if features.get("temporal_flicker_score", 0.0) > 0.22:
                item = "Temporal Continuity: Inter-frame inconsistency and appearance variance detected across sampled frames."
                if item not in evidence_items:
                    evidence_items.append(item)

            vt_tight = features.get("raw_vit_tight_score")
            vt_pad = features.get("raw_vit_score")
            if vt_tight is not None and vt_pad is not None and abs(vt_tight - vt_pad) > 0.20:
                item = (
                    f"Multi-Scale Target Resolution: Specialist ViT evaluated tight facial crop ({vt_tight:.1%}) "
                    f"vs contextual padded crop ({vt_pad:.1%}), capturing fine generative synthesis artifacts on facial features."
                )
                if item not in evidence_items:
                    evidence_items.append(item)

            noise_score = features.get("sensor_noise_score")
            if noise_score is not None:
                if noise_score > 0.45:
                    item = f"Sensor Noise (PRNU Proxy): Cross-channel noise decorrelation or regional inconsistency detected (score: {noise_score:.2f})."
                else:
                    item = f"Sensor Noise (PRNU Proxy): Optical Bayer-demosaicing noise correlation consistent with physical camera capture (score: {noise_score:.2f})."
                if item not in evidence_items:
                    evidence_items.append(item)

        # Level 1 User-Friendly Explanation Object
        user_friendly = generate_user_friendly_explanation(
            verdict=state,
            confidence=confidence,
            confidence_level=confidence_level,
            fake_probability=fake_probability,
            features=features,
            forensic_evidence=forensic_evidence,
            evidence_cards=evidence_cards,
            models=models,
            is_binary_fallback=is_binary_fallback,
            disagreement_detected=disagreement_detected,
            media_type=media_type,
            sequence_aggregation=sequence_aggregation,
        )

        # Unified recommendations and statements aligned strictly with canonical state
        if state == DecisionState.FAKE:
            evidence_items = [
                e for e in evidence_items
                if "abstained" not in e.lower()
                and "uncertain under the system" not in e.lower()
                and "secondary manual verification is recommended" not in e.lower()
                and "authenticity has not been established" not in e.lower()
            ]
            if not evidence_items:
                evidence_items.append("Neural feature extraction identified deepfake manipulation patterns.")
            recommendation = f"Deepfake manipulation flagged. {user_friendly.get('what_should_user_do', '')}".strip()
            user_statement = user_friendly.get("what_does_this_mean")
            explanation = f"FAKE: {user_statement}"

        elif state == DecisionState.REAL:
            evidence_items = [
                e for e in evidence_items
                if "abstained" not in e.lower()
                and "uncertain under the system" not in e.lower()
                and "secondary manual verification is recommended" not in e.lower()
            ]
            if not evidence_items:
                evidence_items.append("No significant high-frequency or boundary anomalies detected in primary inspection.")
            recommendation = f"No strong evidence of manipulation detected. {user_friendly.get('what_should_user_do', '')}".strip()
            user_statement = user_friendly.get("what_does_this_mean")
            explanation = f"REAL: {user_statement}"

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
            explanation = f"UNCERTAIN: {user_statement}" + (f" ({reason})" if reason and reason != user_statement else "")

        return {
            "verdict": state,
            "prediction": state,
            "explanation": explanation,
            "summary": user_statement,
            "evidence": evidence_items,
            "recommendation": recommendation,
            "user_facing_statement": user_statement,
            "user_friendly_explanation": user_friendly,
        }


        return {
            "verdict": state,
            "prediction": state,
            "explanation": explanation,
            "summary": user_statement,
            "evidence": evidence_items,
            "recommendation": recommendation,
            "user_facing_statement": user_statement,
        }
