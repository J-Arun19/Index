import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportGenerator:
    @staticmethod
    def generate_report(context_dict: dict) -> str:
        """
        Generates a professional PDF report from the AnalysisContext.
        Saves it inside backend/reports/ and returns the absolute file path.
        """
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(backend_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        analysis_id = context_dict.get("analysis_id", "N/A")
        report_filename = f"report_{analysis_id}.pdf"
        report_path = os.path.join(reports_dir, report_filename)

        user = context_dict.get("user") or {}
        user_name = user.get("name", "Guest")
        user_email = user.get("email", "guest@satyadrishti.ai")

        uploaded_file = str(context_dict.get("uploaded_file") or "N/A")
        media_type = str(context_dict.get("media_type") or "N/A").upper()
        
        validation = context_dict.get("validation_result") or {}
        validation_status = validation.get("status", "Unknown").upper()
        validation_issues = ", ".join(validation.get("issues", [])) or "None"

        is_video = media_type.lower() == "video"
        agent_res = context_dict.get("video_agent_result") if is_video else context_dict.get("image_agent_result")
        agent_res = agent_res or {}
        agent_summary = agent_res.get("summary", "N/A")

        confidence_eval = context_dict.get("confidence_result") or {}
        conf_score = confidence_eval.get("confidence", 0)
        conf_level = confidence_eval.get("level", "N/A")

        fusion = context_dict.get("fusion_result") or {}
        primary_verdict = fusion.get("primary_verdict") or fusion.get("verdict", "N/A")
        original_verdict = fusion.get("original_verdict", primary_verdict)
        is_binary_fallback = fusion.get("is_binary_fallback", False)
        reliability_warning = fusion.get("reliability_warning")
        verdict = primary_verdict

        explanation_data = context_dict.get("explanation") or {}
        evidence = explanation_data.get("evidence", [])
        recommendation = explanation_data.get("recommendation", "N/A")

        doc = SimpleDocTemplate(report_path, pagesize=letter,
                                rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor('#06b6d4'),
            spaceAfter=15
        )
        
        h2_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor('#334155'),
            leading=14
        )

        bold_label_style = ParagraphStyle(
            'BoldLabel',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        story = []

        # Title / Branding
        story.append(Paragraph("SatyaDrishti", title_style))
        story.append(Paragraph("<b>Digital Forensic Analysis Report</b>", ParagraphStyle('Sub', parent=body_style, fontSize=11, textColor=colors.HexColor('#64748b'))))
        story.append(Spacer(1, 15))

        # Metadata Table
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_data = [
            [Paragraph("Analysis ID:", bold_label_style), Paragraph(analysis_id, body_style), Paragraph("Date / Time:", bold_label_style), Paragraph(now, body_style)],
            [Paragraph("User Name:", bold_label_style), Paragraph(user_name, body_style), Paragraph("Email:", bold_label_style), Paragraph(user_email, body_style)],
            [Paragraph("File Name:", bold_label_style), Paragraph(uploaded_file, body_style), Paragraph("Media Type:", bold_label_style), Paragraph(media_type, body_style)]
        ]
        t = Table(meta_data, colWidths=[90, 160, 90, 160])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))

        # Verdict Summary Banner
        is_provisional = bool(is_binary_fallback or original_verdict in ("UNCERTAIN", "INCONCLUSIVE") or verdict in ("SUSPICIOUS - provisional", "INCONCLUSIVE", "UNCERTAIN"))
        fake_prob = fusion.get("fake_probability")
        fake_prob_pct = f"{round(float(fake_prob) * 100, 2)}%" if fake_prob is not None else "N/A"

        # Canonical normalization using centralized policy
        from backend.core.decision_system import PresentationDecisionPolicy
        canonical_v = PresentationDecisionPolicy.get_canonical_verdict(
            verdict=verdict,
            primary_verdict=primary_verdict,
            original_verdict=original_verdict,
            is_binary_fallback=is_binary_fallback,
            fake_probability=fake_prob,
        )

        if canonical_v == "NOT_SUITABLE":
            banner_color = colors.HexColor('#f8fafc')
            text_color = colors.HexColor('#334155')
            border_color = colors.HexColor('#94a3b8')
            verdict_str = "<b>VERDICT: NOT SUITABLE FOR FACE ANALYSIS</b><br/><font size=8 color='#64748b'>No detectable human face found in media</font>"
            conf_str = "<b>Status: Ineligible (No Face Detected)</b>"
        elif canonical_v == "FAKE":
            banner_color = colors.HexColor('#fef2f2')
            text_color = colors.HexColor('#991b1b')
            border_color = colors.HexColor('#f43f5e')
            verdict_str = "<b>VERDICT: FAKE</b><br/><font size=8 color='#991b1b'>Multi-signal consensus confirms deepfake manipulation patterns in media features.</font>"
            conf_str = f"<b>Confidence: {conf_score}% ({conf_level} Confidence)</b>"
        elif canonical_v == "REAL":
            banner_color = colors.HexColor('#ecfdf5')
            text_color = colors.HexColor('#065f46')
            border_color = colors.HexColor('#10b981')
            verdict_str = "<b>VERDICT: REAL</b><br/><font size=8 color='#047857'>Authentic natural media features observed. Notice: Absence of detected manipulation does not prove absolute authenticity.</font>"
            conf_str = f"<b>Confidence: {conf_score}% ({conf_level} Confidence)</b>"
        else:
            canonical_v = "UNCERTAIN"
            banner_color = colors.HexColor('#fffbeb')
            text_color = colors.HexColor('#92400e')
            border_color = colors.HexColor('#f59e0b')
            subtext = "Disagreement detected between specialist detectors. Authenticity has NOT been established; manual verification is required." if is_binary_fallback else "Media quality or indeterminate forensic signals prevent definitive classification."
            verdict_str = f"<b>VERDICT: UNCERTAIN</b><br/><font size=8 color='#92400e'>{subtext}</font>"
            if is_binary_fallback and primary_verdict in ("FAKE", "REAL"):
                verdict_str += f"<br/><font size=7 color='#b45309'>Unverified model suggestion ({primary_verdict}, estimated probability: {fake_prob_pct}) — not an established authenticity verdict.</font>"
            conf_str = "<b>Status: Uncertified / Provisional</b><br/><font size=8>Confidence: Not certified</font>"


        banner_data = [
            [Paragraph(verdict_str, ParagraphStyle('BVerdict', parent=body_style, fontSize=14, textColor=text_color, fontName='Helvetica-Bold')),
             Paragraph(conf_str, ParagraphStyle('BConf', parent=body_style, fontSize=12, textColor=colors.HexColor('#475569'), alignment=2))]
        ]
        tb = Table(banner_data, colWidths=[250, 250])
        tb.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), banner_color),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1.5, border_color),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(tb)

        # Fallback Warning Note (if binary fallback applied)
        if is_binary_fallback and reliability_warning:
            story.append(Spacer(1, 10))
            warning_style = ParagraphStyle(
                'WarningText',
                parent=body_style,
                fontSize=9,
                textColor=colors.HexColor('#92400e'),
                leading=13
            )
            warn_data = [
                [Paragraph(f"<b>Caution:</b> {reliability_warning}", warning_style)]
            ]
            tw = Table(warn_data, colWidths=[500])
            tw.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fffbeb')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#f59e0b')),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(tw)

        story.append(Spacer(1, 20))


        # Section Details
        story.append(Paragraph("1. Pipeline Validation & Media Suitability", h2_style))
        val_text = f"<b>Status:</b> {validation_status}<br/><b>Detected issues:</b> {validation_issues}"
        story.append(Paragraph(val_text, body_style))
        story.append(Spacer(1, 10))

        # Facial Region & Image Quality Overview
        q_gate = fusion.get("quality_gate") or agent_res.get("quality_gate") or {}
        face_detected = fusion.get("face_detected", agent_res.get("face_detected", True))
        face_bbox = fusion.get("face_bbox") or agent_res.get("face_bbox")
        bbox_str = f"[{face_bbox[0]}, {face_bbox[1]}, {face_bbox[2]}, {face_bbox[3]}]" if (face_bbox and len(face_bbox) == 4) else ("Detected (Full Face)" if face_detected else "None")
        blur_val = q_gate.get("blur_score", "N/A")
        blur_str = f"{blur_val:.1f}" if isinstance(blur_val, (int, float)) else str(blur_val)
        meas = agent_res.get("measurements", {}) or fusion.get("measurements", {})
        meas_res = meas.get("resolution", {}) if isinstance(meas, dict) else {}
        if isinstance(meas_res, dict) and meas_res.get("formatted") and meas_res.get("formatted") != "[object Object]":
            res_val = str(meas_res.get("formatted"))
        else:
            raw_res = q_gate.get("resolution")
            if isinstance(raw_res, dict):
                w = raw_res.get("width") or raw_res.get("w")
                h = raw_res.get("height") or raw_res.get("h")
                res_val = f"{w} × {h} px" if (w and h and w != "N/A" and h != "N/A" and int(w or 0) > 0 and int(h or 0) > 0) else "Not assessed"
            elif isinstance(raw_res, (list, tuple)) and len(raw_res) >= 2 and int(raw_res[0] or 0) > 0 and int(raw_res[1] or 0) > 0:
                res_val = f"{raw_res[0]} × {raw_res[1]} px"
            elif isinstance(raw_res, str) and raw_res not in ("[object Object]", "Standard Native Resolution"):
                res_val = raw_res
            else:
                res_val = "Not assessed"
        
        story.append(Paragraph("2. Image Evidence & Quality Overview", h2_style))
        overview_data = [
            [Paragraph("Face Detection:", bold_label_style), Paragraph("Detected & Analyzed" if face_detected else "No Face Detected", body_style),
             Paragraph("Face Region (BBox):", bold_label_style), Paragraph(bbox_str, body_style)],
            [Paragraph("Blur Metric (Laplacian):", bold_label_style), Paragraph(f"{blur_str} (Variance)", body_style),
             Paragraph("Resolution / Grid:", bold_label_style), Paragraph(str(res_val), body_style)],
            [Paragraph("Estimated Fake Probability:", bold_label_style), Paragraph(fake_prob_pct, body_style),
             Paragraph("Prediction Quality Tier:", bold_label_style), Paragraph("Provisional" if is_provisional else "Calibrated Standard", body_style)],
        ]
        to = Table(overview_data, colWidths=[120, 130, 120, 130])
        to.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(to)
        story.append(Spacer(1, 12))

        # Model Comparison Table
        models_list = fusion.get("models") or agent_res.get("models") or []
        if models_list:
            story.append(Paragraph("Model Inference & Specialist Task Breakdown", h2_style))
            m_headers = [
                Paragraph("<b>Model Name</b>", bold_label_style),
                Paragraph("<b>Specialist Task</b>", bold_label_style),
                Paragraph("<b>Class</b>", bold_label_style),
                Paragraph("<b>Score</b>", bold_label_style),
                Paragraph("<b>Calibration Status</b>", bold_label_style),
            ]
            m_rows = [m_headers]
            for m in models_list:
                pred_c = m.get("predicted_class", "N/A")
                c_col = "#dc2626" if pred_c == "FAKE" else ("#16a34a" if pred_c == "REAL" else "#64748b")
                raw_s = m.get("raw_score")
                s_str = f"{raw_s*100:.1f}%" if (isinstance(raw_s, (int, float)) and raw_s is not None) else "N/A"
                m_rows.append([
                    Paragraph(f"<b>{m.get('name', 'Model')}</b>", body_style),
                    Paragraph(f"<font size=7 color='#475569'>{m.get('task', 'Forensic Detection')}</font>", body_style),
                    Paragraph(f"<font color='{c_col}'><b>{pred_c}</b></font>", body_style),
                    Paragraph(s_str, body_style),
                    Paragraph(f"<font size=7 color='#64748b'>{m.get('calibration_status', 'Raw')}</font>", body_style),
                ])
            m_table = Table(m_rows, colWidths=[120, 150, 60, 60, 110])
            m_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(m_table)
            story.append(Spacer(1, 12))

        story.append(Paragraph(f"3. {media_type.capitalize()} Analysis Agent Consensus", h2_style))
        summary_override = context_dict.get("summary") or fusion.get("summary") or agent_res.get("summary")
        if summary_override and len(summary_override.strip()) > 10:
            fusion_summary = summary_override
        elif canonical_v == "NOT_SUITABLE":
            fusion_summary = "Media suitability verification determined that the uploaded content does not contain a detectable human face. Authenticity analysis was not performed as the deepfake detector requires facial features."
        elif canonical_v == "UNCERTAIN":
            fusion_summary = f"The specialized {media_type.lower()} analysis agent and forensic checks evaluated the media. Insufficient evidence or conflicting signals prevent reliable classification; canonical verdict is <b>UNCERTAIN</b>."
        elif canonical_v == "FAKE":
            fusion_summary = f"Multi-signal consensus confirms synthetic manipulation indicators. Final verdict is <b>FAKE</b> with fake probability <b>{fake_prob_pct}</b> and confidence <b>{conf_score}%</b>."
        else:
            fusion_summary = f"Multi-signal evaluation completed across facial regions and frequency spectrum. No strong manipulation evidence detected. Final verdict is <b>REAL</b> with score <b>{conf_score}%</b>. (Limitation: Absence of detected manipulation does not prove absolute authenticity)."
        story.append(Paragraph(fusion_summary, body_style))
        story.append(Spacer(1, 12))

        # Multi-Face Tracking Summary Table for Videos
        tracked_faces = fusion.get("tracked_faces") or agent_res.get("tracked_faces") or []
        if is_video and tracked_faces:
            story.append(Paragraph("Per-Face Temporal Facial Forensics (Tracked Faces)", h2_style))
            t_headers = [
                Paragraph("<b>Track ID</b>", bold_label_style),
                Paragraph("<b>Track Verdict</b>", bold_label_style),
                Paragraph("<b>Fake Prob</b>", bold_label_style),
                Paragraph("<b>Seam Flicker</b>", bold_label_style),
                Paragraph("<b>Oral Texture</b>", bold_label_style),
                Paragraph("<b>Anomalies</b>", bold_label_style),
            ]
            t_rows = [t_headers]
            for tf in tracked_faces[:4]:
                tf_v = tf.get("verdict", "INCONCLUSIVE")
                v_color = "#dc2626" if tf_v == "FAKE" else ("#16a34a" if tf_v == "REAL" else "#d97706")
                t_rows.append([
                    Paragraph(f"<b>Track {tf.get('track_id', 'T1')}</b><br/><font size=7 color='#64748b'>({tf.get('total_frames', 0)} frames)</font>", body_style),
                    Paragraph(f"<font color='{v_color}'><b>{tf_v}</b></font>", body_style),
                    Paragraph(f"{(tf.get('fake_probability', 0.0) * 100):.1f}%", body_style),
                    Paragraph(f"{tf.get('boundary_flicker', 0.0):.1f}", body_style),
                    Paragraph(f"{tf.get('mouth_texture', 0.0):.1f}", body_style),
                    Paragraph(f"{tf.get('anomalies_detected', 0)}", body_style),
                ])
            tf_table = Table(t_rows, colWidths=[90, 85, 80, 85, 85, 75])
            tf_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ]))
            story.append(tf_table)
            if len(tracked_faces) > 1:
                story.append(Spacer(1, 4))
                story.append(Paragraph("<font size=7 color='#64748b'>* Multi-person isolation rule: Each face is tracked independently. A single manipulated face drives the overall video verdict.</font>", body_style))
            story.append(Spacer(1, 10))

        # 4. Strongest Evidence Findings (Snapshots for Video / Regional Cards for Image)
        cards = fusion.get("evidence_snapshots") or fusion.get("evidence_cards") or agent_res.get("evidence_snapshots") or agent_res.get("evidence_cards") or []
        if cards:
            section_title = "4. Strongest Evidence Snapshots & Temporal Observations" if is_video else "4. Strongest Forensic Evidence & Regional Observations"
            story.append(Paragraph(section_title, h2_style))
            for idx, c in enumerate(cards[:6], 1):
                c_title = c.get("finding_title") or c.get("title", f"Forensic Finding {idx}")
                c_timestamp = c.get("timestamp")
                c_source = c.get("evidence_source")
                c_region = c.get("region_name", "Facial Region")
                c_support = c.get("verdict_support", "").replace("_", " ")
                c_strength = c.get("evidence_strength", "").replace("_", " ")
                c_measurement = c.get("measurement_label", "")
                c_observed = c.get("observed_finding", "")
                c_plain = c.get("plain_explanation", "")
                c_limit = c.get("limitations", "")

                card_box_color = colors.HexColor('#fef2f2') if "FAKE" in c_support else colors.HexColor('#f0fdf4')
                card_border_color = colors.HexColor('#fecaca') if "FAKE" in c_support else colors.HexColor('#bbf7d0')

                if is_video and c_timestamp:
                    header_line = f"<b>{idx}. {c_title}</b> — <font size=8 color='#0284c7'><b>[{c_timestamp}]</b></font> | <font size=8 color='#475569'>Source: {c_source or 'Temporal Analyzer'}</font>"
                else:
                    header_line = f"<b>{idx}. {c_title}</b> — <font size=8 color='#64748b'>Region: {c_region}</font>"

                card_content = [
                    Paragraph(header_line, bold_label_style),
                    Paragraph(f"<b>Observed Measurement:</b> {c_measurement}", ParagraphStyle('M', parent=body_style, fontSize=9)),
                    Paragraph(f"<b>Finding:</b> {c_observed}", ParagraphStyle('F', parent=body_style, fontSize=9)),
                    Paragraph(f"<b>Significance ({c_support} | {c_strength}):</b> {c_plain}", ParagraphStyle('P', parent=body_style, fontSize=9)),
                    Paragraph(f"<b>Known Limitations & Alternatives:</b> <font color='#64748b'>{c_limit}</font>", ParagraphStyle('L', parent=body_style, fontSize=8)),
                ]
                card_table = Table([[card_content]], colWidths=[500])
                card_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), card_box_color),
                    ('BOX', (0,0), (-1,-1), 1, card_border_color),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(card_table)
                story.append(Spacer(1, 6))

            # Audio-Visual Temporal Synchronization Note for Video
            if is_video:
                av_sync = fusion.get("audio_visual_sync") or agent_res.get("audio_visual_sync") or {}
                av_analyzed = av_sync.get("analyzed", False)
                av_reason = av_sync.get("reason", "Lip-sync analysis not performed (audio-visual temporal model not configured).")

                av_content = [
                    Paragraph(f"<b>Audio-Visual Temporal Synchronization:</b> <font color='#b45309'><b>{'Analyzed' if av_analyzed else 'Not Analyzed'}</b></font>", bold_label_style),
                    Paragraph(f"<font size=8 color='#475569'>{av_reason}</font>", ParagraphStyle('AV', parent=body_style, fontSize=8)),
                ]
                av_table = Table([[av_content]], colWidths=[500])
                av_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fffbeb')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#fde68a')),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ]))
                story.append(av_table)
                story.append(Spacer(1, 6))

            story.append(Spacer(1, 10))

        # Additional Evidence Details
        story.append(Paragraph("5. Full Evidence Summary", h2_style))
        for item in evidence:
            bullet_text = f"• {item}"
            story.append(Paragraph(bullet_text, ParagraphStyle('Bullet', parent=body_style, leftIndent=15, bulletIndent=5)))
        story.append(Spacer(1, 12))

        # Audit Recommendation
        story.append(Paragraph("Audit Recommendation", h2_style))
        story.append(Paragraph(recommendation, body_style))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph("This document was generated automatically by the SatyaDrishti multi-agent digital media authentication platform.", ParagraphStyle('Foot', parent=body_style, fontSize=8, textColor=colors.HexColor('#94a3b8'), alignment=1)))

        doc.build(story)

        # Upload report to Supabase Storage and store metadata in reports table if configured
        try:
            from backend.services.supabase_service import SupabaseService
            from backend.services.database_service import DatabaseService
            
            if SupabaseService.verify_connection():
                report_filename = f"report_{analysis_id}.pdf"
                supabase_report_path = SupabaseService.upload_report(report_path, report_filename)
                if supabase_report_path:
                    bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
                    public_report_url = SupabaseService.get_client().storage.from_(bucket).get_public_url(report_filename)
                    file_size = os.path.getsize(report_path)
                    DatabaseService.save_report_metadata(
                        analysis_id=analysis_id,
                        report_url=public_report_url,
                        file_size=file_size,
                        report_path=supabase_report_path
                    )
        except Exception as e:
            print(f"ReportGenerator: Failed to upload report metadata to Supabase: {str(e)}")

        return report_path


