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

        uploaded_file = context_dict.get("uploaded_file", "N/A")
        media_type = context_dict.get("media_type", "N/A").upper()
        
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
        verdict = fusion.get("verdict", "N/A")

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
        if verdict in ("NOT_SUITABLE", "UNSUPPORTED MEDIA"):
            banner_color = colors.HexColor('#f8fafc')
            text_color = colors.HexColor('#334155')
            border_color = colors.HexColor('#94a3b8')
            verdict_str = "<b>VERDICT: NOT SUITABLE FOR FACE ANALYSIS</b>"
            conf_str = "<b>Status: Ineligible (No Face Detected)</b>"
        elif verdict in ("DEEPFAKE", "FAKE"):
            banner_color = colors.HexColor('#fef2f2')
            text_color = colors.HexColor('#991b1b')
            border_color = colors.HexColor('#f43f5e')
            verdict_str = f"<b>VERDICT: {verdict}</b>"
            conf_str = f"<b>Confidence: {conf_score}% ({conf_level} Confidence)</b>"
        elif verdict == "UNCERTAIN":
            banner_color = colors.HexColor('#fffbeb')
            text_color = colors.HexColor('#92400e')
            border_color = colors.HexColor('#f59e0b')
            verdict_str = f"<b>VERDICT: {verdict}</b>"
            conf_str = f"<b>Confidence: {conf_score}% ({conf_level} Confidence)</b>"
        else:
            banner_color = colors.HexColor('#ecfdf5')
            text_color = colors.HexColor('#065f46')
            border_color = colors.HexColor('#10b981')
            verdict_str = f"<b>VERDICT: {verdict}</b>"
            conf_str = f"<b>Confidence: {conf_score}% ({conf_level} Confidence)</b>"

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
        story.append(Spacer(1, 20))

        # Section Details
        story.append(Paragraph("1. Pipeline Validation Results", h2_style))
        val_text = f"<b>Status:</b> {validation_status}<br/><b>Detected issues:</b> {validation_issues}"
        story.append(Paragraph(val_text, body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"2. {media_type.capitalize()} Analysis Agent Trace", h2_style))
        story.append(Paragraph(agent_summary, body_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("3. Consensus & Fusion Decision", h2_style))
        if verdict == "NOT_SUITABLE":
            fusion_summary = "Media suitability verification determined that the uploaded content does not contain a detectable human face. Authenticity analysis was not performed as the deepfake detector requires facial features."
        else:
            fusion_summary = f"The specialized {media_type.lower()} analysis agent and confidence evaluation agent reached consensus. Classification decision yielded a final verdict of <b>{verdict}</b> with a score of <b>{conf_score}%</b>."
        story.append(Paragraph(fusion_summary, body_style))
        story.append(Spacer(1, 15))

        # Evidence
        story.append(Paragraph("Forensic Evidence Details", h2_style))
        for item in evidence:
            bullet_text = f"• {item}"
            story.append(Paragraph(bullet_text, ParagraphStyle('Bullet', parent=body_style, leftIndent=15, bulletIndent=5)))
        story.append(Spacer(1, 15))

        # Recommendation
        story.append(Paragraph("Audit Recommendation", h2_style))
        story.append(Paragraph(recommendation, body_style))
        story.append(Spacer(1, 40))

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


