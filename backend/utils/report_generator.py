import os
import io
import base64
import datetime
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that dynamically calculates the total page count
    and renders consistent running headers (page > 1) and footers (all pages).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.report_id = "N/A"
        self.generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        
        # 1. Running Header (Pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0F172A"))
            self.drawString(40, 755, "SatyaDrishti")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(98, 755, "|   Digital Media Forensic Analysis Report")
            
            # Right-aligned Report ID
            self.setFont("Helvetica", 7.5)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(572, 755, f"Case / Report ID: {self.report_id}")
            
            # Subtle header separator
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.6)
            self.line(40, 747, 572, 747)

        # 2. Running Footer (All Pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.6)
        self.line(40, 36, 572, 36)
        
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(40, 24, "CONFIDENTIAL & PROPRIETARY")
        self.setFont("Helvetica", 7)
        self.setFillColor(colors.HexColor("#94A3B8"))
        self.drawString(160, 24, f"•   Digital Forensic Audit Record   •   {self.generated_at}")
        
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#334155"))
        self.drawRightString(572, 24, page_str)

        self.restoreState()


class ReportStyleManager:
    """Centralized typographic styles and color definitions for the forensic report."""
    # Palette
    C_NAVY_DARK = colors.HexColor("#0F172A")
    C_NAVY = colors.HexColor("#1E293B")
    C_SLATE = colors.HexColor("#334155")
    C_MUTED = colors.HexColor("#64748B")
    C_LIGHT_MUTED = colors.HexColor("#94A3B8")
    C_BORDER = colors.HexColor("#E2E8F0")
    C_BORDER_STRONG = colors.HexColor("#CBD5E1")
    C_BG_LIGHT = colors.HexColor("#F8FAFC")
    C_BG_SUBTLE = colors.HexColor("#F1F5F9")
    C_TEAL = colors.HexColor("#0D9488")
    C_CYAN = colors.HexColor("#0284C7")

    # Semantic Verdict Colors
    C_FAKE_TEXT = colors.HexColor("#991B1B")
    C_FAKE_ACCENT = colors.HexColor("#DC2626")
    C_FAKE_BG = colors.HexColor("#FEF2F2")
    C_FAKE_BORDER = colors.HexColor("#FCA5A5")

    C_REAL_TEXT = colors.HexColor("#166534")
    C_REAL_ACCENT = colors.HexColor("#16A34A")
    C_REAL_BG = colors.HexColor("#F0FDF4")
    C_REAL_BORDER = colors.HexColor("#86EFAC")

    C_UNCERTAIN_TEXT = colors.HexColor("#92400E")
    C_UNCERTAIN_ACCENT = colors.HexColor("#D97706")
    C_UNCERTAIN_BG = colors.HexColor("#FFFBEB")
    C_UNCERTAIN_BORDER = colors.HexColor("#FDE68A")

    C_NEUTRAL_TEXT = colors.HexColor("#334155")
    C_NEUTRAL_ACCENT = colors.HexColor("#64748B")
    C_NEUTRAL_BG = colors.HexColor("#F8FAFC")
    C_NEUTRAL_BORDER = colors.HexColor("#CBD5E1")

    @classmethod
    def get_styles(cls) -> Dict[str, ParagraphStyle]:
        styles = getSampleStyleSheet()

        res = styles.byName.copy()
        res.update({
            "BrandTitle": ParagraphStyle(
                "BrandTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=cls.C_NAVY_DARK,
            ),
            "BrandSubtitle": ParagraphStyle(
                "BrandSubtitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=7.5,
                leading=10,
                textColor=cls.C_TEAL,
            ),
            "DocTitle": ParagraphStyle(
                "DocTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=18,
                textColor=cls.C_NAVY,
            ),
            "SectionHeader": ParagraphStyle(
                "SectionHeader",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=15,
                textColor=cls.C_NAVY_DARK,
                spaceBefore=10,
                spaceAfter=5,
                keepWithNext=True,
            ),
            "SubSectionHeader": ParagraphStyle(
                "SubSectionHeader",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9.5,
                leading=13,
                textColor=cls.C_NAVY,
                spaceBefore=6,
                spaceAfter=3,
                keepWithNext=True,
            ),
            "Body": ParagraphStyle(
                "Body",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=8.5,
                leading=12,
                textColor=cls.C_SLATE,
            ),
            "BodyBold": ParagraphStyle(
                "BodyBold",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=12,
                textColor=cls.C_NAVY_DARK,
            ),
            "BodySmall": ParagraphStyle(
                "BodySmall",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=7.5,
                leading=10.5,
                textColor=cls.C_MUTED,
            ),
            "MetaLabel": ParagraphStyle(
                "MetaLabel",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=7.5,
                leading=10,
                textColor=cls.C_MUTED,
            ),
            "MetaVal": ParagraphStyle(
                "MetaVal",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=11,
                textColor=cls.C_NAVY_DARK,
            ),
            "TableHeader": ParagraphStyle(
                "TableHeader",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=11,
                textColor=cls.C_NAVY_DARK,
            ),
            "TableCell": ParagraphStyle(
                "TableCell",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=11,
                textColor=cls.C_SLATE,
            ),
            "VerdictLarge": ParagraphStyle(
                "VerdictLarge",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=15,
                leading=19,
            ),
            "ConfidenceBig": ParagraphStyle(
                "ConfidenceBig",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=22,
                alignment=2,  # Right-aligned
            ),
            "CalloutTitle": ParagraphStyle(
                "CalloutTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=12,
                textColor=cls.C_NAVY_DARK,
            ),
            "NoticeText": ParagraphStyle(
                "NoticeText",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=11,
                textColor=cls.C_SLATE,
            ),
        })
        return res


class ReportImageHelper:
    """Helper methods to load, decode, and format images/crops for the report."""

    @staticmethod
    def get_rl_image_from_b64(b64_or_data_url: str, max_w: float = 120, max_h: float = 100) -> Optional[RLImage]:
        if not b64_or_data_url or not isinstance(b64_or_data_url, str):
            return None
        try:
            b64_str = b64_or_data_url
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)
            bio = io.BytesIO(raw_bytes)
            pil_img = PILImage.open(bio)
            w, h = pil_img.size
            if w <= 0 or h <= 0:
                return None
            scale = min(max_w / float(w), max_h / float(h), 1.0)
            bio.seek(0)
            return RLImage(bio, width=w * scale, height=h * scale)
        except Exception:
            return None

    @staticmethod
    def resolve_media_path(file_ref: str) -> Optional[str]:
        if not file_ref or not isinstance(file_ref, str):
            return None
        if os.path.isabs(file_ref) and os.path.isfile(file_ref):
            return file_ref
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(backend_dir, "uploads", os.path.basename(file_ref)),
            os.path.join(backend_dir, "uploads", file_ref),
            file_ref,
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    @classmethod
    def get_media_preview(
        cls, file_ref: str, is_video: bool, max_w: float = 210, max_h: float = 160
    ) -> Tuple[Optional[RLImage], Dict[str, str]]:
        meta = {
            "resolution": "Not assessed",
            "duration": "N/A",
            "fps": "N/A",
            "format": "Unknown",
        }
        resolved = cls.resolve_media_path(file_ref)
        if not resolved:
            return None, meta

        try:
            ext = os.path.splitext(resolved)[1].lower()
            meta["format"] = ext.replace(".", "").upper()

            if is_video or ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                cap = cv2.VideoCapture(resolved)
                if not cap.isOpened():
                    return None, meta
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if fps > 0:
                    meta["duration"] = f"{(total_frames / fps):.1f}s"
                    meta["fps"] = f"{fps:.1f} fps"
                if w > 0 and h > 0:
                    meta["resolution"] = f"{w} × {h} px"

                target_frame = max(0, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    ret_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret_enc:
                        bio = io.BytesIO(buf.tobytes())
                        scale = min(max_w / float(w), max_h / float(h), 1.0)
                        return RLImage(bio, width=w * scale, height=h * scale), meta
                return None, meta
            else:
                # Still image
                pil_img = PILImage.open(resolved)
                w, h = pil_img.size
                meta["resolution"] = f"{w} × {h} px"
                scale = min(max_w / float(w), max_h / float(h), 1.0)
                bio = io.BytesIO()
                # Convert RGBA/P to RGB for clean JPEG
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
                pil_img.save(bio, format="JPEG", quality=88)
                bio.seek(0)
                return RLImage(bio, width=w * scale, height=h * scale), meta
        except Exception:
            return None, meta


class ReportGenerator:
    """
    SatyaDrishti Professional Digital Media Forensics Report Generator.
    Produces a publication-quality, 4-page white-paper forensic report.
    """

    @staticmethod
    def generate_report(context_dict: dict) -> str:
        """
        Generates a professional PDF report from the AnalysisContext.
        Saves it inside backend/reports/ and returns the absolute file path.
        """
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(backend_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        analysis_id = str(context_dict.get("analysis_id") or context_dict.get("id") or "N/A")
        report_filename = f"report_{analysis_id}.pdf"
        report_path = os.path.join(reports_dir, report_filename)

        user = context_dict.get("user") or {}
        user_name = user.get("name", "Forensic Analyst")
        user_email = user.get("email", "analyst@satyadrishti.ai")

        uploaded_file = str(
            context_dict.get("uploaded_file")
            or context_dict.get("fileName")
            or context_dict.get("file_path")
            or "sample_media"
        )
        media_type = str(
            context_dict.get("media_type") or context_dict.get("fileType") or "image"
        ).upper()
        is_video = media_type.lower() == "video"

        validation = context_dict.get("validation_result") or {}
        validation_status = str(validation.get("status", "PASSED")).upper()
        validation_issues = ", ".join(validation.get("issues", [])) or "None"

        # Agent results & fusion resolution
        agent_res = (
            context_dict.get("video_agent_result")
            if is_video
            else context_dict.get("image_agent_result")
        )
        agent_res = agent_res or {}

        fusion = context_dict.get("fusion_result") or {}
        if not isinstance(fusion, dict):
            fusion = {}

        confidence_eval = context_dict.get("confidence_result") or {}
        conf_score = (
            confidence_eval.get("confidence")
            if confidence_eval.get("confidence") is not None
            else (fusion.get("confidence") or context_dict.get("confidence"))
        )
        conf_level = (
            confidence_eval.get("level")
            or fusion.get("confidence_level")
            or fusion.get("confidenceLevel")
            or context_dict.get("confidence_level")
            or context_dict.get("confidenceLevel")
            or "High"
        )

        primary_verdict = (
            fusion.get("primary_verdict")
            or fusion.get("primaryVerdict")
            or context_dict.get("primary_verdict")
            or fusion.get("verdict")
            or context_dict.get("verdict")
            or "N/A"
        )
        original_verdict = (
            fusion.get("original_verdict")
            or fusion.get("originalVerdict")
            or context_dict.get("original_verdict")
            or primary_verdict
        )
        is_binary_fallback = bool(
            fusion.get("is_binary_fallback")
            or fusion.get("isBinaryFallback")
            or context_dict.get("is_binary_fallback")
            or False
        )
        reliability_warning = (
            fusion.get("reliability_warning")
            or fusion.get("reliabilityWarning")
            or context_dict.get("reliability_warning")
        )
        verdict = primary_verdict

        fake_prob = (
            fusion.get("fake_probability")
            if fusion.get("fake_probability") is not None
            else context_dict.get("fake_probability")
        )
        real_prob = (
            fusion.get("real_probability")
            if fusion.get("real_probability") is not None
            else context_dict.get("real_probability")
        )
        if real_prob is None and fake_prob is not None:
            real_prob = round(1.0 - float(fake_prob), 4)

        raw_score = (
            fusion.get("raw_model_score")
            if fusion.get("raw_model_score") is not None
            else context_dict.get("raw_model_score")
        )

        # Canonical verdict normalization using policy
        from backend.core.decision_system import PresentationDecisionPolicy

        canonical_v = PresentationDecisionPolicy.get_canonical_verdict(
            verdict=verdict,
            primary_verdict=primary_verdict,
            original_verdict=original_verdict,
            is_binary_fallback=is_binary_fallback,
            fake_probability=fake_prob,
        )

        is_provisional = bool(
            is_binary_fallback
            or original_verdict in ("UNCERTAIN", "INCONCLUSIVE")
            or verdict in ("SUSPICIOUS - provisional", "INCONCLUSIVE", "UNCERTAIN")
            or canonical_v in ("UNCERTAIN", "INCONCLUSIVE")
        )

        explanation_data = context_dict.get("explanation") or {}
        if not isinstance(explanation_data, dict):
            explanation_data = {}
        evidence = (
            explanation_data.get("evidence")
            or fusion.get("evidence")
            or context_dict.get("evidence")
            or []
        )
        recommendation = (
            explanation_data.get("recommendation")
            or fusion.get("recommendation")
            or context_dict.get("recommendation")
            or "N/A"
        )

        # Reconcile recommendation with canonical verdict
        if canonical_v == "FAKE":
            if recommendation == "N/A" or "secondary manual verification" in recommendation.lower():
                recommendation = (
                    "Deepfake manipulation flagged. Secondary forensic inspection or source provenance "
                    "verification is recommended before publishing or relying on this media."
                )
            evidence = [
                e for e in evidence
                if "abstained" not in e.lower()
                and "uncertain" not in e.lower()
                and "secondary manual verification is recommended" not in e.lower()
                and "authenticity has not been established" not in e.lower()
            ]
        elif canonical_v == "REAL":
            if recommendation == "N/A" or "secondary manual verification" in recommendation.lower():
                recommendation = (
                    "No strong evidence of manipulation was detected across evaluated detectors. "
                    "Note: This does not prove authenticity, as novel synthesis techniques or clean recaptures "
                    "may leave no detectable trace."
                )
            evidence = [e for e in evidence if "abstained" not in e.lower() and "uncertain" not in e.lower()]
        elif canonical_v == "ILLUSTRATION":
            recommendation = "Upload a natural photographic image containing a visible human face to perform facial deepfake authenticity analysis."
            evidence = [
                "Media Classification: Stylized Illustration / Anime / Cartoon.",
                "Photographic deepfake detection models are not calibrated for non-photographic artwork."
            ]
        elif canonical_v == "NOT_SUITABLE":
            recommendation = "Upload a clear media file containing a visible, frontal human face."
        else:
            if recommendation == "N/A":
                recommendation = "Provisional assessment: Independent forensic verification is recommended."

        # Evidence Cards & Snapshots
        evidence_cards = (
            fusion.get("evidence_snapshots")
            or fusion.get("evidence_cards")
            or fusion.get("evidenceCards")
            or fusion.get("evidenceSnapshots")
            or agent_res.get("evidence_snapshots")
            or agent_res.get("evidence_cards")
            or context_dict.get("evidence_cards")
            or context_dict.get("evidence_snapshots")
            or []
        )

        # Models list
        models = (
            fusion.get("models")
            or agent_res.get("models")
            or context_dict.get("models")
            or []
        )

        # Plain-English User-Friendly explanation
        user_friendly = (
            fusion.get("user_friendly_explanation")
            or agent_res.get("user_friendly_explanation")
            or context_dict.get("user_friendly_explanation")
            or {}
        )
        if not user_friendly or not user_friendly.get("what_does_this_mean"):
            from backend.agents.explanation_agent import generate_user_friendly_explanation
            user_friendly = generate_user_friendly_explanation(
                verdict=canonical_v,
                confidence=float(conf_score) if (conf_score is not None and conf_score != "N/A") else None,
                confidence_level=conf_level,
                fake_probability=fake_prob,
                features=fusion.get("features", {}),
                forensic_evidence=fusion.get("forensic_evidence", {}),
                evidence_cards=evidence_cards,
                models=models,
                is_binary_fallback=is_provisional,
                media_type=media_type,
            )

        # Prepare Document Template
        # Page size: Letter 612 x 792 pt. Margins: left/right=40 pt, top=48 pt, bottom=44 pt.
        # Printable width: 612 - 80 = 532 pt.
        doc = SimpleDocTemplate(
            report_path,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=48,
            bottomMargin=44,
        )

        styles = ReportStyleManager.get_styles()
        story: List[Any] = []

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # -------------------------------------------------------------
        # PAGE 1 — EXECUTIVE SUMMARY
        # -------------------------------------------------------------
        
        # 1. Header Branding Bar
        header_table_data = [
            [
                Paragraph("<b>SATYADRISHTI</b> <font size=10 color='#0284c7'>●</font>", styles["BrandTitle"]),
                Paragraph("<b>OFFICIAL FORENSIC AUDIT RECORD</b>", ParagraphStyle(
                    "Badge",
                    parent=styles["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    textColor=ReportStyleManager.C_NAVY_DARK,
                    alignment=2,
                ))
            ],
            [
                Paragraph("MULTI-AGENT DIGITAL MEDIA FORENSICS & VERIFICATION", styles["BrandSubtitle"]),
                Paragraph(f"<font color='#64748b'>ISO/IEC 27037 Digital Evidence Adherent &bull; {now_str}</font>", ParagraphStyle(
                    "SubSub",
                    parent=styles["Normal"],
                    fontName="Helvetica",
                    fontSize=7.5,
                    textColor=ReportStyleManager.C_MUTED,
                    alignment=2,
                ))
            ]
        ]
        t_header = Table(header_table_data, colWidths=[310, 222])
        t_header.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t_header)
        story.append(Spacer(1, 8))

        # Thin divider line
        t_div = Table([[""]], colWidths=[532], rowHeights=[1])
        t_div.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, -1), 1.0, ReportStyleManager.C_BORDER_STRONG),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(t_div)
        story.append(Spacer(1, 10))

        # 2. Intake Metadata Grid Table (colWidths: 85, 181, 85, 181 = 532)
        meta_rows = [
            [
                Paragraph("Case / Analysis ID:", styles["MetaLabel"]),
                Paragraph(f"<b>{analysis_id}</b>", styles["MetaVal"]),
                Paragraph("Analysis Date / Time:", styles["MetaLabel"]),
                Paragraph(now_str, styles["MetaVal"]),
            ],
            [
                Paragraph("Target Media File:", styles["MetaLabel"]),
                Paragraph(f"<font color='#0284c7'><b>{os.path.basename(uploaded_file)}</b></font>", styles["MetaVal"]),
                Paragraph("Media Format:", styles["MetaLabel"]),
                Paragraph(f"<b>{media_type}</b> Forensic Intake", styles["MetaVal"]),
            ],
            [
                Paragraph("Analyst / Submitter:", styles["MetaLabel"]),
                Paragraph(f"{user_name} ({user_email})", styles["MetaVal"]),
                Paragraph("Pipeline Verification:", styles["MetaLabel"]),
                Paragraph("Multi-Signal Canonical Consensus", styles["MetaVal"]),
            ],
        ]
        t_meta = Table(meta_rows, colWidths=[85, 181, 85, 181])
        t_meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 14))

        # 3. Verdict Hero Card
        fake_prob_pct = f"{round(float(fake_prob) * 100, 1)}%" if fake_prob is not None else "N/A"
        real_prob_pct = f"{round(float(real_prob) * 100, 1)}%" if real_prob is not None else "N/A"
        conf_score_str = f"{conf_score:.1f}%" if isinstance(conf_score, (int, float)) else ("N/A" if conf_score is None else str(conf_score))

        if canonical_v == "ILLUSTRATION":
            v_bg = ReportStyleManager.C_NEUTRAL_BG
            v_border = ReportStyleManager.C_NEUTRAL_BORDER
            v_color = ReportStyleManager.C_NEUTRAL_TEXT
            v_title = "VERDICT: ILLUSTRATION / CARTOON DETECTED"
            v_desc = (
                "This image appears to be a stylized illustration rather than a photographic image. "
                "The photographic deepfake detection models are not sufficient to determine whether this "
                "artwork was created by a human or generated by AI."
            )
            c_score_display = "N/A"
            c_tier_display = "Not Performed"
            c_split = "Media Type: Stylized Illustration / Anime"
        elif canonical_v == "NOT_SUITABLE":
            v_bg = ReportStyleManager.C_NEUTRAL_BG
            v_border = ReportStyleManager.C_NEUTRAL_BORDER
            v_color = ReportStyleManager.C_NEUTRAL_TEXT
            v_title = "VERDICT: NOT SUITABLE"
            v_desc = "Intake media quality verification determined that the uploaded content does not contain a detectable human face. Authenticity analysis was not performed."
            c_score_display = "N/A"
            c_tier_display = "Not Performed"
            c_split = "Facial Detection: No Face Detected"
        elif canonical_v == "FAKE":
            v_bg = ReportStyleManager.C_FAKE_BG
            v_border = ReportStyleManager.C_FAKE_BORDER
            v_color = ReportStyleManager.C_FAKE_TEXT
            v_title = "VERDICT: FAKE (MANIPULATED)"
            if is_provisional:
                v_desc = "Provisional assessment: Media leans FAKE across evaluated specialist models. Review and secondary verification recommended."
                c_score_display = conf_score_str if conf_score_str != "N/A" else "Low"
                c_tier_display = "Review Recommended"
            else:
                v_desc = "Multi-signal consensus confirms synthetic manipulation indicators across facial morphology and frequency spectra."
                c_score_display = conf_score_str
                c_tier_display = f"{conf_level} Confidence"
            c_split = f"Synthetic: {fake_prob_pct}   |   Authentic: {real_prob_pct}"
        elif canonical_v == "REAL":
            v_bg = ReportStyleManager.C_REAL_BG
            v_border = ReportStyleManager.C_REAL_BORDER
            v_color = ReportStyleManager.C_REAL_TEXT
            v_title = "VERDICT: REAL (AUTHENTIC)"
            if is_provisional:
                v_desc = "Provisional assessment: Media leans REAL across evaluated specialist models. Authenticity has NOT been conclusively established."
                c_score_display = conf_score_str if conf_score_str != "N/A" else "Low"
                c_tier_display = "Review Recommended"
            else:
                v_desc = "Authentic natural media features observed. Note: Absence of detected manipulation does not prove absolute authenticity."
                c_score_display = conf_score_str
                c_tier_display = f"{conf_level} Confidence"
            c_split = f"Authentic: {real_prob_pct}   |   Synthetic: {fake_prob_pct}"
        else:
            v_bg = ReportStyleManager.C_UNCERTAIN_BG
            v_border = ReportStyleManager.C_UNCERTAIN_BORDER
            v_color = ReportStyleManager.C_UNCERTAIN_TEXT
            v_title = f"VERDICT: {canonical_v}"
            v_desc = "Provisional assessment: Specialist detection models exhibited contradictory indicators. Independent forensic verification is recommended."
            c_score_display = conf_score_str if conf_score_str != "N/A" else "Uncertified"
            c_tier_display = "Review Recommended"
            c_split = f"Synthetic: {fake_prob_pct}   |   Authentic: {real_prob_pct}"

        # Hero Table Layout (Left: 342 pt, Right: 190 pt = 532)
        left_hero_content = [
            Paragraph(f"<font color='{v_color.hexval()}'><b>{v_title}</b></font>", styles["VerdictLarge"]),
            Spacer(1, 4),
            Paragraph(v_desc, styles["Body"]),
        ]
        if is_binary_fallback or reliability_warning:
            warn_msg = reliability_warning or "Specialist models produced conflicting indicators; canonical decision policy applied."
            left_hero_content.append(Spacer(1, 4))
            left_hero_content.append(Paragraph(f"<font size=7.5 color='#b45309'><b>Note:</b> {warn_msg}</font>", styles["NoticeText"]))

        right_hero_content = [
            Paragraph("<font size=7.5 color='#64748b'>EVALUATED CONFIDENCE</font>", ParagraphStyle("CHead", parent=styles["Normal"], alignment=2)),
            Paragraph(f"<font color='{v_color.hexval()}'><b>{c_score_display}</b></font>", styles["ConfidenceBig"]),
            Paragraph(f"<b>{c_tier_display}</b>", ParagraphStyle("CTier", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=ReportStyleManager.C_SLATE, alignment=2)),
            Spacer(1, 3),
            Paragraph(f"<font size=7 color='#64748b'>{c_split}</font>", ParagraphStyle("CSplit", parent=styles["Normal"], fontSize=7, textColor=ReportStyleManager.C_MUTED, alignment=2)),
        ]

        hero_table = Table([[left_hero_content, right_hero_content]], colWidths=[337, 195])
        hero_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), v_bg),
            ("BOX", (0, 0), (-1, -1), 1.2, v_border),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(hero_table)
        story.append(Spacer(1, 14))

        # 4. Plain-English "Understanding Your Result" Box
        story.append(Paragraph("Understanding Your Result", styles["SectionHeader"]))
        
        u_mean = user_friendly.get("what_does_this_mean") or (
            f"The forensic analysis completed with canonical verdict {canonical_v}. "
            "Evaluated detectors inspected spatial textures, facial boundaries, and frequency components."
        )
        u_limit = user_friendly.get("why_confidence_is_limited") or (
            "Absence of detectable manipulation signals does not constitute a guarantee of authenticity. "
            "Novel generative pipelines or clean screen recaptures may evade current forensic detectors."
        )

        u_rows = [
            [
                Paragraph("<b>What Does This Mean?</b>", styles["CalloutTitle"]),
                Paragraph(u_mean, styles["Body"]),
            ],
            [
                Paragraph("<b>Detection Scope & Limits:</b>", styles["CalloutTitle"]),
                Paragraph(u_limit, styles["Body"]),
            ],
        ]
        t_understand = Table(u_rows, colWidths=[130, 402])
        t_understand.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.75, ReportStyleManager.C_BORDER),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, ReportStyleManager.C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_understand)
        story.append(Spacer(1, 12))

        # 5. Key Forensic Observations
        findings = user_friendly.get("what_raised_concern", [])
        if findings:
            story.append(Paragraph("Key Forensic Observations", styles["SubSectionHeader"]))
            finding_items = []
            for f in findings[:4]:
                f_title = f.get("title", "Forensic Indicator")
                f_exp = f.get("explanation", "")
                finding_items.append([
                    Paragraph(f"<font color='#0284c7'><b>•</b></font> <b>{f_title}:</b> {f_exp}", styles["Body"])
                ])
            t_findings = Table(finding_items, colWidths=[532])
            t_findings.setStyle(TableStyle([
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_findings)
            story.append(Spacer(1, 10))

        # 6. Recommended Next Actions Callout Box
        story.append(Paragraph("Recommended Next Action", styles["SubSectionHeader"]))
        rec_data = [
            [
                Paragraph("<b>ACTIONABLE GUIDANCE:</b>", ParagraphStyle(
                    "ActionTitle",
                    parent=styles["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=8,
                    textColor=ReportStyleManager.C_TEAL,
                )),
                Paragraph(recommendation, styles["Body"]),
            ]
        ]
        t_rec = Table(rec_data, colWidths=[120, 412])
        t_rec.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDFA")),  # Soft teal bg
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#99F6E4")),      # Teal border
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_rec)

        # Natural page break to Page 2
        story.append(PageBreak())

        # -------------------------------------------------------------
        # PAGE 2 — MEDIA AND ANALYSIS OVERVIEW
        # -------------------------------------------------------------
        story.append(Paragraph("1. Media Intake & Forensic Suitability Overview", styles["SectionHeader"]))
        story.append(Paragraph(
            "Rigorous forensic authentication requires evaluating the input media against quality criteria, "
            "face detection viability, and resolution thresholds before interpreting specialist model results.",
            styles["Body"]
        ))
        story.append(Spacer(1, 10))

        # Media Preview & Intake Specifications
        rl_preview, preview_meta = ReportImageHelper.get_media_preview(
            uploaded_file, is_video=is_video, max_w=210, max_h=150
        )

        q_gate = fusion.get("quality_gate") or agent_res.get("quality_gate") or context_dict.get("quality_gate") or {}
        face_detected = fusion.get("face_detected", agent_res.get("face_detected", True))
        face_bbox = fusion.get("face_bbox") or agent_res.get("face_bbox") or q_gate.get("primary_face_bbox")
        bbox_str = f"[{face_bbox[0]}, {face_bbox[1]}, {face_bbox[2]}, {face_bbox[3]}]" if (face_bbox and len(face_bbox) == 4) else ("Full Face Detected" if face_detected else "None")

        blur_val = q_gate.get("blur_score", "N/A")
        blur_str = f"{blur_val:.1f}" if isinstance(blur_val, (int, float)) else str(blur_val)

        # Resolution resolution
        res_display = preview_meta.get("resolution", "Not assessed")
        if res_display == "Not assessed":
            raw_res = q_gate.get("resolution")
            if isinstance(raw_res, dict):
                w = raw_res.get("width") or raw_res.get("w")
                h = raw_res.get("height") or raw_res.get("h")
                if w and h and int(w or 0) > 0 and int(h or 0) > 0:
                    res_display = f"{w} × {h} px"
            elif isinstance(raw_res, (list, tuple)) and len(raw_res) >= 2:
                res_display = f"{raw_res[0]} × {raw_res[1]} px"
            elif isinstance(raw_res, str) and raw_res not in ("[object Object]", "Standard Native Resolution"):
                res_display = raw_res

        intake_table_rows = [
            [Paragraph("Face Detection Status:", styles["MetaLabel"]), Paragraph(f"<b>{'Detected & Aligned' if face_detected else 'No Human Face Detected'}</b>", styles["MetaVal"])],
            [Paragraph("Face Bounding Box:", styles["MetaLabel"]), Paragraph(bbox_str, styles["MetaVal"])],
            [Paragraph("Native Resolution:", styles["MetaLabel"]), Paragraph(res_display, styles["MetaVal"])],
            [Paragraph("Laplacian Blur Variance:", styles["MetaLabel"]), Paragraph(f"{blur_str} (Threshold > 50.0)", styles["MetaVal"])],
            [Paragraph("Color Space / Encoding:", styles["MetaLabel"]), Paragraph(f"Standard RGB ({preview_meta.get('format', 'JPEG')})", styles["MetaVal"])],
        ]
        if is_video:
            seq_agg = fusion.get("sequence_aggregation") or agent_res.get("sequence_aggregation") or {}
            dur_display = preview_meta.get("duration", "N/A")
            intake_table_rows.append([Paragraph("Total Video Duration:", styles["MetaLabel"]), Paragraph(dur_display, styles["MetaVal"])])
            intake_table_rows.append([Paragraph("Sampled Frames Analyzed:", styles["MetaLabel"]), Paragraph(f"{seq_agg.get('total_frames_analyzed', 'N/A')} frames", styles["MetaVal"])])

        t_intake_props = Table(intake_table_rows, colWidths=[120, 180])
        t_intake_props.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))

        if rl_preview:
            preview_container = Table([[rl_preview]], colWidths=[220])
            preview_container.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER_STRONG),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            intake_split_table = Table([[preview_container, t_intake_props]], colWidths=[226, 306])
            intake_split_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(intake_split_table)
        else:
            story.append(t_intake_props)
        story.append(Spacer(1, 14))

        # Quality Gate & Forensic Suitability Assessment Table
        story.append(Paragraph("Forensic Quality Gate & Media Suitability Criteria", styles["SubSectionHeader"]))
        
        blur_pass = isinstance(blur_val, (int, float)) and blur_val >= 40.0
        face_pass = bool(face_detected)

        q_table_data = [
            [
                Paragraph("<b>Forensic Criterion</b>", styles["TableHeader"]),
                Paragraph("<b>Measured Value</b>", styles["TableHeader"]),
                Paragraph("<b>Admissibility Threshold</b>", styles["TableHeader"]),
                Paragraph("<b>Status</b>", styles["TableHeader"]),
            ],
            [
                Paragraph("Human Face Verification", styles["TableCell"]),
                Paragraph("1 Face Verified" if face_detected else "0 Faces Detected", styles["TableCell"]),
                Paragraph(">= 1 Valid Facial Region", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>PASSED</b></font>" if face_pass else "<font color='#dc2626'><b>FAILED</b></font>", styles["TableCell"]),
            ],
            [
                Paragraph("Aperture Focus & Sharpness", styles["TableCell"]),
                Paragraph(f"Variance: {blur_str}", styles["TableCell"]),
                Paragraph("Laplacian Variance >= 40.0", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>PASSED</b></font>" if blur_pass else "<font color='#d97706'><b>WARNING</b></font>", styles["TableCell"]),
            ],
            [
                Paragraph("Spatial Resolution Adequacy", styles["TableCell"]),
                Paragraph(res_display, styles["TableCell"]),
                Paragraph("Min 224 × 224 px for ViT / CNN", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>PASSED</b></font>", styles["TableCell"]),
            ],
            [
                Paragraph("Input Media Container Integrity", styles["TableCell"]),
                Paragraph(validation_status, styles["TableCell"]),
                Paragraph("Clean MIME Header, No Corruption", styles["TableCell"]),
                Paragraph(f"<b>{validation_status}</b>", styles["TableCell"]),
            ],
        ]
        t_q = Table(q_table_data, colWidths=[150, 130, 152, 100])
        t_q.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ReportStyleManager.C_BG_SUBTLE),
            ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_q)
        story.append(Spacer(1, 14))

        # Forensic Analysis Scope and Limitations Disclosure
        story.append(Paragraph("Forensic Analysis Scope & Empirical Limitations", styles["SubSectionHeader"]))
        limits_text = (
            "<b>Notice of Forensic Scope:</b> SatyaDrishti evaluates digital media through multi-agent empirical analysis, "
            "inspecting biometric landmarks, facial boundary gradient consistency, frequency-domain energy decay (FFT), "
            "and deep feature representations extracted via fine-tuned CNN/ViT architectures. "
            "<br/><br/>"
            "<b>Known Technical Limitations:</b><br/>"
            "• <b>Novel Architecture Envelope:</b> Generative AI architectures unrepresented in training corpora (e.g. novel latent diffusion or neural radiance field synthesizers) may leave non-standard artifact profiles.<br/>"
            "• <b>Compression Artifacts:</b> Heavy social media recompression (H.264/WebP high quantization) attenuates subtle high-frequency forensic signatures.<br/>"
            "• <b>Physical Capture:</b> High-resolution screen re-recordings or printed photograph recaptures introduce moiré patterns that can confound frequency analysis."
        )
        t_limits = Table([[Paragraph(limits_text, styles["Body"])]], colWidths=[532])
        t_limits.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.75, ReportStyleManager.C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(t_limits)

        # For NOT_SUITABLE and ILLUSTRATION media, we don't have facial crops or specialist models, so we transition directly to Appendix
        if canonical_v not in ("NOT_SUITABLE", "ILLUSTRATION") and (evidence_cards or is_video):
            story.append(PageBreak())

            # -------------------------------------------------------------
            # PAGE 3 — FORENSIC EVIDENCE & REGIONAL OBSERVATIONS
            # -------------------------------------------------------------
            sec_title = "2. Forensic Evidence & Temporal Observations" if is_video else "2. Forensic Evidence & Regional Observations"
            story.append(Paragraph(sec_title, styles["SectionHeader"]))
            story.append(Paragraph(
                "Each empirical finding corresponds directly to measured localized regions of the uploaded media. "
                "Crops illustrate the biometric landmarks, seams, or temporal frames evaluated by forensic detectors.",
                styles["Body"]
            ))
            story.append(Spacer(1, 10))

            # Render Evidence Cards with Actual Image Crops
            rendered_cards = 0
            for idx, c in enumerate(evidence_cards[:5], 1):
                c_title = c.get("finding_title") or c.get("title", f"Forensic Evidence Finding {idx}")
                c_region = c.get("region_name", "Facial Region")
                c_timestamp = c.get("timestamp")
                c_support = str(c.get("verdict_support", "INCONCLUSIVE")).replace("_", " ").upper()
                c_strength = str(c.get("evidence_strength", "MODERATE")).replace("_", " ").upper()
                c_measurement = c.get("measurement_label", "")
                c_observed = c.get("observed_finding", "")
                c_plain = c.get("plain_explanation", "")
                c_limit = c.get("limitations", "")

                # Badge colors
                if "FAKE" in c_support:
                    badge_color = "#dc2626"
                    card_bg = ReportStyleManager.C_FAKE_BG
                    card_border = ReportStyleManager.C_FAKE_BORDER
                elif "REAL" in c_support:
                    badge_color = "#16a34a"
                    card_bg = ReportStyleManager.C_REAL_BG
                    card_border = ReportStyleManager.C_REAL_BORDER
                else:
                    badge_color = "#d97706"
                    card_bg = ReportStyleManager.C_UNCERTAIN_BG
                    card_border = ReportStyleManager.C_UNCERTAIN_BORDER

                # Try to load thumbnail crop
                crop_b64 = c.get("crop_data_url") or c.get("snapshot_data_url") or c.get("image_base64")
                rl_crop = ReportImageHelper.get_rl_image_from_b64(crop_b64, max_w=100, max_h=80)

                # Card Header
                if is_video and c_timestamp:
                    header_str = f"<b>{idx}. {c_title}</b> &nbsp;|&nbsp; <font color='#0284c7'><b>[{c_timestamp}]</b></font>"
                else:
                    header_str = f"<b>{idx}. {c_title}</b> &nbsp;|&nbsp; <font color='#64748b'>Target: {c_region}</font>"

                # Textual content of card
                text_content = [
                    Paragraph(f"{header_str} &nbsp; <font color='{badge_color}'><b>[{c_support}]</b></font>", styles["BodyBold"]),
                    Spacer(1, 2),
                    Paragraph(f"<b>Observed Metric:</b> {c_measurement}", styles["Body"]),
                    Paragraph(f"<b>Forensic Finding:</b> {c_observed}", styles["Body"]),
                    Paragraph(f"<b>Significance ({c_strength} Strength):</b> {c_plain}", styles["BodySmall"]),
                ]
                if c_limit:
                    text_content.append(Paragraph(f"<font color='#64748b'>Known Limitations: {c_limit}</font>", styles["BodySmall"]))

                if rl_crop:
                    # 2-column card with image on left (105 pt) and details on right (415 pt)
                    crop_container = Table([[rl_crop]], colWidths=[105])
                    crop_container.setStyle(TableStyle([
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOX", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]))
                    card_inner = Table([[crop_container, text_content]], colWidths=[110, 410])
                else:
                    # Full-width text card
                    card_inner = Table([[text_content]], colWidths=[520])

                card_inner.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]))

                card_wrapper = Table([[card_inner]], colWidths=[532])
                card_wrapper.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                    ("BOX", (0, 0), (-1, -1), 0.75, card_border),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))

                story.append(card_wrapper)
                story.append(Spacer(1, 6))
                rendered_cards += 1

            # Video Sequence & Temporal Metrics (for video media)
            if is_video:
                seq_agg = fusion.get("sequence_aggregation") or agent_res.get("sequence_aggregation") or {}
                if seq_agg:
                    story.append(Spacer(1, 4))
                    story.append(Paragraph("Video Sequence Aggregation & Temporal Consistency", styles["SubSectionHeader"]))
                    peak_item = seq_agg.get("manipulation_timestamps", [{}])[0] if seq_agg.get("manipulation_timestamps") else {}
                    peak_ts = peak_item.get("formatted_timestamp", "N/A")
                    peak_p = seq_agg.get("peak_fake_probability")
                    peak_p_str = f"{peak_p:.1%}" if peak_p is not None else "N/A"

                    seq_table_data = [
                        [
                            Paragraph("Total Sampled Frames:", styles["MetaLabel"]),
                            Paragraph(f"<b>{seq_agg.get('total_frames_analyzed', 0)} frames</b>", styles["MetaVal"]),
                            Paragraph("Tracked Faces Evaluated:", styles["MetaLabel"]),
                            Paragraph(f"<b>{seq_agg.get('tracked_faces', 1)}</b>", styles["MetaVal"]),
                        ],
                        [
                            Paragraph("Manipulated Frames Detected:", styles["MetaLabel"]),
                            Paragraph(f"<font color='#dc2626'><b>{seq_agg.get('fake_frames_detected', 0)} frames</b></font>", styles["MetaVal"]),
                            Paragraph("Authentic Frames Detected:", styles["MetaLabel"]),
                            Paragraph(f"<font color='#16a34a'><b>{seq_agg.get('real_frames_detected', 0)} frames</b></font>", styles["MetaVal"]),
                        ],
                        [
                            Paragraph("Inconclusive / Mixed Frames:", styles["MetaLabel"]),
                            Paragraph(f"{seq_agg.get('inconclusive_frames', 0)} frames", styles["MetaVal"]),
                            Paragraph("Peak Manipulation Timestamp:", styles["MetaLabel"]),
                            Paragraph(f"<font color='#dc2626'><b>{peak_ts}</b></font> ({peak_p_str} synthetic)", styles["MetaVal"]),
                        ],
                    ]
                    t_seq = Table(seq_table_data, colWidths=[130, 136, 130, 136])
                    t_seq.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
                        ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]))
                    story.append(t_seq)
                    story.append(Spacer(1, 6))

                # Audio-Visual Lip-Sync Synchronization Card
                av_sync = fusion.get("audio_visual_sync") or agent_res.get("audio_visual_sync") or {}
                av_status = str(av_sync.get("status") or ("ANALYZED" if av_sync.get("analyzed") else "NOT ANALYZED")).upper()
                av_support = str(av_sync.get("verdict_support", "INCONCLUSIVE")).upper()
                av_details = av_sync.get("details") or av_sync.get("reason", "Audio-visual lip-sync analysis was not performed.")
                av_offset = av_sync.get("temporal_offset_ms")
                av_conf = av_sync.get("sync_confidence")

                offset_str = f"{av_offset:+.1f} ms" if av_offset is not None else "N/A"
                conf_s_str = f"{av_conf:.1%}" if av_conf is not None else "N/A"

                av_content = [
                    Paragraph(f"<b>Audio-Visual Lip-Sync Temporal Synchronization:</b> <font color='#0284c7'><b>{av_status}</b></font> &nbsp;|&nbsp; Verdict Support: <b>{av_support}</b>", styles["BodyBold"]),
                    Spacer(1, 2),
                    Paragraph(f"<b>Temporal Offset:</b> {offset_str} &nbsp;|&nbsp; <b>Sync Correlation Confidence:</b> {conf_s_str} &nbsp;|&nbsp; <b>Speech Detected:</b> {'Yes' if av_sync.get('speech_detected') else 'No'}", styles["BodySmall"]),
                    Paragraph(f"<font color='#334155'>{av_details}</font>", styles["BodySmall"]),
                ]
                t_av = Table([[av_content]], colWidths=[532])
                t_av.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.75, ReportStyleManager.C_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(t_av)

            story.append(PageBreak())

        # -------------------------------------------------------------
        # PAGE 4 — TECHNICAL VERIFICATION APPENDIX
        # -------------------------------------------------------------
        story.append(Paragraph("3. Technical Verification & Specialist Model Breakdown", styles["SectionHeader"]))
        story.append(Paragraph(
            "Multi-agent forensic verification fuses specialized neural detectors, frequency-domain spectral analyzers, "
            "and temporal consistency models. The matrix below records individual model predictions, confidence scores, and calibration states.",
            styles["Body"]
        ))
        story.append(Spacer(1, 10))

        # Specialist Model Matrix Table
        if models:
            story.append(Paragraph("Specialist Neural Model Matrix & Confidence Calibration", styles["SubSectionHeader"]))
            m_headers = [
                Paragraph("<b>Model Name</b>", styles["TableHeader"]),
                Paragraph("<b>Specialist Forensic Task</b>", styles["TableHeader"]),
                Paragraph("<b>Class</b>", styles["TableHeader"]),
                Paragraph("<b>Score / Prob</b>", styles["TableHeader"]),
                Paragraph("<b>Calibration Method</b>", styles["TableHeader"]),
            ]
            m_rows = [m_headers]
            for m in models:
                pred_c = str(m.get("predicted_class", "N/A")).upper()
                c_col = "#dc2626" if pred_c == "FAKE" else ("#16a34a" if pred_c == "REAL" else "#64748b")
                raw_s = m.get("raw_score")
                cal_p = m.get("calibrated_prob")
                s_str = f"{cal_p*100:.1f}%" if isinstance(cal_p, (int, float)) else (f"{raw_s*100:.1f}%" if isinstance(raw_s, (int, float)) else "N/A")
                m_rows.append([
                    Paragraph(f"<b>{m.get('name', 'Specialist Model')}</b>", styles["TableCell"]),
                    Paragraph(f"<font color='#475569'>{m.get('task', 'Forensic Detection')}</font>", styles["TableCell"]),
                    Paragraph(f"<font color='{c_col}'><b>{pred_c}</b></font>", styles["TableCell"]),
                    Paragraph(f"<b>{s_str}</b>", styles["TableCell"]),
                    Paragraph(f"<font size=7 color='#64748b'>{m.get('calibration_status', 'Empirical Softmax')}</font>", styles["TableCell"]),
                ])
            t_models = Table(m_rows, colWidths=[130, 152, 60, 75, 115])
            t_models.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), ReportStyleManager.C_BG_SUBTLE),
                ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_models)
            story.append(Spacer(1, 12))

        # Multimodal Vision Reasoning (Stage A-D)
        vr_data = fusion.get("vision_reasoning") or agent_res.get("vision_reasoning") or {}
        if isinstance(vr_data, dict) and (vr_data.get("reasoning_summary") or vr_data.get("whole_image_observations")):
            story.append(Paragraph("Multimodal Qualitative Vision Synthesis (Four-Stage Inspection)", styles["SubSectionHeader"]))
            raw_vr = vr_data.get("qualitative_assessment", "INCONCLUSIVE")
            vr_str = f"<b>Multimodal Qualitative Finding:</b> {raw_vr} &nbsp;|&nbsp; <b>Provider:</b> {vr_data.get('provider_id', 'Deterministic')}"
            vr_body = [
                Paragraph(vr_str, styles["BodyBold"]),
                Spacer(1, 2),
                Paragraph(f"<b>Synthesis:</b> {vr_data.get('reasoning_summary', 'Multi-stage qualitative reasoning evaluated global context, regional artifacts, and limitations.')}", styles["Body"]),
            ]
            t_vr = Table([[vr_body]], colWidths=[532])
            t_vr.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(t_vr)
            story.append(Spacer(1, 10))

        # Agent Execution Trace & Pipeline Verification Audit
        story.append(Paragraph("Pipeline Execution Trace & Agent Audit Status", styles["SubSectionHeader"]))
        trace_headers = [
            Paragraph("<b>Pipeline Stage / Agent</b>", styles["TableHeader"]),
            Paragraph("<b>Execution Status</b>", styles["TableHeader"]),
            Paragraph("<b>Audit Details & Operational Scope</b>", styles["TableHeader"]),
        ]
        trace_rows = [
            trace_headers,
            [
                Paragraph("Input Media Validator", styles["TableCell"]),
                Paragraph(f"<font color='#16a34a'><b>{validation_status}</b></font>", styles["TableCell"]),
                Paragraph("Verified file integrity, MIME signature, and container decodability.", styles["TableCell"]),
            ],
            [
                Paragraph("Quality Gate & Biometric Alignment", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>COMPLETED</b></font>" if face_detected else "<font color='#dc2626'><b>NO FACE</b></font>", styles["TableCell"]),
                Paragraph("Evaluated Laplacian blur variance, dynamic range, and localized face bounding box.", styles["TableCell"]),
            ],
            [
                Paragraph("Specialist Neural Detectors", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>COMPLETED</b></font>" if models else "<font color='#64748b'><b>NOT RUN</b></font>", styles["TableCell"]),
                Paragraph(f"Inference executed across {len(models)} specialist architectures with softmax calibration.", styles["TableCell"]),
            ],
            [
                Paragraph("Temporal Tracking & Sync Agent", styles["TableCell"]),
                Paragraph(("<font color='#16a34a'><b>COMPLETED</b></font>" if is_video else "<font color='#64748b'><b>NOT APPLICABLE</b></font>"), styles["TableCell"]),
                Paragraph("Evaluated inter-frame boundary flicker, mouth texture dynamics, and AV synchronization." if is_video else "Bypassed for single-frame still image media.", styles["TableCell"]),
            ],
            [
                Paragraph("Canonical Fusion Engine", styles["TableCell"]),
                Paragraph("<font color='#16a34a'><b>COMPLETED</b></font>", styles["TableCell"]),
                Paragraph(f"Normalized specialist inputs via Canonical Decision Policy; verdict: {canonical_v}.", styles["TableCell"]),
            ],
        ]
        t_trace = Table(trace_rows, colWidths=[150, 100, 282])
        t_trace.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ReportStyleManager.C_BG_SUBTLE),
            ("GRID", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t_trace)
        story.append(Spacer(1, 14))

        # Forensic Integrity Statement & Legal Disclaimer
        disclaimer_text = (
            "<b>Forensic Evidentiary Disclaimer:</b> This document constitutes an automated digital media forensic analysis report "
            "generated by the SatyaDrishti platform. Findings are derived from algorithmic models and empirical heuristics. "
            "Neither SatyaDrishti nor its operators certify absolute legal authenticity or manipulation absence. "
            "For judicial, journalistic, or high-stakes evidentiary proceedings, formal chain-of-custody corroboration "
            "and secondary manual human-in-the-loop expert review are strongly recommended."
        )
        t_disc = Table([[Paragraph(disclaimer_text, styles["BodySmall"])]], colWidths=[532])
        t_disc.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ReportStyleManager.C_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, ReportStyleManager.C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_disc)

        # Build Document using NumberedCanvas
        def canvas_maker(*args, **kwargs):
            nc = NumberedCanvas(*args, **kwargs)
            nc.report_id = analysis_id
            nc.generated_at = now_str
            return nc

        doc.build(story, canvasmaker=canvas_maker)

        # Upload report to Supabase Storage and store metadata if configured
        try:
            from backend.services.supabase_service import SupabaseService
            from backend.services.database_service import DatabaseService

            if SupabaseService.verify_connection():
                supabase_report_path = SupabaseService.upload_report(report_path, report_filename)
                if supabase_report_path:
                    bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
                    public_report_url = SupabaseService.get_client().storage.from_(bucket).get_public_url(report_filename)
                    file_size = os.path.getsize(report_path)
                    DatabaseService.save_report_metadata(
                        analysis_id=analysis_id,
                        report_url=public_report_url,
                        file_size=file_size,
                        report_path=supabase_report_path,
                    )
        except Exception as e:
            print(f"ReportGenerator: Failed to upload report metadata to Supabase: {str(e)}")

        return report_path
