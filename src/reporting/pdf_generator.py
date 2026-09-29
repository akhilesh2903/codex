from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import time


class ReportGenerator:
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, result_json, image_path=None, gradcam_path=None, overlay_path=None, patient_name="", phone_number=""):
        image_id = result_json.get("image_id", "UNKNOWN")
        out_pdf = os.path.join(self.output_dir, f"report_{image_id}.pdf")

        doc = SimpleDocTemplate(out_pdf, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", fontSize=16, fontName="Helvetica-Bold",
                                     alignment=TA_CENTER, spaceAfter=6)
        section_style = ParagraphStyle("section", fontSize=12, fontName="Helvetica-Bold",
                                       spaceAfter=4, spaceBefore=12)
        body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica", spaceAfter=3)
        disclaimer_style = ParagraphStyle("disclaimer", fontSize=8, fontName="Helvetica-Oblique",
                                          textColor=colors.grey, alignment=TA_CENTER)

        story = []

        # ── Header ──────────────────────────────────────────────────────────────
        story.append(Paragraph("DIABETIC RETINOPATHY SCREENING REPORT", title_style))
        story.append(Paragraph("Explainable AI-Based Telemedicine System", styles["Normal"]))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 0.1*inch))

        # ── Metadata ────────────────────────────────────────────────────────────
        meta_data = [
            ["Patient Name", patient_name or "N/A"],
            ["Phone Number", phone_number or "N/A"],
            ["Image ID", str(image_id)],
            ["Report Generated", time.strftime("%Y-%m-%d %H:%M:%S")],
            ["Model Version", "EfficientNet-B0 v1.0"],
        ]
        meta_table = Table(meta_data, colWidths=[2*inch, 4.5*inch])
        meta_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.15*inch))

        # ── 1. Image Quality ─────────────────────────────────────────────────────
        story.append(Paragraph("1. Image Quality Assessment", section_style))
        q = result_json.get("quality", {})
        status = q.get("status", "N/A")
        status_color = colors.green if status == "GOOD" else (colors.orange if status == "BORDERLINE" else colors.red)
        quality_data = [
            ["Quality Status", status],
            ["Quality Score", f"{q.get('quality_score', 0):.3f}"],
            ["Blur Score", f"{q.get('blur_score', 0):.1f}"],
            ["Brightness", f"{q.get('brightness', 0):.1f}"],
            ["Field of View", f"{q.get('field_of_view_score', 0):.2f}"],
        ]
        q_table = Table(quality_data, colWidths=[2.5*inch, 4*inch])
        q_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TEXTCOLOR", (1, 0), (1, 0), status_color),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(q_table)

        # ── 2. DR Classification ─────────────────────────────────────────────────
        story.append(Paragraph("2. DR Severity Classification", section_style))
        pred = result_json.get("prediction", {})
        grade = pred.get("grade", "N/A")
        label = pred.get("label", "N/A")
        conf = pred.get("confidence", 0)
        referable = "YES ⚠" if pred.get("referable") else "NO ✓"
        conf_tier = "HIGH" if conf >= 0.8 else ("MODERATE" if conf >= 0.6 else "LOW")

        clf_data = [
            ["DR Grade", f"Level {grade} – {label}"],
            ["Model Confidence", f"{conf*100:.1f}%  ({conf_tier})"],
            ["Referable DR", referable],
            ["Clinical Action", "REFER TO OPHTHALMOLOGIST" if pred.get("referable") else "ROUTINE FOLLOW-UP"],
        ]
        clf_table = Table(clf_data, colWidths=[2.5*inch, 4*inch])
        clf_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TEXTCOLOR", (1, 2), (1, 2), colors.red if pred.get("referable") else colors.green),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(clf_table)

        # ── Processing time ──────────────────────────────────────────────────────
        proc_time = result_json.get("processing_time_ms", None)
        if proc_time is not None:
            story.append(Paragraph(f"Processing Time: {proc_time:.0f} ms", body_style))

        # ── 3. Lesion Analysis ────────────────────────────────────────────────────
        story.append(Paragraph("3. Detected Lesion Findings", section_style))
        story.append(Paragraph(
            "Note: Lesion detection uses OpenCV heuristics. A trained DL model would improve accuracy.",
            ParagraphStyle("note", fontSize=8, fontName="Helvetica-Oblique", textColor=colors.grey)
        ))
        lesions = result_json.get("lesions", {})
        lesion_data = [["Lesion Type", "Detected", "Count", "Confidence"]]
        for ltype in ["microaneurysms", "hemorrhages", "exudates"]:
            d = lesions.get(ltype, {})
            lesion_data.append([
                ltype.replace("_", " ").title(),
                "YES" if d.get("detected") else "NO",
                str(d.get("count", 0)),
                f"{d.get('confidence', 0)*100:.0f}%"
            ])
        neo = lesions.get("neovascularization", {})
        lesion_data.append(["Neovascularization", "N/A", "N/A", "Requires DL model"])

        l_table = Table(lesion_data, colWidths=[2.2*inch, 1.2*inch, 1.2*inch, 1.9*inch])
        l_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(l_table)

        # ── 4. Class Probabilities ────────────────────────────────────────────────
        story.append(Paragraph("4. Class Probability Breakdown", section_style))
        xai = result_json.get("explainability", {})
        class_probs = xai.get("class_probabilities", {})
        if class_probs:
            prob_data = [["DR Level", "Probability"]]
            for label_name, prob in class_probs.items():
                prob_data.append([label_name, f"{prob*100:.1f}%"])
            p_table = Table(prob_data, colWidths=[3.5*inch, 3*inch])
            p_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(p_table)

        # ── 5. Explainability Images ──────────────────────────────────────────────
        img_width = 2.8 * inch
        img_height = 2.0 * inch
        image_panels = []

        def try_image(path, caption):
            if path and os.path.exists(path):
                try:
                    img = RLImage(path, width=img_width, height=img_height)
                    return [img, Paragraph(caption, ParagraphStyle("cap", fontSize=7, alignment=TA_CENTER))]
                except Exception:
                    pass
            return [Paragraph(f"[{caption}: not available]",
                              ParagraphStyle("cap", fontSize=7, alignment=TA_CENTER, textColor=colors.grey))]

        panels = []
        if image_path:
            panels.extend(try_image(image_path, "Original Fundus"))
        if gradcam_path:
            panels.extend(try_image(gradcam_path, "Grad-CAM Heatmap"))
        if overlay_path:
            panels.extend(try_image(overlay_path, "Lesion Overlay"))

        if panels:
            story.append(Paragraph("5. Visual Evidence", section_style))
            # Group into rows of 2
            row_items = []
            for i, item in enumerate(panels):
                row_items.append(item)
                if (i + 1) % 2 == 0:
                    story.append(Table([row_items], colWidths=[img_width + 0.3*inch] * 2))
                    story.append(Spacer(1, 0.05*inch))
                    row_items = []
            if row_items:
                story.append(Table([row_items + [""]], colWidths=[img_width + 0.3*inch] * 2))

        # ── 6. Clinical Review Section ────────────────────────────────────────────
        story.append(Spacer(1, 0.2*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Paragraph("6. Clinical Review", section_style))
        review_data = [
            ["Ophthalmologist Name", "________________________"],
            ["Review Decision", "☐ CONFIRM   ☐ REJECT   ☐ RECAPTURE"],
            ["Clinical Notes", "________________________"],
            ["Date / Signature", "________________________"],
        ]
        r_table = Table(review_data, colWidths=[2.5*inch, 4*inch])
        r_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(r_table)

        # ── Disclaimer ────────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.3*inch))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Paragraph(
            "⚠ This AI output is intended for screening support and clinician review only. "
            "It is NOT an autonomous clinical diagnosis and does NOT replace ophthalmologist assessment. "
            "All screening decisions must be validated by a qualified clinician.",
            disclaimer_style
        ))

        doc.build(story)
        return out_pdf
