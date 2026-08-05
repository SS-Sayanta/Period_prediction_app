"""
report_generator.py — Medical PDF Report Generator for FemCare AI.
Generates clinical-grade PDF summary reports for gynecologist consultation using ReportLab.
"""

from __future__ import annotations
import io
from datetime import date, datetime
from typing import Dict, Any, Optional
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether


def generate_medical_pdf(
    user_name: str,
    user_id: str,
    user_hist_df: pd.DataFrame,
    symptoms_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    current_res: Any,
    anomaly_data: Dict[str, Any]
) -> bytes:
    """
    Compiles patient cycle history, symptom averages, ML accuracy metrics, and AI insights into a PDF report.
    Returns the binary content (bytes) ready for Streamlit st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    primary_color = colors.HexColor("#E11D48")
    dark_text = colors.HexColor("#0F172A")
    muted_text = colors.HexColor("#475569")
    bg_light = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#CBD5E1")
    accent_teal = colors.HexColor("#0D9488")
    accent_purple = colors.HexColor("#7C3AED")

    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=muted_text,
        spaceAfter=12
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=dark_text,
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=dark_text
    )

    disclaimer_style = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748B")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("🌸 FemCare AI — Clinical Health Report", title_style))
    elements.append(Paragraph(f"CONFIDENTIAL MEDICAL SUMMARY • GENERATED ON {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

    # 2. Patient Demographics & Session Card Table
    p_start_str = current_res.next_period_start.strftime("%b %d, %Y") if hasattr(current_res, "next_period_start") else "N/A"
    ov_str = current_res.ovulation_date.strftime("%b %d, %Y") if hasattr(current_res, "ovulation_date") else "N/A"
    pred_days = f"{current_res.predicted_cycle_days:.1f} Days" if hasattr(current_res, "predicted_cycle_days") else "N/A"

    demo_data = [
        [
            Paragraph("<b>Patient Name:</b> " + str(user_name), body_style),
            Paragraph("<b>Patient ID:</b> " + str(user_id[:12]) + "...", body_style),
        ],
        [
            Paragraph("<b>Average Cycle Duration:</b> " + str(anomaly_data.get("mean_length", 28.0)) + " Days", body_style),
            Paragraph("<b>Cycle Variability (Std Dev):</b> " + str(anomaly_data.get("std_length", 0.0)) + " Days", body_style),
        ],
        [
            Paragraph("<b>Predicted Next Period:</b> " + p_start_str, body_style),
            Paragraph("<b>Estimated Ovulation:</b> " + ov_str, body_style),
        ]
    ]

    t_demo = Table(demo_data, colWidths=[270, 270])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_demo)
    elements.append(Spacer(1, 12))

    # 3. AI Anomaly & Clinical Insights Section
    elements.append(Paragraph("📋 Algorithmic Health & Anomaly Evaluation", h2_style))
    status_lvl = anomaly_data.get("status_level", "normal").upper()
    status_color = "#059669" if status_lvl == "NORMAL" else ("#D97706" if status_lvl == "WARNING" else "#DC2626")

    summary_html = f"<b>Overall Status:</b> <font color='{status_color}'><b>{status_lvl}</b></font> — {anomaly_data.get('summary_text', '')}"
    elements.append(Paragraph(summary_html, body_style))
    elements.append(Spacer(1, 6))

    insights_list = anomaly_data.get("insights", [])
    if insights_list:
        for ins in insights_list:
            ins_text = f"• <b>{ins['title']}:</b> {ins['desc']}"
            elements.append(Paragraph(ins_text, body_style))
            elements.append(Spacer(1, 4))

    elements.append(Spacer(1, 10))

    # 4. Historical Predictions Table (Past Cycles)
    elements.append(Paragraph("📊 Historical Predictions Log (`user_history.csv`)", h2_style))
    hist_table_data = [["Timestamp", "User ID", "Last Period Date", "Predicted Days"]]

    if not user_hist_df.empty:
        # Take last 8 rows
        recent_hist = user_hist_df.tail(8)
        for _, row in recent_hist.iterrows():
            ts = str(row.get("timestamp", ""))[:16]
            uid = str(row.get("user_id", ""))[:8]
            lpd = str(row.get("last_period_date", ""))
            pdays = str(row.get("predicted_cycle_days", ""))
            hist_table_data.append([ts, uid, lpd, pdays])
    else:
        hist_table_data.append(["No records logged", "-", "-", "-"])

    t_hist = Table(hist_table_data, colWidths=[140, 100, 150, 150])
    t_hist.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
    ]))
    elements.append(t_hist)
    elements.append(Spacer(1, 12))

    # 5. Symptom Averages Summary Table
    elements.append(Paragraph("📝 Recorded Symptom Logs Summary (`symptoms_log.csv`)", h2_style))
    symptom_table_data = [["Log Date", "Flow Level", "Cramps (0-5)", "Mood (1-5)", "Fatigue (0-5)", "Notes"]]

    if not symptoms_df.empty:
        recent_symptoms = symptoms_df.tail(6)
        for _, row in recent_symptoms.iterrows():
            ldate = str(row.get("log_date", ""))
            flow = str(row.get("flow_level", "None"))
            cramps = str(row.get("cramps", 0))
            mood = str(row.get("mood", 3))
            fatigue = str(row.get("fatigue", 0))
            notes = str(row.get("notes", ""))[:25]
            symptom_table_data.append([ldate, flow, cramps, mood, fatigue, notes])
    else:
        symptom_table_data.append(["No symptoms logged", "-", "-", "-", "-", "-"])

    t_sym = Table(symptom_table_data, colWidths=[90, 80, 80, 80, 80, 130])
    t_sym.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dark_text),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
    ]))
    elements.append(t_sym)
    elements.append(Spacer(1, 14))

    # 6. Physician Verification Box & Notes
    doc_box_data = [
        [Paragraph("<b>Physician Observations & Gynecologist Notes:</b>", body_style)],
        [Paragraph("<br/><br/><br/><b>Signature:</b> ___________________________ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Date:</b> ______________", body_style)]
    ]
    t_doc = Table(doc_box_data, colWidths=[540])
    t_doc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(t_doc)
    elements.append(Spacer(1, 12))

    # 7. Medical Disclaimer Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceAfter=6))
    elements.append(Paragraph(anomaly_data.get("disclaimer", ""), disclaimer_style))

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
