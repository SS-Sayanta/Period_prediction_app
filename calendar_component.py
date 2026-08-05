"""
calendar_component.py — Custom HTML/CSS Grid Calendar Component for FemCare AI.
Renders clean medical-grade calendar cards with phase indicators and symptom tags.
"""

from __future__ import annotations
import calendar
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List


def _clean_html(html_str: str) -> str:
    """
    Strips leading and trailing whitespace from each line in the HTML string.
    This prevents Streamlit's Markdown parser from treating lines with 4+ spaces of indentation as code blocks.
    """
    return "\n".join(line.strip() for line in html_str.splitlines() if line.strip())


def render_html_calendar(
    year: int,
    month: int,
    prediction_result: Any,
    logged_symptom_dates: Optional[List[str]] = None,
    theme: str = "dark"
) -> str:
    """
    Generates a pure, validated HTML/CSS grid string for month calendar view.
    Designed for rendering via st.markdown(html, unsafe_allow_html=True).
    """
    if logged_symptom_dates is None:
        logged_symptom_dates = []

    cal = calendar.Calendar(firstweekday=6)  # Sunday start
    month_days = cal.monthdatescalendar(year, month)
    month_name = calendar.month_name[month]

    # Pre-calculate phase date ranges if prediction result is present
    p_start = getattr(prediction_result, 'next_period_start', None)
    p_end = getattr(prediction_result, 'next_period_end', None)
    ovulation = getattr(prediction_result, 'ovulation_date', None)
    f_start = getattr(prediction_result, 'fertile_window_start', None)
    f_end = getattr(prediction_result, 'fertile_window_end', None)

    # Convert str dates to date objects if needed
    if isinstance(p_start, str):
        p_start = datetime.strptime(p_start, "%Y-%m-%d").date()
    if isinstance(p_end, str):
        p_end = datetime.strptime(p_end, "%Y-%m-%d").date()
    if isinstance(ovulation, str):
        ovulation = datetime.strptime(ovulation, "%Y-%m-%d").date()
    if isinstance(f_start, str):
        f_start = datetime.strptime(f_start, "%Y-%m-%d").date()
    if isinstance(f_end, str):
        f_end = datetime.strptime(f_end, "%Y-%m-%d").date()

    # Color palette matching theme
    is_dark = (theme == "dark")
    card_bg = "#1E293B" if is_dark else "#FFFFFF"
    border_color = "#334155" if is_dark else "#E2E8F0"
    text_color = "#F8FAFC" if is_dark else "#0F172A"
    muted_text = "#94A3B8" if is_dark else "#64748B"
    header_bg = "rgba(255, 255, 255, 0.03)" if is_dark else "#F8FAFC"
    cell_default_bg = "rgba(255, 255, 255, 0.02)" if is_dark else "#F8FAFC"
    cell_muted_bg = "rgba(0, 0, 0, 0.15)" if is_dark else "#F1F5F9"

    today = date.today()
    symptoms_in_month = 0

    lines = []
    lines.append(f'<div style="background:{card_bg}; border:1px solid {border_color}; border-radius:20px; padding:1.5rem; font-family:Inter, sans-serif; box-shadow:0 12px 30px rgba(0,0,0,0.25);">')
    
    # Header Row
    lines.append(f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.4rem; flex-wrap:wrap; gap:12px;">')
    lines.append(f'<div>')
    lines.append(f'<h3 style="margin:0; color:{text_color}; font-size:1.4rem; font-weight:700; letter-spacing:-0.01em;">📅 {month_name} {year}</h3>')
    lines.append(f'<div style="font-size:0.78rem; color:{muted_text}; margin-top:2px;">Interactive Cycle & Symptom Tracking Grid</div>')
    lines.append(f'</div>')
    
    # Visual Legend Pills
    lines.append(f'<div style="display:flex; gap:8px; font-size:0.76rem; flex-wrap:wrap;">')
    lines.append(f'<span style="display:inline-flex; align-items:center; gap:4px; background:rgba(244,114,182,0.18); border:1px solid rgba(244,114,182,0.4); color:#F472B6; padding:4px 10px; border-radius:8px; font-weight:600;">🩸 Predicted Period</span>')
    lines.append(f'<span style="display:inline-flex; align-items:center; gap:4px; background:rgba(45,212,191,0.18); border:1px solid rgba(45,212,191,0.4); color:#2DD4BF; padding:4px 10px; border-radius:8px; font-weight:600;">🌿 Fertile Window</span>')
    lines.append(f'<span style="display:inline-flex; align-items:center; gap:4px; background:rgba(167,139,250,0.22); border:1px solid rgba(167,139,250,0.4); color:#A78BFA; padding:4px 10px; border-radius:8px; font-weight:600;">⭐ Ovulation Peak</span>')
    lines.append(f'<span style="display:inline-flex; align-items:center; gap:4px; background:rgba(56,189,248,0.18); border:1px solid rgba(56,189,248,0.4); color:#38BDF8; padding:4px 10px; border-radius:8px; font-weight:600;">📝 Symptom Logged</span>')
    lines.append(f'</div>')
    lines.append(f'</div>')

    # Calendar Table Grid
    lines.append(f'<table style="width:100%; border-collapse:separate; border-spacing:8px; text-align:center; color:{text_color}; table-layout:fixed;">')
    lines.append(f'<thead>')
    lines.append(f'<tr style="color:{muted_text}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;">')
    for day_head in ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]:
        lines.append(f'<th style="padding:8px 4px; background:{header_bg}; border-radius:8px;">{day_head}</th>')
    lines.append(f'</tr>')
    lines.append(f'</thead>')
    lines.append(f'<tbody>')

    for week in month_days:
        lines.append(f'<tr>')
        for d in week:
            is_current_month = (d.month == month)
            day_num = d.day
            d_str = d.strftime("%Y-%m-%d")

            cell_bg = cell_default_bg if is_current_month else cell_muted_bg
            cell_border = f"1px solid {border_color}" if is_current_month else "1px dashed transparent"
            cell_color = text_color if is_current_month else muted_text
            badge_content = ""
            cell_title = f"{d.strftime('%b %d, %Y')}"
            opacity = "1.0" if is_current_month else "0.35"

            if is_current_month:
                is_today = (d == today)

                # Status flags
                is_period = bool(p_start and p_end and p_start <= d <= p_end)
                is_ovulation = bool(ovulation and d == ovulation)
                is_fertile = bool(f_start and f_end and f_start <= d <= f_end)
                has_symptom = any(d_str in s_date for s_date in logged_symptom_dates)

                if has_symptom:
                    symptoms_in_month += 1

                if is_today:
                    cell_border = "2px solid #38BDF8"
                    cell_title += " (Today)"

                if is_period:
                    cell_bg = "rgba(244, 114, 182, 0.25)"
                    cell_border = "1px solid #F472B6"
                    badge_content += '<span style="color:#F472B6; font-size:0.72rem; font-weight:700; display:inline-block; margin-top:2px;">🩸 Period</span>'
                elif is_ovulation:
                    cell_bg = "rgba(167, 139, 250, 0.32)"
                    cell_border = "1px solid #A78BFA"
                    badge_content += '<span style="color:#A78BFA; font-size:0.72rem; font-weight:700; display:inline-block; margin-top:2px;">⭐ Ovulation</span>'
                elif is_fertile:
                    cell_bg = "rgba(45, 212, 191, 0.22)"
                    cell_border = "1px solid #2DD4BF"
                    badge_content += '<span style="color:#2DD4BF; font-size:0.72rem; font-weight:600; display:inline-block; margin-top:2px;">🌿 Fertile</span>'

                if has_symptom:
                    badge_content += '<span style="color:#38BDF8; font-size:0.7rem; font-weight:600; display:inline-block; margin-top:2px; margin-left:3px;">📝 Logged</span>'

                if is_today and not (is_period or is_ovulation or is_fertile):
                    badge_content += '<span style="color:#38BDF8; font-size:0.7rem; font-weight:700; display:inline-block; margin-top:2px;">TODAY</span>'

            today_dot = '<span style="width:6px; height:6px; background:#38BDF8; border-radius:50%; display:inline-block;"></span>' if (is_current_month and d == today) else ''
            day_font_size = "1.02rem" if (is_current_month and d == today) else "0.9rem"

            lines.append(
                f'<td title="{cell_title}" style="padding:10px 4px; border:{cell_border}; background:{cell_bg}; color:{cell_color}; '
                f'border-radius:12px; vertical-align:top; height:68px; opacity:{opacity}; transition:all 0.2s ease;">'
                f'<div style="font-weight:700; font-size:{day_font_size}; display:flex; justify-content:space-between; align-items:center; padding:0 4px;">'
                f'<span>{day_num}</span>'
                f'{today_dot}'
                f'</div>'
                f'<div style="margin-top:4px; min-height:22px; display:flex; flex-direction:column; align-items:center; justify-content:center;">{badge_content}</div>'
                f'</td>'
            )
        lines.append(f'</tr>')

    lines.append(f'</tbody>')
    lines.append(f'</table>')

    # Footer Card Summary
    p_range_str = f"{p_start.strftime('%b %d')} - {p_end.strftime('%b %d')}" if (p_start and p_end) else "N/A"
    ov_str = ovulation.strftime('%b %d, %Y') if ovulation else "N/A"
    f_range_str = f"{f_start.strftime('%b %d')} - {f_end.strftime('%b %d')}" if (f_start and f_end) else "N/A"

    lines.append(f'<div style="margin-top:1.2rem; padding:0.9rem 1.2rem; background:{header_bg}; border:1px solid {border_color}; border-radius:14px; display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:12px; font-size:0.82rem;">')
    lines.append(f'<div style="color:{text_color};">🩸 <b>Next Period:</b> <span style="color:#F472B6; font-weight:600;">{p_range_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">⭐ <b>Ovulation Peak:</b> <span style="color:#A78BFA; font-weight:600;">{ov_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">🌿 <b>Fertile Window:</b> <span style="color:#2DD4BF; font-weight:600;">{f_range_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">📝 <b>Logged Symptoms:</b> <span style="color:#38BDF8; font-weight:600;">{symptoms_in_month} Days</span></div>')
    lines.append(f'</div>')

    lines.append(f'</div>')

    full_html = "\n".join(lines)
    return _clean_html(full_html)
