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
    
    # Embedded Responsive Stylesheet for Streamlit & Standalone HTML
    lines.append("<style>")
    lines.append(f"""
    .calendar-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 1.5rem;
        font-family: Inter, sans-serif;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        width: 100%;
        box-sizing: border-box;
        overflow-x: auto;
        max-width: 100%;
    }}
    .calendar-grid {{
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 4px;
        width: 100%;
        box-sizing: border-box;
    }}
    .calendar-header-cell {{
        padding: 8px 4px;
        background: {header_bg};
        border-radius: 8px;
        color: {muted_text};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        text-align: center;
        box-sizing: border-box;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .calendar-day-cell {{
        padding: 8px 4px;
        border-radius: 12px;
        min-height: 64px;
        box-sizing: border-box;
        min-width: 0;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease;
    }}
    .calendar-badge {{
        font-size: 0.72rem;
        font-weight: 600;
        display: block;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        text-align: center;
    }}
    @media (max-width: 600px) {{
        .calendar-card {{
            padding: 0.8rem;
            border-radius: 14px;
        }}
        .calendar-grid {{
            display: grid !important;
            grid-template-columns: repeat(7, minmax(0, 1fr)) !important;
            gap: 3px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}
        .calendar-header-cell {{
            padding: 4px 1px;
            font-size: 0.62rem;
            letter-spacing: 0;
        }}
        .calendar-day-cell {{
            padding: 4px 2px;
            min-height: 46px;
            border-radius: 8px;
        }}
        .calendar-day-num {{
            font-size: 0.75rem !important;
        }}
        .calendar-badge {{
            font-size: 0.6rem !important;
            line-height: 1.1;
            margin-top: 1px;
        }}
        .calendar-legend-pill {{
            padding: 2px 6px !important;
            font-size: 0.68rem !important;
        }}
        .calendar-footer-summary {{
            padding: 0.6rem 0.8rem !important;
            font-size: 0.75rem !important;
            flex-direction: column;
            align-items: flex-start !important;
            gap: 6px !important;
        }}
    }}
    """)
    lines.append("</style>")

    lines.append('<div class="calendar-card">')
    
    # Header Row
    lines.append('<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.4rem; flex-wrap:wrap; gap:12px;">')
    lines.append('<div>')
    lines.append(f'<h3 style="margin:0; color:{text_color}; font-size:1.4rem; font-weight:700; letter-spacing:-0.01em;">📅 {month_name} {year}</h3>')
    lines.append(f'<div style="font-size:0.78rem; color:{muted_text}; margin-top:2px;">Interactive Cycle & Symptom Tracking Grid</div>')
    lines.append('</div>')
    
    # Visual Legend Pills
    lines.append('<div style="display:flex; gap:6px; font-size:0.76rem; flex-wrap:wrap;">')
    lines.append('<span class="calendar-legend-pill" style="display:inline-flex; align-items:center; gap:4px; background:rgba(244,114,182,0.18); border:1px solid rgba(244,114,182,0.4); color:#F472B6; padding:4px 10px; border-radius:8px; font-weight:600;">🩸 Period</span>')
    lines.append('<span class="calendar-legend-pill" style="display:inline-flex; align-items:center; gap:4px; background:rgba(45,212,191,0.18); border:1px solid rgba(45,212,191,0.4); color:#2DD4BF; padding:4px 10px; border-radius:8px; font-weight:600;">🌿 Fertile</span>')
    lines.append('<span class="calendar-legend-pill" style="display:inline-flex; align-items:center; gap:4px; background:rgba(167,139,250,0.22); border:1px solid rgba(167,139,250,0.4); color:#A78BFA; padding:4px 10px; border-radius:8px; font-weight:600;">⭐ Ovulation</span>')
    lines.append('<span class="calendar-legend-pill" style="display:inline-flex; align-items:center; gap:4px; background:rgba(56,189,248,0.18); border:1px solid rgba(56,189,248,0.4); color:#38BDF8; padding:4px 10px; border-radius:8px; font-weight:600;">📝 Logged</span>')
    lines.append('</div>')
    lines.append('</div>')

    # Calendar Grid
    lines.append('<div class="calendar-grid">')

    # Headers
    for day_head in ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]:
        lines.append(f'<div class="calendar-header-cell">{day_head}</div>')

    # Day Cells
    for week in month_days:
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
                    badge_content += '<span class="calendar-badge" style="color:#F472B6;">🩸 Period</span>'
                elif is_ovulation:
                    cell_bg = "rgba(167, 139, 250, 0.32)"
                    cell_border = "1px solid #A78BFA"
                    badge_content += '<span class="calendar-badge" style="color:#A78BFA;">⭐ Ovulation</span>'
                elif is_fertile:
                    cell_bg = "rgba(45, 212, 191, 0.22)"
                    cell_border = "1px solid #2DD4BF"
                    badge_content += '<span class="calendar-badge" style="color:#2DD4BF;">🌿 Fertile</span>'

                if has_symptom:
                    badge_content += '<span class="calendar-badge" style="color:#38BDF8;">📝 Logged</span>'

                if is_today and not (is_period or is_ovulation or is_fertile):
                    badge_content += '<span class="calendar-badge" style="color:#38BDF8; font-weight:700;">TODAY</span>'

            today_dot = '<span style="width:5px; height:5px; background:#38BDF8; border-radius:50%; display:inline-block; flex-shrink:0;"></span>' if (is_current_month and d == today) else ''
            day_font_size = "1.02rem" if (is_current_month and d == today) else "0.9rem"

            lines.append(
                f'<div class="calendar-day-cell" title="{cell_title}" style="border:{cell_border}; background:{cell_bg}; color:{cell_color}; opacity:{opacity};">'
                f'<div style="font-weight:700; display:flex; justify-content:space-between; align-items:center; padding:0 2px;">'
                f'<span class="calendar-day-num" style="font-size:{day_font_size};">{day_num}</span>'
                f'{today_dot}'
                f'</div>'
                f'<div style="margin-top:2px; min-height:16px; display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; overflow:hidden;">{badge_content}</div>'
                f'</div>'
            )

    lines.append('</div>')  # end calendar-grid

    # Footer Card Summary
    p_range_str = f"{p_start.strftime('%b %d')} - {p_end.strftime('%b %d')}" if (p_start and p_end) else "N/A"
    ov_str = ovulation.strftime('%b %d, %Y') if ovulation else "N/A"
    f_range_str = f"{f_start.strftime('%b %d')} - {f_end.strftime('%b %d')}" if (f_start and f_end) else "N/A"

    lines.append(f'<div class="calendar-footer-summary" style="margin-top:1.2rem; padding:0.9rem 1.2rem; background:{header_bg}; border:1px solid {border_color}; border-radius:14px; display:flex; justify-content:space-around; align-items:center; flex-wrap:wrap; gap:12px; font-size:0.82rem;">')
    lines.append(f'<div style="color:{text_color};">🩸 <b>Next Period:</b> <span style="color:#F472B6; font-weight:600;">{p_range_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">⭐ <b>Ovulation Peak:</b> <span style="color:#A78BFA; font-weight:600;">{ov_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">🌿 <b>Fertile Window:</b> <span style="color:#2DD4BF; font-weight:600;">{f_range_str}</span></div>')
    lines.append(f'<div style="color:{text_color};">📝 <b>Logged Symptoms:</b> <span style="color:#38BDF8; font-weight:600;">{symptoms_in_month} Days</span></div>')
    lines.append('</div>')

    lines.append('</div>')  # end calendar-card

    full_html = "\n".join(lines)
    return _clean_html(full_html)

