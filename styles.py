"""
styles.py — Enterprise HealthTech CSS Design System for FemCare AI.
Provides light and dark theme styles, soft pastel glassmorphism, responsive metrics, and UI polish.
"""

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

/* Base Font & Theme setup */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F8FAFC;
}

h1, h2, h3, h4, .brand-title {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.02em;
}

/* Hide default streamlit header & footer clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(244, 114, 182, 0.15) 0%, rgba(167, 139, 250, 0.15) 50%, rgba(45, 212, 191, 0.15) 100%);
    border: 1px solid rgba(244, 114, 182, 0.25);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(244, 114, 182, 0.15);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #F472B6, #A78BFA, #2DD4BF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #94A3B8;
    margin-bottom: 0.5rem;
}

/* Metric Cards Grid */
.metric-card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    backdrop-filter: blur(8px);
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    border-color: rgba(244, 114, 182, 0.3);
}

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
}

.metric-sub {
    font-size: 0.8rem;
    color: #34D399;
    margin-top: 0.3rem;
}

/* Operating Mode Badge */
.mode-badge-tracking {
    background: rgba(244, 114, 182, 0.15);
    border: 1px solid rgba(244, 114, 182, 0.4);
    color: #F472B6;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.mode-badge-conception {
    background: rgba(45, 212, 191, 0.15);
    border: 1px solid rgba(45, 212, 191, 0.4);
    color: #2DD4BF;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.mode-badge-pregnancy {
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.4);
    color: #A78BFA;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Anomaly Insight Cards */
.insight-card {
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
}

.insight-card-green {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.3);
}

.insight-card-amber {
    background: rgba(245, 158, 11, 0.08);
    border-color: rgba(245, 158, 11, 0.3);
}

.insight-card-rose {
    background: rgba(244, 63, 94, 0.08);
    border-color: rgba(244, 63, 94, 0.3);
}

/* Status Badges */
.badge-ml {
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: #34D399;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.badge-fallback {
    background: rgba(251, 146, 60, 0.15);
    border: 1px solid rgba(251, 146, 60, 0.3);
    color: #FB923C;
    padding: 0.35rem 0.85rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Phase Pills */
.phase-pill {
    padding: 0.4rem 1rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.88rem;
    display: inline-block;
}

.phase-menstrual {
    background: rgba(244, 114, 182, 0.2);
    color: #F472B6;
    border: 1px solid rgba(244, 114, 182, 0.4);
}

.phase-follicular {
    background: rgba(167, 139, 250, 0.2);
    color: #A78BFA;
    border: 1px solid rgba(167, 139, 250, 0.4);
}

.phase-fertile {
    background: rgba(45, 212, 191, 0.2);
    color: #2DD4BF;
    border: 1px solid rgba(45, 212, 191, 0.4);
}

.phase-ovulation {
    background: rgba(250, 204, 21, 0.2);
    color: #FACC15;
    border: 1px solid rgba(250, 204, 21, 0.4);
}

.phase-luteal {
    background: rgba(56, 189, 248, 0.2);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

/* Form Container Styling */
.form-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Custom Button Styling */
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #F472B6 0%, #E11D48 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(244, 114, 182, 0.3) !important;
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(244, 114, 182, 0.4) !important;
}

/* Streamlit DataFrame & Tabs Refinement */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 16px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: rgba(244, 114, 182, 0.15) !important;
    color: #F472B6 !important;
}

/* Responsive 7-Column Calendar Grid System */
.calendar-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 1.5rem;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    width: 100%;
    box-sizing: border-box;
    overflow-x: auto;
    max-width: 100%;
}

.calendar-grid {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 4px;
    width: 100%;
    box-sizing: border-box;
}

.calendar-header-cell {
    padding: 8px 4px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    color: #94A3B8;
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
}

.calendar-day-cell {
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
}

.calendar-badge {
    font-size: 0.72rem;
    font-weight: 600;
    display: block;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
    text-align: center;
}

@media (max-width: 600px) {
    .calendar-card {
        padding: 0.8rem;
        border-radius: 14px;
    }
    .calendar-grid {
        display: grid !important;
        grid-template-columns: repeat(7, minmax(0, 1fr)) !important;
        gap: 4px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    .calendar-header-cell {
        padding: 4px 1px;
        font-size: 0.62rem;
        letter-spacing: 0;
    }
    .calendar-day-cell {
        padding: 4px 2px;
        min-height: 46px;
        border-radius: 8px;
    }
    .calendar-day-num {
        font-size: 0.75rem !important;
    }
    .calendar-badge {
        font-size: 0.6rem !important;
        line-height: 1.1;
        margin-top: 1px;
    }
    .calendar-legend-pill {
        padding: 2px 6px !important;
        font-size: 0.68rem !important;
    }
    .calendar-footer-summary {
        padding: 0.6rem 0.8rem !important;
        font-size: 0.75rem !important;
        flex-direction: column;
        align-items: flex-start !important;
        gap: 6px !important;
    }
}
</style>
"""
