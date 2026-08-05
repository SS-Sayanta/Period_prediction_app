"""
analytics.py — Smart Anomaly Detection Engine & Multi-Cycle Projection for FemCare AI.
Analyzes user history and symptoms for irregularity patterns, PCOS indicators, and stress correlations.
Generates multi-cycle forward projections.
"""

from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def generate_multi_cycle_projection(
    start_date: date,
    predicted_cycle_days: float,
    num_cycles: int = 6
) -> List[Dict[str, Any]]:
    """
    Generates a multi-cycle (default 6 months) predictive projection list.
    Each item contains cycle index, period start date, period end date, ovulation peak date, and fertile window start/end.
    """
    projections = []
    current_start = start_date
    cycle_duration = max(20.0, min(50.0, float(predicted_cycle_days)))

    for i in range(1, num_cycles + 1):
        # Estimated period duration ~ 5 days
        p_end = current_start + timedelta(days=4)
        
        # Next period start based on predicted cycle length
        next_start = current_start + timedelta(days=int(round(cycle_duration)))
        
        # Estimated ovulation date (typically 14 days before next period)
        ovulation_date = next_start - timedelta(days=14)
        
        # Fertile window: 5 days before ovulation to 1 day after
        fertile_start = ovulation_date - timedelta(days=5)
        fertile_end = ovulation_date + timedelta(days=1)

        projections.append({
            "cycle_number": i,
            "period_start": current_start,
            "period_end": p_end,
            "ovulation_date": ovulation_date,
            "fertile_start": fertile_start,
            "fertile_end": fertile_end,
            "cycle_length": int(round(cycle_duration))
        })

        current_start = next_start

    return projections


def analyze_cycle_anomalies(
    user_hist_df: pd.DataFrame,
    symptoms_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Performs algorithmic health analysis on historical user cycle data and symptom logs.
    Identifies cycle variability, PCOS risk factors, stress correlations, and symptom clusters.
    """
    insights = []
    anomalies_detected = []
    
    # Defaults
    cycle_lengths = []
    mean_length = 28.0
    std_length = 0.0
    range_length = 0.0
    status_level = "normal"  # normal, warning, caution
    pcos_risk = False
    irregularity_flag = False

    if not user_hist_df.empty and "predicted_cycle_days" in user_hist_df.columns:
        lengths_raw = pd.to_numeric(user_hist_df["predicted_cycle_days"], errors="coerce").dropna().values
        if len(lengths_raw) > 0:
            cycle_lengths = list(lengths_raw)
            mean_length = float(np.mean(cycle_lengths))
            std_length = float(np.std(cycle_lengths)) if len(cycle_lengths) > 1 else 0.0
            range_length = float(np.ptp(cycle_lengths)) if len(cycle_lengths) > 1 else 0.0

    # Rule 1: Cycle Variability Check
    if std_length > 4.5 or range_length > 7.0:
        irregularity_flag = True
        status_level = "warning"
        anomalies_detected.append("High Cycle Duration Variability (> 4.5 days std dev)")
        insights.append({
            "title": "⚠️ Significant Cycle Variability",
            "type": "warning",
            "desc": f"Your recorded cycle lengths vary by {range_length:.1f} days (std dev: {std_length:.1f} days). High fluctuation can indicate stress, sleep disruption, or hormonal recalibration."
        })
    else:
        insights.append({
            "title": "✅ Regular Cycle Rhythm",
            "type": "normal",
            "desc": f"Your cycle length is stable with an average of {mean_length:.1f} days and minimal variation (std dev: {std_length:.1f} days)."
        })

    # Rule 2: Prolonged / Oligomenorrhea / PCOS Risk Marker
    if mean_length > 35.0 or (len(cycle_lengths) >= 2 and min(cycle_lengths) > 35.0):
        pcos_risk = True
        status_level = "caution"
        anomalies_detected.append("Prolonged Cycle Length (> 35 days)")
        insights.append({
            "title": "🩺 Prolonged Cycle Marker (Oligomenorrhea)",
            "type": "caution",
            "desc": "Consistently long cycles (> 35 days) can be associated with anovulatory cycles or polycystic ovary syndrome (PCOS). We recommend sharing these trends with your healthcare provider."
        })
    elif mean_length < 21.0:
        status_level = "warning"
        anomalies_detected.append("Shortened Cycle Length (< 21 days)")
        insights.append({
            "title": "⚡ Shortened Cycle Marker (Polymenorrhea)",
            "type": "warning",
            "desc": "Cycle durations shorter than 21 days may reflect luteal phase insufficiency or frequent uterine shedding. Track your ovulation peaks closely."
        })

    # Rule 3: Symptom Severity & Stress Correlation
    high_cramps = False
    high_fatigue = False
    if not symptoms_df.empty:
        if "cramps" in symptoms_df.columns:
            avg_cramps = pd.to_numeric(symptoms_df["cramps"], errors="coerce").mean()
            if avg_cramps >= 3.5:
                high_cramps = True
                insights.append({
                    "title": "🩸 High Cramp Severity Cluster",
                    "type": "warning",
                    "desc": f"Logged cramp severity averages {avg_cramps:.1f}/5. Consider anti-inflammatory nutrition (omega-3s, magnesium) and heat therapy."
                })
        if "fatigue" in symptoms_df.columns:
            avg_fatigue = pd.to_numeric(symptoms_df["fatigue"], errors="coerce").mean()
            if avg_fatigue >= 3.5:
                high_fatigue = True
                insights.append({
                    "title": "🌙 Elevated Fatigue Pattern",
                    "type": "warning",
                    "desc": f"Logged fatigue averages {avg_fatigue:.1f}/5 during luteal and menstrual phases. Ensure adequate iron intake and sleep hygiene."
                })

    # Overall Summary
    summary_text = "Your cycle health markers indicate optimal regularity."
    if status_level == "caution":
        summary_text = "Notable hormonal variations detected. Review AI insights and medical guidance below."
    elif status_level == "warning":
        summary_text = "Moderate cycle length or symptom fluctuations recorded."

    medical_disclaimer = (
        "Medical Disclaimer: FemCare AI anomaly detection algorithms are intended for informational "
        "and health tracking purposes only. They do not constitute formal medical diagnosis or clinical treatment advice. "
        "Please consult a licensed healthcare professional or board-certified gynecologist for medical evaluations."
    )

    return {
        "status_level": status_level,
        "mean_length": round(mean_length, 1),
        "std_length": round(std_length, 1),
        "range_length": round(range_length, 1),
        "irregularity_flag": irregularity_flag,
        "pcos_risk": pcos_risk,
        "anomalies_detected": anomalies_detected,
        "insights": insights,
        "summary_text": summary_text,
        "disclaimer": medical_disclaimer
    }
