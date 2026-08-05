"""
main.py — FastAPI Backend Engine for Enterprise React + Vercel FemCare AI Platform.
Wraps ML prediction pipeline, anomaly detection, multi-cycle forecasting,
symptom logging, ground-truth verification, and PDF report generation.
"""

from __future__ import annotations
import os
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import domain modules
from data_manager import (
    init_and_clean_files,
    save_user_history,
    save_symptoms_log,
    save_verified_feedback,
    load_user_history,
    load_symptoms_log,
    load_verified_feedback,
    load_reference_dataset
)
from ml_pipeline import (
    ModelManager,
    predict_cycle_length,
    compute_cycle_phases,
    CyclePredictionResult
)
from analytics import analyze_cycle_anomalies, generate_multi_cycle_projection
from report_generator import generate_medical_pdf
from assistant import get_ai_assistant_response

# Initialize CSV storage files on startup
init_and_clean_files()
model_mgr = ModelManager.get_instance()

app = FastAPI(
    title="FemCare AI Enterprise API",
    description="Vercel REST API for FemCare AI Enterprise React Application",
    version="3.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Request Models ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    user_name: str = "Valued User"
    user_id: Optional[str] = None
    last_period_date: str = Field(default_factory=lambda: (date.today() - timedelta(days=14)).isoformat())
    age: int = 25
    weight_kg: float = 60.0
    stress_level: int = 3
    avg_sleep_hours: float = 7.5
    exercise_hours: float = 3.0
    water_intake_liters: float = 2.0
    mood_score: int = 3
    last_cycle_length: float = 28.0
    cycle_length_2: float = 28.0
    cycle_length_3: float = 28.0


class SymptomLogRequest(BaseModel):
    user_id: Optional[str] = None
    log_date: str = Field(default_factory=lambda: date.today().isoformat())
    cramps: int = 0
    bloating: int = 0
    mood: int = 3
    energy: int = 3
    headache: int = 0
    fatigue: int = 0
    flow_level: str = "None"
    notes: str = ""


class FeedbackRequest(BaseModel):
    user_id: Optional[str] = None
    feedback_date: str = Field(default_factory=lambda: date.today().isoformat())
    actual_cycle_length: float = 28.0
    predicted_cycle_length: Optional[float] = None


class AssistantRequest(BaseModel):
    query: str
    current_phase: str = "Follicular"
    operating_mode: str = "Cycle Tracking"
    age: int = 25
    stress_level: int = 3


# ── Helper for Internal Inference Execution ──────────────────────────────
def run_internal_prediction(req: PredictRequest) -> Dict[str, Any]:
    try:
        p_date = date.fromisoformat(req.last_period_date)
    except Exception:
        p_date = date.today() - timedelta(days=14)

    uid = req.user_id if req.user_id else str(uuid.uuid4())

    pred_days, used_ml, confidence, notes, fdict = predict_cycle_length(
        age=req.age,
        weight_kg=req.weight_kg,
        stress_level=req.stress_level,
        avg_sleep_hours=req.avg_sleep_hours,
        exercise_hours=req.exercise_hours,
        water_intake_liters=req.water_intake_liters,
        mood_score=req.mood_score,
        last_cycle_length=req.last_cycle_length,
        cycle_length_2=req.cycle_length_2,
        cycle_length_3=req.cycle_length_3,
    )

    res = compute_cycle_phases(last_period_date=p_date, predicted_cycle_days=pred_days)
    res.used_ml_model = used_ml
    res.confidence_score = confidence
    res.model_notes = notes
    res.feature_summary = fdict

    save_user_history(
        user_id=uid,
        user_name=req.user_name,
        last_period_date=p_date,
        predicted_cycle_days=pred_days
    )

    user_hist_df = load_user_history()
    symptoms_df = load_symptoms_log()
    anomaly_data = analyze_cycle_anomalies(user_hist_df, symptoms_df)
    multi_projections = generate_multi_cycle_projection(p_date, pred_days, num_cycles=6)

    formatted_projections = []
    for proj in multi_projections:
        formatted_projections.append({
            "cycle_number": proj["cycle_number"],
            "period_start": proj["period_start"].isoformat(),
            "period_end": proj["period_end"].isoformat(),
            "ovulation_date": proj["ovulation_date"].isoformat(),
            "fertile_start": proj["fertile_start"].isoformat(),
            "fertile_end": proj["fertile_end"].isoformat(),
            "cycle_length": proj["cycle_length"]
        })

    return {
        "user_id": uid,
        "user_name": req.user_name,
        "predicted_cycle_days": round(pred_days, 1),
        "days_until_next_period": res.days_until_next_period,
        "next_period_start": res.next_period_start.isoformat(),
        "next_period_end": res.next_period_end.isoformat(),
        "ovulation_date": res.ovulation_date.isoformat(),
        "fertile_window_start": res.fertile_window_start.isoformat(),
        "fertile_window_end": res.fertile_window_end.isoformat(),
        "current_phase": res.current_phase,
        "current_cycle_day": res.current_cycle_day,
        "used_ml_model": used_ml,
        "confidence_score": round(confidence, 2),
        "model_notes": notes,
        "anomaly_data": anomaly_data,
        "multi_projections": formatted_projections
    }


# ── REST API Endpoints ───────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "FemCare AI API", "ml_loaded": model_mgr.is_loaded}


# 1. Prediction Overview Endpoint (supporting both /api/overview and /api/prediction/overview)
@app.get("/api/overview")
@app.get("/api/prediction/overview")
def get_prediction_overview():
    req = PredictRequest()
    return run_internal_prediction(req)


@app.post("/api/predict")
def post_predict(req: PredictRequest):
    return run_internal_prediction(req)


# 2. Calendar Events Endpoint (supporting both /api/calendar and /api/calendar/events)
@app.get("/api/calendar")
@app.get("/api/calendar/events")
def get_calendar_events(
    year: int = Query(default=datetime.now().year),
    month: int = Query(default=datetime.now().month),
    last_period_date: str = Query(default=(date.today() - timedelta(days=14)).isoformat()),
    cycle_days: float = Query(default=28.0)
):
    try:
        p_date = date.fromisoformat(last_period_date)
    except Exception:
        p_date = date.today() - timedelta(days=14)

    res = compute_cycle_phases(last_period_date=p_date, predicted_cycle_days=cycle_days)
    symptoms_df = load_symptoms_log()
    logged_dates = symptoms_df["log_date"].astype(str).tolist() if not symptoms_df.empty and "log_date" in symptoms_df.columns else []

    import math
    import calendar as pycal
    cal = pycal.Calendar(firstweekday=6)
    month_days = cal.monthdatescalendar(year, month)
    today_str = date.today().isoformat()

    cycle_length = float(cycle_days)
    period_duration = 5

    events = []
    weeks = []

    for week in month_days:
        w_days = []
        for d in week:
            d_str = d.isoformat()
            is_current = (d.month == month)

            # Dynamic multi-cycle phase alignment
            days_diff = (d - p_date).days
            n_cycles = math.floor(days_diff / cycle_length)
            
            c_start = p_date + timedelta(days=round(n_cycles * cycle_length))
            c_next_start = p_date + timedelta(days=round((n_cycles + 1) * cycle_length))
            c_period_end = c_start + timedelta(days=period_duration - 1)
            c_ovulation = c_next_start - timedelta(days=14)
            c_fertile_start = c_ovulation - timedelta(days=4)
            c_fertile_end = c_ovulation + timedelta(days=1)

            is_period = bool(c_start <= d <= c_period_end)
            is_ovulation = bool(d == c_ovulation)
            is_fertile = bool(c_fertile_start <= d <= c_fertile_end)

            if is_period:
                phase_name = "Menstrual Phase (Period)"
                phase_code = "period"
                conception_prob = "Very Low (< 1%)"
            elif is_ovulation:
                phase_name = "Ovulation Day Peak ⭐"
                phase_code = "ovulation"
                conception_prob = "Peak (~98%)"
            elif is_fertile:
                phase_name = "High Fertility Window 🌿"
                phase_code = "fertile"
                conception_prob = "High (~85%)"
            elif d < c_ovulation:
                phase_name = "Follicular Phase 🌱"
                phase_code = "follicular"
                conception_prob = "Moderate (~20%)"
            else:
                phase_name = "Luteal Phase 🌾"
                phase_code = "luteal"
                conception_prob = "Low (< 5%)"

            symptom_detail = None
            if not symptoms_df.empty and "log_date" in symptoms_df.columns:
                match = symptoms_df[symptoms_df["log_date"].astype(str) == d_str]
                if not match.empty:
                    row = match.iloc[-1]
                    symptom_detail = {
                        "flow": str(row.get("flow_level", "None")),
                        "cramps": int(row.get("cramps", 0)),
                        "fatigue": int(row.get("fatigue", 0)),
                        "notes": str(row.get("notes", ""))
                    }
            has_symptom = symptom_detail is not None

            day_obj = {
                "date": d_str,
                "day_number": d.day,
                "is_current_month": is_current,
                "is_today": (d_str == today_str),
                "is_period": is_period,
                "is_ovulation": is_ovulation,
                "is_fertile": is_fertile,
                "has_symptom": has_symptom,
                "symptom_detail": symptom_detail,
                "phase": phase_name,
                "phase_code": phase_code,
                "conception_probability": conception_prob
            }
            w_days.append(day_obj)

            if is_period:
                events.append({"date": d_str, "type": "period", "title": "Predicted Period", "color": "#F472B6"})
            if is_ovulation:
                events.append({"date": d_str, "type": "ovulation", "title": "Ovulation Peak", "color": "#A78BFA"})
            if is_fertile and not is_ovulation:
                events.append({"date": d_str, "type": "fertile", "title": "Fertile Window", "color": "#2DD4BF"})
            if has_symptom:
                events.append({"date": d_str, "type": "symptom", "title": "Symptom Logged", "color": "#38BDF8"})

        weeks.append(w_days)

    return {
        "year": year,
        "month": month,
        "month_name": pycal.month_name[month],
        "events": events,
        "weeks": weeks,
        "next_period_start": res.next_period_start.isoformat(),
        "next_period_end": res.next_period_end.isoformat(),
        "ovulation_date": res.ovulation_date.isoformat(),
        "fertile_window_start": res.fertile_window_start.isoformat(),
        "fertile_window_end": res.fertile_window_end.isoformat(),
    }


# 3. Log Symptoms Endpoint (supporting both /api/log-symptoms and /api/symptoms/log)
@app.post("/api/log-symptoms")
@app.post("/api/symptoms/log")
def log_symptoms(req: SymptomLogRequest):
    uid = req.user_id if req.user_id else str(uuid.uuid4())
    try:
        ld = date.fromisoformat(req.log_date)
    except Exception:
        ld = date.today()

    ok = save_symptoms_log(
        user_id=uid,
        log_date=ld,
        cramps=req.cramps,
        bloating=req.bloating,
        mood=req.mood,
        energy=req.energy,
        headache=req.headache,
        fatigue=req.fatigue,
        flow_level=req.flow_level,
        notes=req.notes
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write symptom entry")
    return {"status": "success", "message": "Daily symptom log recorded!"}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    uid = req.user_id if req.user_id else str(uuid.uuid4())
    try:
        fd = date.fromisoformat(req.feedback_date)
    except Exception:
        fd = date.today()

    pred_val = req.predicted_cycle_length if req.predicted_cycle_length else 28.0

    ok = save_verified_feedback(
        user_id=uid,
        feedback_date=fd,
        actual_cycle_length=req.actual_cycle_length,
        predicted_cycle_length=pred_val,
        feature_dict={}
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write feedback entry")
    return {"status": "success", "message": "Ground truth verified feedback recorded!"}


# 4. Analytics Endpoint (/api/analytics)
@app.get("/api/analytics")
def get_analytics():
    user_hist_df = load_user_history()
    symptoms_df = load_symptoms_log()
    feedback_df = load_verified_feedback()
    ref_df = load_reference_dataset()

    mae = 0.0
    if not feedback_df.empty and "actual_cycle_length" in feedback_df.columns and "predicted_cycle_length" in feedback_df.columns:
        actuals = pd.to_numeric(feedback_df["actual_cycle_length"], errors="coerce")
        preds = pd.to_numeric(feedback_df["predicted_cycle_length"], errors="coerce")
        valid = (actuals.notnull()) & (preds.notnull())
        if valid.any():
            mae = float(np.mean(np.abs(actuals[valid] - preds[valid])))

    history_records = []
    if not user_hist_df.empty:
        df_copy = user_hist_df.tail(20).copy()
        for _, row in df_copy.iterrows():
            history_records.append({
                "user_reference": f"User ({str(row.get('user_id', ''))[:6]}...)",
                "user_name": str(row.get("user_name", "Valued User")),
                "last_period_date": str(row.get("last_period_date", "")),
                "predicted_cycle_days": f"{float(row.get('predicted_cycle_days', 28.0)):.1f} Days",
                "recorded_timestamp": str(row.get("timestamp", str(row.get("recorded_at", ""))))[:16]
            })

    symptom_records = []
    symptom_heatmap = {"cramps": 0, "fatigue": 0, "bloating": 0, "headache": 0}
    if not symptoms_df.empty:
        s_copy = symptoms_df.tail(20).copy()
        for _, row in s_copy.iterrows():
            c_val = int(row.get("cramps", 0))
            f_val = int(row.get("fatigue", 0))
            b_val = int(row.get("bloating", 0))
            h_val = int(row.get("headache", 0))

            symptom_heatmap["cramps"] += c_val
            symptom_heatmap["fatigue"] += f_val
            symptom_heatmap["bloating"] += b_val
            symptom_heatmap["headache"] += h_val

            symptom_records.append({
                "user_reference": f"User ({str(row.get('user_id', ''))[:6]}...)",
                "log_date": str(row.get("log_date", "")),
                "flow_intensity": str(row.get("flow_level", "None")),
                "cramps": c_val,
                "bloating": b_val,
                "mood": int(row.get("mood", 3)),
                "energy": int(row.get("energy", 3)),
                "headache": h_val,
                "fatigue": f_val,
                "notes": str(row.get("notes", ""))
            })

    anomaly_data = analyze_cycle_anomalies(user_hist_df, symptoms_df)

    return {
        "mae": round(mae, 2),
        "history_records": history_records,
        "symptom_records": symptom_records,
        "symptom_heatmap": symptom_heatmap,
        "anomaly_data": anomaly_data,
        "ml_loaded": model_mgr.is_loaded
    }


# 5. AI Assistant Endpoint (/api/assistant and /api/chat)
@app.post("/api/assistant")
@app.post("/api/chat")
def chat_assistant(req: AssistantRequest):
    answer = get_ai_assistant_response(
        query=req.query,
        current_phase=req.current_phase,
        operating_mode=req.operating_mode,
        age=req.age,
        stress_level=req.stress_level
    )
    return {"query": req.query, "response": answer, "answer": answer}


# 6. PDF Export Endpoint (/api/export-pdf)
@app.get("/api/export-pdf")
def export_pdf(
    user_name: str = Query(default="Valued User"),
    user_id: str = Query(default="demo-user-session")
):
    user_hist_df = load_user_history()
    symptoms_df = load_symptoms_log()
    feedback_df = load_verified_feedback()
    anomaly_data = analyze_cycle_anomalies(user_hist_df, symptoms_df)

    p_date = date.today() - timedelta(days=14)
    res = compute_cycle_phases(last_period_date=p_date, predicted_cycle_days=28.0)

    pdf_bytes = generate_medical_pdf(
        user_name=user_name,
        user_id=user_id,
        user_hist_df=user_hist_df,
        symptoms_df=symptoms_df,
        feedback_df=feedback_df,
        current_res=res,
        anomaly_data=anomaly_data
    )

    filename = f"FemCare_Medical_Report_{user_name.replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# Serve Static Frontend Files from "public" Directory
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
