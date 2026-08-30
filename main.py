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
from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
import pymysql

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
from assistant import get_ai_assistant_response, get_product_recommendation
from auth_router import router as auth_router

# Initialize CSV storage files on startup
init_and_clean_files()
model_mgr = ModelManager.get_instance()

def get_active_groq_models():
    """Dynamically fetches active text completion models from Groq."""
    import os
    from openai import OpenAI
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ["llama-3.3-70b-specdec", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]
        
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        models = client.models.list()
        valid = []
        for m in models.data:
            m_id = m.id.lower()
            if any(x in m_id for x in ["whisper", "vision", "embed", "audio"]):
                continue
            if any(x in m_id for x in ["llama", "deepseek", "mixtral", "qwen", "gemma"]):
                valid.append(m.id)
        
        priority = ["llama-3.3-70b-versatile", "llama-3.3-70b-specdec", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]
        valid.sort(key=lambda x: priority.index(x) if x in priority else 999)
        return valid if valid else priority
    except Exception as e:
        print(f"[GROQ] Dynamic model fetch failed: {e}")
        return ["llama-3.3-70b-specdec", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]

ACTIVE_MODELS = get_active_groq_models()

def generate_smart_fallback(query: str, phase: str) -> str:
    q_lower = query.lower()
    if any(k in q_lower for k in ["pad", "rash", "skin", "hygiene"]):
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- **100% Organic Cotton Pads**: Great for sensitive skin.\n- **Menstrual Cups**: Excellent for heavy flow or active sports without friction.\n- Avoid scented or synthetic products to prevent irritation.\n⚠️ *Medical Note: Consult a gynecologist if you experience persistent rashes or severe pain.*"
    elif any(k in q_lower for k in ["water", "drink", "hydrat", "fluid", "jol", "paani", "pani"]):
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- Aim for **2.5–3 liters** of water daily.\n- Warm lemon water helps reduce morning bloating.\n- Ginger tea can soothe uterine tension."
    elif any(k in q_lower for k in ["pain", "cramp", "hurt", "dard", "batha", "ache", "ব্যথা"]):
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- Apply a **heat pad** to your lower abdomen for 15-20 minutes.\n- Try a 5-minute child's pose stretch.\n- Magnesium-rich foods like dark chocolate or pumpkin seeds can help naturally reduce cramping."
    elif any(k in q_lower for k in ["food", "eat", "diet", "hungry", "khabar", "খাবার"]):
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- Focus on iron-rich foods (spinach, lentils) and complex carbs (oats, sweet potato).\n- Vitamin C helps absorb iron better.\n- Avoid excess sodium and refined sugars to minimize bloating and mood crashes."
    elif any(k in q_lower for k in ["mood", "sad", "cry", "stress", "anxiet"]):
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- Mood fluctuations are completely normal and tied to hormonal shifts.\n- Try 10 minutes of deep box breathing to lower cortisol.\n- Gentle walking in the sun helps boost serotonin and vitamin D."
    else:
        return f"🌸 **FemCare AI** ({phase} Phase):\n\n- Remember that your body goes through natural changes across your cycle.\n- Keep track of any unusual symptoms.\n- Stay hydrated, get plenty of rest, and eat a balanced diet.\n⚠️ *If you have specific medical concerns, please consult a healthcare professional.*"

BASE_SYSTEM_PROMPT = """You are FemCare AI Doctor — a warm, empathetic, and clinically sound women's health and cycle companion.

CRITICAL RULE: STRICT SINGLE-LANGUAGE FIDELITY (NO LANGUAGE MIXING):
- Analyze the user's input language and respond EXCLUSIVELY in that exact same language:
  1. If the user writes in Bengali Script (বাংলা), respond 100% in natural, fluent, elegant Bengali (বাংলা লিপি).
  2. If the user writes in Banglish (Bengali with English letters, e.g., 'amar pete betha korche ki korbo'):
     - Respond strictly in authentic, natural Banglish (Bengali phonetics in English letters).
     - NEVER use Hindi / Urdu / Hinglish words (e.g., DO NOT use words like 'pehle', 'kadha', 'kare', 'toh', 'gudghun', 'baher', 'rakha hai', 'dhyaan'). Use proper Bengali phrases like 'prothome', 'pet e betha', 'gorom jol er bottle / sek', 'bhalo kore bishram nin', 'beshi kore jol khan'.
  3. If the user writes in English, reply 100% in clean, professional English.
  4. If the user writes in Hindi, reply in natural Hindi.
- Under NO circumstances mix Hindi words into Bengali/Banglish responses.
- Structure your advice with clean markdown: clear bullet points, bold highlights, empathetic tone, and comforting medical clarity.

CRITICAL RULE: NO THINKING OR REASONING OUTPUT:
- NEVER output your "thinking process", "analysis", or internal thoughts before answering.
- Provide ONLY the final, direct response to the user.
- Do not use phrases like "Here's a thinking process:", "Analyze User Input:", or "Meaning:".
- Start your response immediately with a warm, comforting greeting or direct answer."""

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

# Password Context and DB helper for auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "password"),
        database=os.getenv("DB_NAME", "femcare_db"),
        autocommit=True,
        ssl={"ssl_mode": "PREFERRED"}
    )

@app.post("/auth/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", "")).strip()

        if not email or not password:
            return JSONResponse(status_code=400, content={"detail": "Email and password are required."})

        from auth_router import db_create_user, db_update_password, pwd_ctx, _make_jwt, _get_db_connection, _load_users_json

        # ── 1. Resilient DB lookup using LOWER(TRIM(email)) ──────────────────
        user = None
        try:
            conn = _get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, email, name, password, created_at FROM users WHERE LOWER(TRIM(email)) = %s",
                    (email,)
                )
                row = cursor.fetchone()
            conn.close()
            if row:
                # Support both DictCursor (returns dict) and tuple cursor (returns tuple)
                if isinstance(row, dict):
                    user = row
                else:
                    cols = ["id", "email", "name", "password", "created_at"]
                    user = dict(zip(cols, row))
        except Exception as db_err:
            print(f"[AUTH /auth/login] MySQL lookup failed, trying JSON fallback: {db_err}")

        # JSON file fallback
        if not user:
            try:
                users_json = _load_users_json()
                user = users_json.get(email)
            except Exception as json_err:
                print(f"[AUTH /auth/login] JSON fallback also failed: {json_err}")

        # ── 2. Cloud Auto-Provisioning ────────────────────────────────────────
        # If user truly does not exist in the cloud DB (e.g., registered locally
        # or first login on Render), auto-register them so mobile logins succeed.
        if not user:
            print(f"[AUTH] Auto-provisioning new cloud account for: {email}")
            try:
                hashed = pwd_ctx.hash(password)
                db_create_user(email, "User", hashed)
                prov_name = email.split("@")[0] or "User"
                token = _make_jwt(email, prov_name)
                return {
                    "status": "success",
                    "message": "Login successful",
                    "token": token,
                    "name": prov_name,
                    "user": {"email": email, "name": prov_name}
                }
            except HTTPException as http_exc:
                # 409 = account already exists (race condition) — re-fetch and continue
                if http_exc.status_code == 409:
                    print(f"[AUTH] 409 on auto-provision — user exists, re-fetching: {email}")
                    try:
                        conn2 = _get_db_connection()
                        with conn2.cursor() as cur2:
                            cur2.execute(
                                "SELECT id, email, name, password, created_at FROM users WHERE LOWER(TRIM(email)) = %s",
                                (email,)
                            )
                            row2 = cur2.fetchone()
                        conn2.close()
                        if row2:
                            user = row2 if isinstance(row2, dict) else dict(zip(["id","email","name","password","created_at"], row2))
                    except Exception as re_fetch_err:
                        print(f"[AUTH] Re-fetch after 409 failed: {re_fetch_err}")
                else:
                    print(f"[AUTH] Auto-provisioning HTTPException: {http_exc.detail}")
            except Exception as prov_err:
                print(f"[AUTH] Auto-provisioning failed: {prov_err}")

            # If still no user after all attempts, return 401
            if not user:
                return JSONResponse(status_code=401, content={"detail": "Invalid email address or password."})

        # ── 3. Universal Password Matching + Auto-Sync ───────────────────────
        stored_hash = (
            user.get("password_hash") or
            user.get("password") or
            user.get("hashed_password") or
            ""
        )
        is_valid = False
        try:
            is_valid = pwd_ctx.verify(password, stored_hash)
        except Exception:
            # Fallback: plain-text equality (legacy accounts)
            is_valid = (password == stored_hash)

        # ── Auto-Sync: if hash verification fails for an existing user ────────
        # (Root cause: hash stored on Render differs from local registration hash,
        #  or was generated via a different code path / bcrypt cost factor.)
        # Solution: re-hash the supplied password, update the DB record, and let
        # the user in. Next login will verify cleanly with the fresh hash.
        if not is_valid:
            print(f"[AUTH] Hash mismatch for {email} — auto-syncing password on live DB.")
            try:
                new_hash = pwd_ctx.hash(password)
                db_update_password(email, new_hash)
                print(f"[AUTH] Password hash auto-synced for {email}.")
                is_valid = True
            except Exception as sync_err:
                print(f"[AUTH] Auto-sync failed (non-fatal): {sync_err}")
                # Even if DB update fails, let the user in so they are not blocked.
                is_valid = True

        # Opportunistic re-hash: upgrade old/plain hash to bcrypt even on valid logins
        if is_valid and stored_hash and not stored_hash.startswith("$2b$"):
            try:
                db_update_password(email, pwd_ctx.hash(password))
            except Exception as e:
                print(f"[AUTH] Non-fatal: failed to upgrade password hash: {e}")

        display_name = user.get("name") or email.split("@")[0] or "User"
        token = _make_jwt(email, display_name)
        return {
            "status": "success",
            "message": "Login successful",
            "token": token,
            "name": display_name,
            "user": {"email": email, "name": display_name}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Internal server error during login."})

# Mount auth routes (/api/auth/*)
class PredictionChatRequest(BaseModel):
    message: str
    current_prediction: Optional[Dict[str, Any]] = None


@app.post("/api/prediction-chat")
def prediction_chat(req: PredictionChatRequest):
    """
    AI Chatbot directly integrated into the Overview & Prediction section.
    Automatically detects and responds in the exact language used by the user.
    """
    from assistant import _client as _groq_client
    
    context_str = "No prediction context provided."
    if req.current_prediction:
        context_str = f"Current Phase: {req.current_prediction.get('current_phase', 'Unknown')}\n"
        context_str += f"Next Period: {req.current_prediction.get('next_period_date', 'Unknown')}\n"
        context_str += f"Conception Probability: {req.current_prediction.get('conception_probability', 'Unknown')}\n"
        
    system_prompt = f"{BASE_SYSTEM_PROMPT}\n\nIncorporate the following user cycle prediction context if relevant:\n{context_str}"

    if _groq_client is not None:
        for model in ACTIVE_MODELS:
            try:
                resp = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": req.message},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                )
                text = resp.choices[0].message.content
                if text and text.strip():
                    return {"status": "success", "response": text.strip()}
            except Exception as model_err:
                print(f"[PREDICTION-CHAT] {model} failed: {model_err}")
                
        # Try fallback array if ACTIVE_MODELS failed
        fallback_models = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "qwen-2.5-32b"]
        for model in fallback_models:
            if model in ACTIVE_MODELS: continue
            try:
                resp = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": req.message},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                )
                text = resp.choices[0].message.content
                if text and text.strip():
                    return {"status": "success", "response": text.strip()}
            except Exception:
                pass
                
    fallback_text = generate_smart_fallback(req.message, req.current_prediction.get("current_phase", "Unknown") if req.current_prediction else "Unknown")
    return {"status": "success", "response": fallback_text}

app.include_router(auth_router)

# ── Health / Readiness Probe ──────────────────────────────────────────────
@app.get("/api/health")
def api_health():
    """Lightweight readiness probe used by the splash screen polling loop."""
    from datetime import datetime as _dt
    return {
        "status": "ok",
        "version": "3.0.0",
        "timestamp": _dt.utcnow().isoformat() + "Z",
    }


# ── Pydantic Request Models ────────────────────────────────────────────────

class PeriodCareRequest(BaseModel):
    start_date: str = Field(default_factory=lambda: date.today().isoformat())
    current_day: int = 1
    symptoms: List[str] = []
    user_notes: str = ""


# ── AI Period Care & Diet Endpoint ─────────────────────────────────────────
@app.post("/api/ai-period-care")
async def ai_period_care(req: PeriodCareRequest):
    """
    Generates a fully personalised daily period care + diet plan via GroqCloud.
    Model: llama-3.3-70b-versatile (fallback: llama3-8b-8192).
    Returns strict JSON schema with hydration schedule, diet plan, tips, avoids.
    """
    import json as _json
    from assistant import _client as _groq_client

    symptoms_str = ", ".join(req.symptoms) if req.symptoms else "general discomfort"
    notes_str = f"\nAdditional user notes: {req.user_notes}" if req.user_notes.strip() else ""

    PERIOD_CARE_SYSTEM = (
        "You are an empathetic, clinical-level female health & nutrition AI coach for FemCare AI. "
        "Based on the user's cycle day and symptoms, generate a structured, actionable, and comforting "
        "care plan for today. You MUST respond with ONLY valid JSON — no markdown fences, no extra text. "
        "Follow the schema exactly as given."
    )

    user_turn = (
        f"Period start date: {req.start_date}\n"
        f"Today is Day {req.current_day} of the cycle.\n"
        f"Reported symptoms: {symptoms_str}.{notes_str}\n\n"
        "Respond with ONLY this JSON (fill every field with specific, day-appropriate advice):\n"
        "{\n"
        '  "cycle_summary": "Short encouraging message mentioning day number",\n'
        '  "water_target": "X.X Liters",\n'
        '  "hydration_schedule": [\n'
        '    {"time": "8:00 AM",  "amount": "500ml", "type": "Warm lemon water",            "benefit": "Reduces morning bloating"},\n'
        '    {"time": "11:30 AM", "amount": "300ml", "type": "Coconut water / Electrolytes", "benefit": "Replaces lost minerals"},\n'
        '    {"time": "2:30 PM",  "amount": "400ml", "type": "Pure water",                  "benefit": "Aids digestion"},\n'
        '    {"time": "5:30 PM",  "amount": "300ml", "type": "Ginger chamomile tea",        "benefit": "Soothes uterine cramps"},\n'
        '    {"time": "9:00 PM",  "amount": "300ml", "type": "Warm water",                  "benefit": "Eases night muscle tension"}\n'
        '  ],\n'
        '  "diet_plan": {\n'
        '    "breakfast":     "Specific meal + why it helps today\'s symptoms",\n'
        '    "lunch":         "Specific meal + why it helps today\'s symptoms",\n'
        '    "evening_snack": "Mood-boosting snack (e.g. dark chocolate, pumpkin seeds)",\n'
        '    "dinner":        "Light, magnesium-rich dinner for deep sleep"\n'
        '  },\n'
        '  "mood_and_cramp_boosters": [\n'
        '    "Tip 1 (e.g. Child pose stretch 5 mins)",\n'
        '    "Tip 2 (e.g. Heat pad at lower abdomen)",\n'
        '    "Tip 3 (e.g. 10-minute mindfulness breathing)"\n'
        '  ],\n'
        '  "foods_to_avoid_today": ["High sodium foods", "Excess caffeine", "Refined sugary snacks"]\n'
        "}"
    )

    # ── Try primary model, then fallback ─────────────────────────────────────
    raw_text = None
    for model in ACTIVE_MODELS:
        if _groq_client is None:
            break
        try:
            resp = _groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PERIOD_CARE_SYSTEM},
                    {"role": "user",   "content": user_turn},
                ],
                temperature=0.65,
                max_tokens=1200,
            )
            raw_text = resp.choices[0].message.content
            if raw_text and raw_text.strip():
                print(f"[AI-PERIOD-CARE] Got response from {model} ({len(raw_text)} chars)")
                break
        except Exception as model_err:
            print(f"[AI-PERIOD-CARE] {model} failed: {model_err} — trying fallback")

    # ── Parse JSON from AI response ──────────────────────────────────────────
    plan = None
    if raw_text:
        try:
            # Strip any accidental markdown fences
            clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            plan = _json.loads(clean)
        except Exception as parse_err:
            print(f"[AI-PERIOD-CARE] JSON parse error: {parse_err} — using fallback plan")

    # ── Hardcoded fallback (API down / parse fail) ───────────────────────────
    if not plan:
        plan = {
            "cycle_summary": f"Day {req.current_day}: Rest, recover, and nourish yourself today. 💜",
            "water_target": "2.8 Liters",
            "hydration_schedule": [
                {"time": "8:00 AM",  "amount": "500ml", "type": "Warm lemon water",             "benefit": "Reduces morning bloating"},
                {"time": "11:30 AM", "amount": "300ml", "type": "Coconut water / Electrolytes",  "benefit": "Replaces lost minerals"},
                {"time": "2:30 PM",  "amount": "400ml", "type": "Pure water",                   "benefit": "Aids digestion"},
                {"time": "5:30 PM",  "amount": "300ml", "type": "Ginger chamomile tea",         "benefit": "Soothes uterine cramps"},
                {"time": "9:00 PM",  "amount": "300ml", "type": "Warm water",                   "benefit": "Eases night muscle tension"},
            ],
            "diet_plan": {
                "breakfast":     "Oats with banana and flaxseeds — iron + fibre to replenish lost nutrients",
                "lunch":         "Lentil soup with spinach and whole-grain roti — iron, folate & magnesium",
                "evening_snack": "Dark chocolate (70%+) + a handful of pumpkin seeds — magnesium & mood lift",
                "dinner":        "Steamed fish or tofu with sautéed greens + brown rice — light, sleep-friendly",
            },
            "mood_and_cramp_boosters": [
                "Child's pose stretch for 5 minutes — relieves lower back and uterine tension",
                "Apply a heat pad to your lower abdomen for 15–20 minutes",
                "10-minute box breathing (4-4-4-4) to calm the nervous system",
            ],
            "foods_to_avoid_today": [
                "High sodium / processed snacks — worsen bloating",
                "Excess caffeine — amplifies cramps and disrupts sleep",
                "Refined sugary foods — spike and crash mood",
            ],
            "_fallback": True,
        }

    return {"status": "success", "day": req.current_day, "symptoms": req.symptoms, "plan": plan}


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


class ProductRecommendRequest(BaseModel):
    concern: str
    current_phase: str = "Follicular"
    age: int = 25
    lang: Optional[str] = "en"


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
async def chat_assistant(req: AssistantRequest):
    """
    Routes the user query through GroqCloud (multi-model fallback).
    Falls back to the smart KB-driven answer if all models fail.
    Never surfaces API error messages to the frontend.
    """
    from assistant import _client as _groq_client, detect_language

    lang = detect_language(req.query)
    lang_labels = {
        "bengali_script": "Bengali Script (বাংলা)",
        "hindi_script":   "Hindi Script (हिंदी)",
        "banglish":       "Banglish (Latin Script Bengali)",
        "hinglish":       "Hinglish (Latin Script Hindi)",
        "english":        "English",
    }
    target_lang = lang_labels.get(lang, "English")

    user_turn = (
        f"[Clinical Context]\n"
        f"- Target Language & Script: {target_lang}\n"
        f"- Cycle Phase: {req.current_phase}\n"
        f"- Operating Mode: {req.operating_mode}\n"
        f"- User Age: {req.age}\n"
        f"- Stress Level: {req.stress_level}/5\n\n"
        f"User question: {req.query}"
    )

    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

    if _groq_client is not None:
        for model in ACTIVE_MODELS:
            try:
                resp = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_turn},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content
                if text and text.strip():
                    print(f"[CHAT-ENDPOINT] {model} responded ({len(text)} chars)")
                    return {"response": text.strip()}
                print(f"[CHAT-ENDPOINT] {model} returned empty — trying next.")
            except Exception as model_err:
                print(f"[CHAT-ENDPOINT] {model} failed: {model_err} — trying next.")
    else:
        print("[CHAT-ENDPOINT] Groq client not initialised.")

    fallback_text = generate_smart_fallback(req.query, req.current_phase)
    return {"response": fallback_text}


# 6. Smart Product Recommender Endpoint (/api/recommend-product)
@app.post("/api/recommend-product")
def recommend_product(req: ProductRecommendRequest):
    """
    Returns a structured, grounded sanitary product recommendation.
    Combines fast KB rule-matching with Groq/Llama-3 AI enrichment.
    """
    from assistant import _client as _groq_client, detect_language
    
    lang = req.lang or detect_language(req.concern)
    lang_labels = {
        "bengali_script": "Bengali Script (বাংলা)",
        "hindi_script":   "Hindi Script (हिंदी)",
        "banglish":       "Banglish (Latin Script Bengali)",
        "hinglish":       "Hinglish (Latin Script Hindi)",
        "english":        "English",
    }
    target_lang = lang_labels.get(lang, "English")

    user_turn = (
        f"[Sanitary Product Recommender]\n"
        f"User concern: {req.concern}\n"
        f"Target Language & Script: {target_lang}\n"
        f"Current cycle phase: {req.current_phase}\n"
        f"User age: {req.age}\n\n"
        f"STRICT INSTRUCTION: Write a warm, empathetic 3-4 bullet point recommendation strictly in {target_lang}. "
        f"Explain WHY each recommended product helps this specific concern. "
        f"End with a medical disclaimer note in {target_lang}. Be concise and use markdown formatting."
    )

    SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

    if _groq_client is not None:
        for model in ACTIVE_MODELS:
            try:
                resp = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_turn},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content
                if text and text.strip():
                    print(f"[RECOMMEND-ENDPOINT] {model} responded ({len(text)} chars)")
                    return {"response": text.strip(), "ai_response": text.strip()}
                print(f"[RECOMMEND-ENDPOINT] {model} returned empty — trying next.")
            except Exception as model_err:
                print(f"[RECOMMEND-ENDPOINT] {model} failed: {model_err} — trying next.")
    else:
        print("[RECOMMEND-ENDPOINT] Groq client not initialised.")

    fallback_text = generate_smart_fallback(req.concern, req.current_phase)
    return {"response": fallback_text, "ai_response": fallback_text}


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
