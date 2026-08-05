"""
FemCare AI – assistant.py
AI Backend: GroqCloud (OpenAI-compatible endpoint)
SDK: openai (v1+)  |  Base URL: https://api.groq.com/openai/v1
Set GROK_API_KEY or GROQ_API_KEY in .env (keys start with gsk_).
No Google/Gemini dependencies.
"""
from __future__ import annotations

import os
import re
import traceback
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# ── 1. Load Environment ────────────────────────────────────────────────────────
load_dotenv(override=True)
# Accept either env var name; GroqCloud keys start with gsk_
_API_KEY: str = (
    os.getenv("GROQ_API_KEY", "").strip()
    or os.getenv("GROK_API_KEY", "").strip()
)

# ── 2. Initialise GroqCloud Client (OpenAI-compatible) ────────────────────────
_client: Optional[OpenAI] = None

if not _API_KEY or _API_KEY in ("your_grok_api_key_here", "your_groq_api_key_here"):
    print("[ERROR] No GroqCloud API key found in .env.")
    print("[ERROR] Add GROQ_API_KEY=gsk_... at https://console.groq.com/keys")
else:
    try:
        _client = OpenAI(
            api_key=_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        print(f"[+] GroqCloud client initialised. Key prefix: {_API_KEY[:8]}...")
    except Exception as exc:
        print(f"[ERROR] Failed to initialise GroqCloud client: {exc}")
        traceback.print_exc()

# ── 3. System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are FemCare AI Companion — an empathetic, expert HealthTech AI assistant \
specialising in menstrual health and reproductive wellness.

CRITICAL RULES (follow strictly, in order):
1. DETECT LANGUAGE: Identify the exact language/script of the user's message \
   (English, Bengali script, Banglish, Hindi script, Hinglish). \
   Reply ONLY in that same language, script, and tone — never switch mid-response.
2. ANSWER DIRECTLY: Address the user's specific question in the FIRST sentence. \
   Do NOT open with generic health summaries or phase overviews.
   Examples:
   - "koto liter jol khabo" → first line: "Protidin 2.5 L theke 3 L jol khawa uchit."
   - "pet batha korche" → first line: immediate cramp relief advice.
   - Any food/symptom/activity question → answer that exact thing first.
3. BE CONCISE & FORMATTED: Use clear markdown bullet points. \
   Keep responses compact for a sidebar chat widget.
4. PHASE-AWARE: After the direct answer, briefly mention how the current cycle \
   phase affects this topic.
5. SAFETY: For severe or unusual symptoms, always recommend consulting a gynaecologist.\
"""

# ── 4. Phase & Mode reference data (offline fallback only) ────────────────────
PHASE_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "Menstrual": {
        "hormone_insight": "Estrogen and progesterone are at their lowest, triggering uterine lining shedding. Rest and recovery are essential.",
        "nutrition": "Iron-rich foods (spinach, lentils, red meat), vitamin C for absorption, and warm magnesium-rich herbal teas.",
        "exercise": "Gentle walking, restorative yoga, and light stretching. Avoid heavy HIIT if cramping.",
    },
    "Follicular": {
        "hormone_insight": "Rising estrogen improves mood, skin elasticity, and insulin sensitivity.",
        "nutrition": "Vibrant, nutrient-dense foods: fermented veggies, lean protein, complex carbs, and flaxseeds.",
        "exercise": "Energy levels surge — ideal for strength training, HIIT, and cardio challenges.",
    },
    "Fertile & Ovulation": {
        "hormone_insight": "Estrogen peaks and testosterone spikes slightly, maximising libido, social energy, and confidence.",
        "nutrition": "Anti-inflammatory foods, folate-rich leafy greens, zinc, berries, and healthy fats (avocado, walnuts).",
        "exercise": "Peak endurance window — ideal for group workouts, dancing, and high-energy cardio.",
    },
    "Luteal": {
        "hormone_insight": "Progesterone promotes calm but can slow digestion. Drop in progesterone signals next period if unfertilised.",
        "nutrition": "Complex carbs (sweet potatoes, oats) to stabilise serotonin; B-vitamins for mood.",
        "exercise": "Steady-state cardio, Pilates, brisk walking, and moderate strength as PMS approaches.",
    },
}

MODE_ADVICE: Dict[str, str] = {
    "Cycle Tracking": "Focus on tracking cycle length consistency, symptom patterns, and lifestyle calibration.",
    "Conception / Ovulation Peak": "Maximise intercourse timing within the 5-day fertile window. Consider daily LH strips and BBT monitoring.",
    "Pregnancy Mode": "Prioritise prenatal vitamins (400 mcg+ folic acid), avoid raw/high-mercury foods, and schedule early prenatal consultations.",
}

# ── 5. Language Detection ──────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Returns 'bengali_script', 'hindi_script', 'banglish', 'hinglish', or 'english'."""
    if not text:
        return "english"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bengali_script"
    if re.search(r"[\u0900-\u097F]", text):
        return "hindi_script"
    lower = text.lower()
    banglish_kw = [
        "khabo", "khabar", "batha", "pet", "kivabe", "amar", "korbo", "koto",
        "kemon", "kora", "hobe", "shokali", "thol", "jol", "khowa", "pani",
        "ache", "achhe", "thakle", "debo", "kori",
    ]
    if any(k in lower for k in banglish_kw):
        return "banglish"
    hinglish_kw = [
        "chahiye", "kya", "dard", "karo", "mera", "ho raha", "khana",
        "paani", "lagayein", "kese", "karna", "kuch", "bata",
    ]
    if any(k in lower for k in hinglish_kw):
        return "hinglish"
    return "english"


# ── 6. Core GroqCloud API Call ────────────────────────────────────────────────
def _call_grok(system: str, user: str) -> Optional[str]:
    """
    Low-level call to GroqCloud via OpenAI-compatible endpoint.
    Returns response text or None. Logs every error to console.
    """
    if _client is None:
        print("[WARN] _call_grok: client not ready — set GROQ_API_KEY in .env.")
        return None
    try:
        response = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.7,
        )
        text = response.choices[0].message.content
        if text:
            print(f"[OK] Groq responded ({len(text)} chars)")
            return text.strip()
        print("[WARN] Groq returned empty content.")
        return None

    except AuthenticationError as exc:
        print(f"[ERROR] Groq AuthenticationError — invalid API key: {exc}")
    except RateLimitError as exc:
        print(f"[ERROR] Groq RateLimitError — quota exceeded: {exc}")
        return (
            "⚠️ **Groq API Rate Limit Reached**\n\n"
            "Your free-tier quota has been exceeded. Please check your plan at "
            "https://console.groq.com or wait a moment before retrying."
        )
    except APIError as exc:
        print(f"[ERROR] Groq APIError (status {exc.status_code}): {exc.message}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error calling Groq: {exc}")
        traceback.print_exc()

    return None


# ── 7. Public Response Function ────────────────────────────────────────────────
def get_gemini_response(
    query: str,
    current_phase: str = "Follicular",
    operating_mode: str = "Cycle Tracking",
    age: int = 25,
    stress_level: int = 3,
) -> Optional[str]:
    """
    Builds the clinical user turn and sends to Grok.
    Function name kept for backward compatibility with main.py imports.
    """
    user_turn = (
        f"[Clinical Context]\n"
        f"- Cycle Phase: {current_phase}\n"
        f"- Operating Mode: {operating_mode}\n"
        f"- User Age: {age}\n"
        f"- Stress Level: {stress_level}/5\n\n"
        f"User question: {query}"
    )
    return _call_grok(system=SYSTEM_PROMPT, user=user_turn)


# ── 8. Offline Fallback ───────────────────────────────────────────────────────
def _offline_fallback(
    query: str,
    current_phase: str,
    operating_mode: str,
) -> str:
    """
    Shown ONLY when the Grok API is completely unreachable.
    Language-aware; does NOT attempt to answer the specific question.
    """
    lang = detect_language(query)
    phase_info = PHASE_GUIDANCE.get(current_phase, PHASE_GUIDANCE["Follicular"])
    mode_text = MODE_ADVICE.get(operating_mode, MODE_ADVICE["Cycle Tracking"])

    if lang == "banglish":
        return (
            f"🌸 **FemCare AI (Offline — {current_phase} Phase):**\n\n"
            "⚠️ *Grok AI connect hote parche na. GROK_API_KEY ta .env-e thik ache ki na check korun.*\n\n"
            f"• **Hormonal Insight:** {phase_info['hormone_insight']}\n"
            f"• **Nutrition:** {phase_info['nutrition']}\n"
            f"• **Exercise:** {phase_info['exercise']}\n\n"
            "💬 *Valid API key thakle apnar proshner direct uttor paben!*"
        )
    if lang == "bengali_script":
        return (
            f"🌸 **ফেমকেয়ার AI (অফলাইন — {current_phase} পর্যায়):**\n\n"
            "⚠️ *Grok AI সংযুক্ত হতে পারছে না। GROK_API_KEY যাচাই করুন।*\n\n"
            f"• **হরমোনাল তথ্য:** {phase_info['hormone_insight']}\n"
            f"• **পুষ্টি:** {phase_info['nutrition']}\n"
            "💬 *সংযোগ ফিরলে আপনার প্রশ্নের সরাসরি উত্তর পাবেন!*"
        )
    if lang == "hindi_script":
        return (
            f"🌸 **फेमकेयर AI (ऑफलाइन — {current_phase} चरण):**\n\n"
            "⚠️ *Grok AI कनेक्ट नहीं हो पा रही। GROK_API_KEY जांचें।*\n\n"
            f"• **हार्मोनल जानकारी:** {phase_info['hormone_insight']}\n"
            f"• **पोषण:** {phase_info['nutrition']}\n"
            "💬 *कनेक्शन होने पर आपके सवाल का सीधा जवाब मिलेगा!*"
        )
    if lang == "hinglish":
        return (
            f"🌸 **FemCare AI (Offline — {current_phase} Phase):**\n\n"
            "⚠️ *Grok AI connect nahi ho pa rahi. GROK_API_KEY check karein.*\n\n"
            f"• **Hormonal Insight:** {phase_info['hormone_insight']}\n"
            f"• **Nutrition:** {phase_info['nutrition']}\n"
            "💬 *Connection hone par seedha jawab milega!*"
        )
    # English default
    return (
        f"🌸 **FemCare AI (Offline — {current_phase} Phase):**\n\n"
        "⚠️ *xAI Grok API is unreachable. Verify your GROK_API_KEY in .env.*\n"
        "Get your key at: https://console.x.ai/\n\n"
        f"• **Hormonal Insight:** {phase_info['hormone_insight']}\n"
        f"• **Nutrition:** {phase_info['nutrition']}\n"
        f"• **Exercise:** {phase_info['exercise']}\n"
        f"• **Mode Focus:** {mode_text}\n\n"
        "💬 *Once reconnected, Grok will answer your specific question directly!*"
    )


# ── 9. Public Entry Point (called by main.py) ──────────────────────────────────
def get_ai_assistant_response(
    query: str,
    current_phase: str = "Follicular",
    operating_mode: str = "Cycle Tracking",
    age: int = 25,
    stress_level: int = 3,
) -> str:
    """
    Primary entry point called by FastAPI routes /api/assistant and /api/chat.
    1. Sends the exact user query to xAI Grok — no keyword interception.
    2. Falls back to an offline notice only if the API is unreachable.
    """
    print(f"\n[CHAT] query='{query}' | phase={current_phase} | mode={operating_mode}")

    result = get_gemini_response(
        query=query,
        current_phase=current_phase,
        operating_mode=operating_mode,
        age=age,
        stress_level=stress_level,
    )

    if result:
        return result

    print("[FALLBACK] Grok unavailable — serving offline message.")
    return _offline_fallback(query, current_phase, operating_mode)
