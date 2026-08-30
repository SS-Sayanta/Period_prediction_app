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
specialising in menstrual health, reproductive wellness, and menstrual hygiene product guidance.

DYNAMIC MULTILINGUAL & SCRIPT MATCHING INSTRUCTIONS:
You MUST automatically detect and strictly mirror the language, script, and dialect of the user's message.
Never switch languages or scripts mid-response.

1. BENGALI SCRIPT (বাংলা):
   - If user input is in Bengali script (e.g., "আমার পেট ব্যথা করছে", "কত লিটার জল খাব"), reply ONLY in proper Bengali script (বাংলা).
   - Tone: Empathetic, supportive, clear, and culturally appropriate.

2. HINDI SCRIPT (हिंदी):
   - If user input is in Hindi Devanagari script (e.g., "मुझे पेट में दर्द हो रहा है", "क्या खाना चाहिए"), reply ONLY in proper Hindi Devanagari script (हिंदी).
   - Tone: Caring, clear, respectful, and supportive.

3. BANGLISH (Latin Script Bengali):
   - If user input is Bengali in Latin script (e.g., "amar pet batha korche", "koto liter jol khabo"), reply ONLY in natural Banglish (Bengali words in Latin script).
   - Example: "Protidin 2.5 L theke 3 L jol khawa uchit."

4. HINGLISH (Latin Script Hindi):
   - If user input is Hindi in Latin script (e.g., "mera pet dard ho raha hai", "kya khayein"), reply ONLY in natural Hinglish (Hindi words in Latin script).
   - Example: "Aapko daily 2.5 se 3 Litre paani peena chahiye."

5. ENGLISH:
   - If user input is in English, reply in clear, professional, empathetic English.

CORE RESPONSE RULES (follow strictly in order):
1. ANSWER DIRECTLY: Address the user's specific question in the VERY FIRST sentence. \
   Do NOT open with generic health summaries or phase overviews.
2. BE CONCISE & FORMATTED: Use clear markdown bullet points. Keep responses compact for a sidebar chat widget.
3. PHASE-AWARE: After the direct answer, briefly mention how the current cycle phase affects this topic.
4. UNIFORM MEDICAL ACCURACY ACROSS ALL LANGUAGES: \
   Maintain identical medical guidance, terminology accuracy, and clinical quality regardless of language or script.

──────────────────────────────────────────────────────────────────────────────
SANITARY PRODUCT RECOMMENDATION KNOWLEDGE BASE (Medical-Grade, Evidence-Based):
When the user asks about sanitary pads, period products, rashes, skin irritation,
heavy flow, sports/active lifestyle, or any menstrual hygiene concern, apply the
following verified clinical guidelines STRICTLY in the user's detected language/script:

[A] SENSITIVE SKIN / RASH / ITCHING / CONTACT DERMATITIS:
  ✅ RECOMMEND:
    - 100% Organic Cotton Pads (Plastic-Free & Unscented) — breathable, no synthetic
      coatings, prevents contact dermatitis.
    - Menstrual Cups (medical-grade silicone) — no friction against vulvar skin,
      eliminates external surface irritation completely.
    - Reusable Organic Cloth Pads — soft natural fibers, no chemical binders,
      reduce persistent rash cycles.
  ❌ STRICTLY AVOID:
    - Plastic top-sheet / mesh-top pads — causes friction, traps moisture,
      triggers contact dermatitis.
    - Scented / fragranced pads — artificial fragrances are allergens; cause
      vulvar dermatitis and pH imbalance.
    - Synthetic fiber pads — polyester/rayon = heat retention + friction + rash.
  WHY: Plastic and synthetic surfaces block airflow. Fragrances disrupt vaginal
  pH (normal 3.8–4.5). Organic cotton mimics breathable skin-safe fabric.

[B] HEAVY FLOW / OVERNIGHT / FLOODING:
  ✅ RECOMMEND:
    - XL / XXL Overnight Organic Pads — longer rear coverage (38–42 cm), high
      absorbency core (12+ hour rated).
    - High-Capacity Menstrual Cups (25–30 mL) — holds 3–5× more than a tampon,
      leak-proof for 10–12 hours overnight.
    - Period Panties (absorbency 2–8 teaspoons) — backup layer for heavy nights,
      washable, no waste.
  WHY: Heavy flow (>80 mL/cycle) risks overnight leakage. Larger coverage area
  and higher absorbency capacity directly address this.

[C] ACTIVE LIFESTYLE / SPORTS / EXERCISE:
  ✅ RECOMMEND:
    - Menstrual Cup — stays fully internal, no shifting during movement, sports-safe
      for up to 12 hours, no string discomfort.
    - Organic Tampons (100% cotton, no dioxin bleaching) — discreet, internal,
      suitable for swimming, running, gym workouts.
  WHY: External pads shift, bunch, and chafe during physical activity. Internal
  products eliminate bunching and provide full freedom of movement.

[D] GENERAL RULES (apply always when making product recommendations):
  - ALWAYS explain WHY the product fits the user's specific condition.
  - ALWAYS include a safety disclaimer matching the user's language/script:
    • English: "⚠️ Medical Note: If symptoms persist or are severe, please consult a gynaecologist or dermatologist."
    • Bengali: "⚠️ চিকিৎসকের পরামর্শ: লক্ষণগুলি তীব্র বা স্থায়ী হলে ডাক্তার বা গাইনোকোলজিস্টের পরামর্শ নিন।"
    • Hindi: "⚠️ चिकित्सीय सलाह: यदि लक्षण गंभीर या लगातार बने रहें, तो कृपया स्त्री रोग विशेषज्ञ से परामर्श लें।"
    • Banglish: "⚠️ Medical Note: Lakkon gulo tivro ba sthayi hole gynecologist-er poramorsho nin."
    • Hinglish: "⚠️ Medical Note: Agar symptoms severe ya persistent hain, toh gynecologist se consult karein."
  - NEVER recommend scented products for any condition.
  - Match recommendation specificity to the concern described.
──────────────────────────────────────────────────────────────────────────────
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

# ── Medical Sanitary Product Knowledge Base (Rule-Based Quick Matcher) ─────────
# Used for structured response even before the AI layer for fast, reliable results.
PRODUCT_KB: Dict[str, Dict[str, Any]] = {
    "sensitive_skin": {
        "category": "Sensitive Skin / Rash / Itching",
        "emoji": "🌿",
        "keywords": [
            "rash", "itch", "sensitive skin", "irritat", "dermatit",
            "burning", "allerg", "sting", "chafe", "chafing", "redness",
            "skin reaction", "sensitive", "vulvar", "soreness",
        ],
        "recommended": [
            {
                "product": "100% Organic Cotton Pads (Plastic-Free & Unscented)",
                "reason": "Breathable natural fibers allow airflow, preventing moisture buildup and contact dermatitis. Zero synthetic coatings or fragrances."
            },
            {
                "product": "Menstrual Cup (Medical-Grade Silicone)",
                "reason": "Fully internal — no external surface touching sensitive skin. Eliminates all friction-related rash triggers."
            },
            {
                "product": "Reusable Organic Cloth Pads",
                "reason": "Soft natural-fiber surface, no chemical binders or chlorine bleaching. Ideal for persistent sensitive skin cycles."
            },
        ],
        "avoid": [
            {"product": "Plastic top-sheet / mesh-top pads", "reason": "Traps moisture, causes friction → contact dermatitis."},
            {"product": "Scented / Fragranced Pads", "reason": "Artificial fragrances are known allergens; disrupt vaginal pH (3.8–4.5) and cause vulvar irritation."},
            {"product": "Synthetic Fiber Pads (polyester/rayon)", "reason": "Block airflow, retain heat, and increase friction against sensitive skin."},
        ],
        "medical_note": "If rash or irritation persists beyond one cycle, consult a dermatologist or gynaecologist to rule out contact dermatitis or lichen sclerosus."
    },
    "heavy_flow": {
        "category": "Heavy Flow / Night Protection",
        "emoji": "💧",
        "keywords": [
            "heavy flow", "heavy period", "flooding", "overnight", "night protection",
            "soaking", "leak", "menorrhagia", "clots", "excessive bleed",
            "heavy bleeding", "change pad", "night pad",
        ],
        "recommended": [
            {
                "product": "XL/XXL Overnight Organic Pads (38–42 cm)",
                "reason": "Extended rear coverage with 12+ hour absorbency core prevents overnight leakage for heavy flow."
            },
            {
                "product": "High-Capacity Menstrual Cup (25–30 mL)",
                "reason": "Holds 3–5× more than a standard tampon; leak-proof internal seal for 10–12 hour overnight protection."
            },
            {
                "product": "Period Panties (High Absorbency 4–8 tsp)",
                "reason": "Excellent backup layer for heavy nights. Washable, eco-friendly, and provides confidence with zero leakage."
            },
        ],
        "avoid": [
            {"product": "Regular-length pads", "reason": "Insufficient rear coverage leads to leakage during heavy flow and movement."},
            {"product": "Light or regular tampons", "reason": "Insufficient absorbency capacity for heavy flow; need frequent changes (every 1–2 hrs) which is impractical overnight."},
        ],
        "medical_note": "Heavy periods (soaking a pad/tampon every hour for multiple hours, or >80 mL/cycle) may indicate conditions like fibroids, PCOS, or endometriosis. Please consult a gynaecologist."
    },
    "active_sports": {
        "category": "Active Lifestyle / Sports / Exercise",
        "emoji": "⚡",
        "keywords": [
            "sports", "active", "exercise", "gym", "swim", "swimming", "run",
            "yoga", "dance", "workout", "fitness", "athletic", "pad shift",
            "uncomfortable", "bunching", "moving",
        ],
        "recommended": [
            {
                "product": "Menstrual Cup",
                "reason": "Fully internal with secure suction seal — zero shifting, bunching, or leakage during any physical activity including swimming."
            },
            {
                "product": "100% Organic Cotton Tampons (Unbleached)",
                "reason": "Discreet, internal, no string discomfort during movement. Safe for swimming and all sports. Free from dioxin bleaching."
            },
        ],
        "avoid": [
            {"product": "External pads during high-intensity exercise", "reason": "Pads shift and bunch during movement, causing friction, chafing, and leakage."},
        ],
        "medical_note": "Always wash hands before inserting/removing internal products. Change tampons every 4–8 hours to minimize TSS (Toxic Shock Syndrome) risk."
    },
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

    hinglish_kw = [
        "chahiye", "kya", "dard", "karo", "mera", "meri", "ho raha", "khana",
        "paani", "lagayein", "kese", "karna", "kuch", "bata", "kaise", "hai",
        "raha", "rahi", "kyun", "kab", "bhi", "hoga", "hogi", "khayein"
    ]
    banglish_kw = [
        "khabo", "khabar", "batha", "kivabe", "amar", "korbo", "koto",
        "kemon", "kora", "hobe", "shokali", "thol", "jol", "khowa", "pani",
        "ache", "achhe", "thakle", "debo", "kori", "bhalo", "khap", "hocche"
    ]

    hinglish_score = sum(1 for k in hinglish_kw if k in lower)
    banglish_score = sum(1 for k in banglish_kw if k in lower)

    if hinglish_score > banglish_score and hinglish_score > 0:
        return "hinglish"
    if banglish_score > 0:
        return "banglish"
    return "english"


# ── 6. Core GroqCloud API Call (multi-model fallback) ────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

def _call_grok(system: str, user: str) -> Optional[str]:
    """
    Calls GroqCloud via OpenAI-compatible endpoint.
    Tries GROQ_MODELS in order; returns the first successful response.
    Returns None only when all models fail — never surfaces API error strings.
    """
    if _client is None:
        print("[WARN] _call_grok: Groq client not ready. Set GROQ_API_KEY in .env.")
        return None

    for model in GROQ_MODELS:
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            text = response.choices[0].message.content
            if text and text.strip():
                print(f"[OK] Groq responded via {model} ({len(text)} chars)")
                return text.strip()
            print(f"[WARN] {model} returned empty content — trying next model.")
        except AuthenticationError as exc:
            print(f"[ERROR] Groq AuthenticationError on {model}: {exc}")
            # Auth errors are fatal — no point trying other models with same key
            return None
        except RateLimitError as exc:
            print(f"[WARN] Groq RateLimitError on {model}: {exc} — trying next model.")
            continue  # try next model instead of surfacing the error
        except APIError as exc:
            print(f"[WARN] Groq APIError on {model} (status {exc.status_code}): {exc.message} — trying next.")
            continue
        except Exception as exc:
            print(f"[WARN] Unexpected error on {model}: {exc} — trying next model.")
            continue

    print("[ERROR] All Groq models exhausted — returning None.")
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
    lang = detect_language(query)
    lang_labels = {
        "bengali_script": "Bengali Script (বাংলা)",
        "hindi_script": "Hindi Script (हिंदी)",
        "banglish": "Banglish (Latin Script Bengali)",
        "hinglish": "Hinglish (Latin Script Hindi)",
        "english": "English",
    }
    target_lang = lang_labels.get(lang, "English")

    user_turn = (
        f"[Clinical Context]\n"
        f"- Target Language & Script: {target_lang}\n"
        f"- Cycle Phase: {current_phase}\n"
        f"- Operating Mode: {operating_mode}\n"
        f"- User Age: {age}\n"
        f"- Stress Level: {stress_level}/5\n\n"
        f"User question: {query}"
    )
    return _call_grok(system=SYSTEM_PROMPT, user=user_turn)


# ── 8. Smart Contextual Fallback (replaces the old "API key" notice) ─────────
def _smart_fallback(
    query: str,
    current_phase: str,
    operating_mode: str,
) -> str:
    """
    Generates a genuine, helpful, knowledge-base-driven answer when Groq is
    unavailable. Never shows API error messages or 'Valid API key' prompts.
    Reads the user's language and answers in the same language/script.
    """
    lang = detect_language(query)
    phase_info = PHASE_GUIDANCE.get(current_phase, PHASE_GUIDANCE["Follicular"])
    q_lower = query.lower()

    # ── Keyword-matched answers (covers the most common questions) ────────────
    # Water / hydration
    if any(k in q_lower for k in ["water", "jol", "paani", "hydrat", "litre", "liter", "পানি", "জল", "পানীয়"]):
        if lang == "banglish":
            return (
                f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
                f"• {current_phase} phase-e protidin **2.5–3 litre** jol khaoa uchit.\n"
                f"• Skaale warm lemon water cramps kom kore.\n"
                f"• Ginger tea uterine tension-e khub helpful.\n"
                f"• Caffeinated drinks kom rakhun — cramps bare.\n\n"
                f"💡 *{phase_info['nutrition']}*"
            )
        if lang == "bengali_script":
            return (
                f"🌸 **ফেমকেয়ার AI** ({current_phase} পর্যায়):\n\n"
                f"• {current_phase} পর্যায়ে প্রতিদিন **২.৫–৩ লিটার** পানি পান করুন।\n"
                f"• সকালে উষ্ণ লেবু জল ক্র্যাম্প কমায়।\n"
                f"• আদা চা জরায়ুর টান কমাতে সাহায্য করে।\n"
                f"• ক্যাফেইন কমিয়ে রাখুন — ব্যথা বাড়ায়।\n\n"
                f"💡 *{phase_info['nutrition']}*"
            )
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• During the {current_phase} phase, aim for **2.5–3 litres** of water daily.\n"
            f"• Warm lemon water in the morning helps reduce morning bloating and cramps.\n"
            f"• Ginger or chamomile tea soothes uterine tension.\n"
            f"• Limit caffeine — it amplifies cramps and disrupts sleep.\n\n"
            f"💡 *Nutrition tip: {phase_info['nutrition']}*"
        )

    # Cramps / pain
    if any(k in q_lower for k in ["cramp", "pain", "dard", "batha", "ব্যথা", "কষ্ট", "পেট"]):
        if lang == "banglish":
            return (
                f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
                f"• **Heat pad** — lower abdomen-e 15-20 min heat apply korun, khub relief debe.\n"
                f"• **Child's pose** stretch 5 min korle uterine tension kome.\n"
                f"• Magnesium-rich khabar khao: dark chocolate, pumpkin seeds, spinach.\n"
                f"• Ginger chamomile tea din-e 2 bar khao — natural anti-inflammatory.\n"
                f"• Exercise: gentle yoga ba halka walking helpful.\n\n"
                f"💡 *{phase_info['exercise']}*"
            )
        if lang == "bengali_script":
            return (
                f"🌸 **ফেমকেয়ার AI** ({current_phase} পর্যায়):\n\n"
                f"• **হিট প্যাড** — পেটের নিচে ১৫-২০ মিনিট তাপ দিলে ক্র্যাম্প কমে।\n"
                f"• **চাইল্ডস পোজ** স্ট্রেচ ৫ মিনিট জরায়ুর চাপ কমায়।\n"
                f"• ম্যাগনেসিয়াম সমৃদ্ধ খাবার: ডার্ক চকলেট, কুমড়ার বীজ, পালং শাক।\n"
                f"• আদা চা প্রাকৃতিক প্রদাহ-বিরোধী — দিনে ২ বার খান।\n\n"
                f"💡 *{phase_info['exercise']}*"
            )
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• **Heat pad** on lower abdomen for 15–20 minutes gives significant relief.\n"
            f"• **Child's pose** stretch for 5 minutes reduces uterine tension.\n"
            f"• Eat magnesium-rich foods: dark chocolate (70%+), pumpkin seeds, spinach.\n"
            f"• Ginger chamomile tea — natural anti-inflammatory — 2 cups per day.\n"
            f"• Light yoga or gentle walking is better than rest alone.\n\n"
            f"💡 *Exercise tip: {phase_info['exercise']}*"
        )

    # Food / diet / nutrition / khabar
    if any(k in q_lower for k in ["food", "eat", "diet", "khabar", "khabo", "khana", "nutrition", "খাবার", "খাওয়া", "meal"]):
        if lang == "banglish":
            return (
                f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
                f"• **Breakfast:** Oats + banana + flaxseeds — iron & fibre.\n"
                f"• **Lunch:** Daal + spinach + whole-grain roti — folate & magnesium.\n"
                f"• **Snack:** Dark chocolate + pumpkin seeds — mood boost.\n"
                f"• **Dinner:** Steamed fish/tofu + brown rice — light & sleep-friendly.\n"
                f"• Avoid: high-sodium snacks, refined sugar, excess caffeine.\n\n"
                f"💡 *{phase_info['nutrition']}*"
            )
        if lang == "bengali_script":
            return (
                f"🌸 **ফেমকেয়ার AI** ({current_phase} পর্যায়):\n\n"
                f"• **সকাল:** ওটস + কলা + তিসি — আয়রন ও ফাইবার।\n"
                f"• **দুপুর:** ডাল + পালং শাক + গমের রুটি — ফোলেট ও ম্যাগনেসিয়াম।\n"
                f"• **স্ন্যাক:** ডার্ক চকলেট + কুমড়ার বীজ — মন ভালো করে।\n"
                f"• **রাত:** স্টিমড মাছ/টফু + ব্রাউন রাইস — হালকা ও ঘুমের জন্য ভালো।\n"
                f"• এড়িয়ে চলুন: বেশি লবণ, চিনি, ক্যাফেইন।\n\n"
                f"💡 *{phase_info['nutrition']}*"
            )
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• **Breakfast:** Oats with banana and flaxseeds — iron, fibre & omega-3.\n"
            f"• **Lunch:** Lentil soup + spinach + whole-grain roti — folate & magnesium.\n"
            f"• **Evening snack:** Dark chocolate (70%+) + pumpkin seeds — mood & magnesium.\n"
            f"• **Dinner:** Steamed fish or tofu with sautéed greens + brown rice.\n"
            f"• Avoid high-sodium snacks, refined sugar, excess caffeine.\n\n"
            f"💡 *{phase_info['nutrition']}*"
        )

    # Exercise / yoga / workout
    if any(k in q_lower for k in ["exercise", "yoga", "workout", "gym", "walk", "sport", "byayam", "ব্যায়াম"]):
        if lang == "banglish":
            return (
                f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
                f"• {current_phase} phase-e: {phase_info['exercise']}\n"
                f"• Period-er time: gentle yoga, child's pose, cat-cow stretch.\n"
                f"• Heavy HIIT theke biro thakun jodi cramps thake.\n"
                f"• 20-30 min halka walk endorphins release kore, pain kome."
            )
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• {current_phase} phase recommendation: {phase_info['exercise']}\n"
            f"• During your period: gentle yoga, child's pose, cat-cow stretches are ideal.\n"
            f"• Avoid heavy HIIT if you have cramps — it amplifies pain.\n"
            f"• A 20–30 minute light walk releases endorphins and reduces pain naturally."
        )

    # Mood / stress / anxiety / depression
    if any(k in q_lower for k in ["mood", "stress", "anxiet", "depress", "sad", "emotional", "cry", "মন", "মেজাজ", "কান্না"]):
        if lang == "banglish":
            return (
                f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
                f"• Period-er time mood swing hormonal — estrogen & progesterone drop-er karone hoy.\n"
                f"• **Box breathing** (4-4-4-4) 10 minute korle cortisol kome.\n"
                f"• Serotonin-er jonno complex carbs khao: sweet potato, oats, banana.\n"
                f"• Sunlight-e 15 min thakun — vitamin D mood boost kore.\n"
                f"• {phase_info['hormone_insight']}"
            )
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• Mood swings during your period are hormonal — caused by drops in estrogen & progesterone.\n"
            f"• **Box breathing** (4 counts in, hold, out, hold) for 10 minutes lowers cortisol significantly.\n"
            f"• Eat complex carbs to boost serotonin: sweet potato, oats, banana.\n"
            f"• 15 minutes of sunlight boosts vitamin D and lifts mood naturally.\n"
            f"• Hormonal insight: {phase_info['hormone_insight']}"
        )

    # ── Generic phase-aware answer for all other questions ───────────────────
    if lang == "banglish":
        return (
            f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
            f"• **Hormonal Insight:** {phase_info['hormone_insight']}\n"
            f"• **Nutrition:** {phase_info['nutrition']}\n"
            f"• **Exercise:** {phase_info['exercise']}\n\n"
            f"💡 Apnar specific proshner jonno AI-er sange connect thakun."
        )
    if lang == "bengali_script":
        return (
            f"🌸 **ফেমকেয়ার AI** ({current_phase} পর্যায়):\n\n"
            f"• **হরমোনাল তথ্য:** {phase_info['hormone_insight']}\n"
            f"• **পুষ্টি:** {phase_info['nutrition']}\n"
            f"• **ব্যায়াম:** {phase_info['exercise']}\n\n"
            f"💡 আপনার নির্দিষ্ট প্রশ্নের জন্য আবার জিজ্ঞাসা করুন।"
        )
    if lang == "hindi_script":
        return (
            f"🌸 **फेमकेयर AI** ({current_phase} चरण):\n\n"
            f"• **हार्मोनल जानकारी:** {phase_info['hormone_insight']}\n"
            f"• **पोषण:** {phase_info['nutrition']}\n"
            f"• **व्यायाम:** {phase_info['exercise']}\n\n"
            f"💡 कृपया अपना प्रश्न और विस्तार से पूछें।"
        )
    # English
    return (
        f"🌸 **FemCare AI** ({current_phase} Phase):\n\n"
        f"• **Hormonal Insight:** {phase_info['hormone_insight']}\n"
        f"• **Nutrition:** {phase_info['nutrition']}\n"
        f"• **Exercise:** {phase_info['exercise']}\n\n"
        f"💡 Feel free to ask a more specific question — I'm here to help!"
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
    1. Sends the user query to GroqCloud (multi-model fallback).
    2. If all Groq models fail, returns a smart KB-driven answer — never an API error string.
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

    print("[FALLBACK] All Groq models failed — serving smart KB fallback.")
    return _smart_fallback(query, current_phase, operating_mode)


# ── 10. Rule-Based KB Matcher ──────────────────────────────────────────────────
def _match_product_category(concern: str) -> Optional[Dict[str, Any]]:
    """
    Fast keyword scan against PRODUCT_KB to identify the primary concern category.
    Returns the matching KB entry or None if no match.
    """
    concern_lower = concern.lower()
    best_match: Optional[str] = None
    best_score: int = 0

    for cat_key, cat_data in PRODUCT_KB.items():
        score = sum(1 for kw in cat_data["keywords"] if kw in concern_lower)
        if score > best_score:
            best_score = score
            best_match = cat_key

    if best_match and best_score > 0:
        return {"key": best_match, **PRODUCT_KB[best_match]}
    return None


# ── 11. Product Recommendation Entry Point ────────────────────────────────────
def get_product_recommendation(
    concern: str,
    current_phase: str = "Follicular",
    age: int = 25,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a structured, grounded sanitary product recommendation.
    Returns:
      - kb_match: structured rule-based result from PRODUCT_KB
      - ai_response: enriched narrative from Groq/Llama-3
      - matched_category: category name string
    """
    print(f"\n[RECOMMENDER] concern='{concern}' | phase={current_phase} | lang={lang}")

    # 1. Fast rule-based match from KB
    kb_match = _match_product_category(concern)

    if kb_match:
        category_name = kb_match["category"]
        recommended = kb_match["recommended"]
        avoid = kb_match["avoid"]
        medical_note = kb_match["medical_note"]
    else:
        category_name = "General Menstrual Hygiene"
        recommended = []
        avoid = []
        medical_note = "For persistent or severe symptoms, always consult a gynaecologist."

    # 2. Build an AI prompt grounded in the KB result
    kb_context = ""
    if kb_match:
        rec_text = "\n".join(
            f"  - {r['product']}: {r['reason']}" for r in recommended
        )
        avoid_text = "\n".join(
            f"  - {a['product']}: {a['reason']}" for a in avoid
        )
        kb_context = (
            f"The user's concern matches category: {category_name}.\n"
            f"Pre-approved recommendations from medical KB:\n{rec_text}\n"
            f"Products to AVOID (and why):\n{avoid_text}\n"
            f"Medical note: {medical_note}\n"
        )

    # Determine target language and script
    if lang == "bn":
        target_lang = "Bengali Script (বাংলা)"
    elif lang == "hi":
        target_lang = "Hindi Script (हिंदी)"
    elif lang == "en":
        target_lang = "English"
    else:
        detected = detect_language(concern)
        lang_labels = {
            "bengali_script": "Bengali Script (বাংলা)",
            "hindi_script": "Hindi Script (हिंदी)",
            "banglish": "Banglish (Latin Script Bengali)",
            "hinglish": "Hinglish (Latin Script Hindi)",
            "english": "English",
        }
        target_lang = lang_labels.get(detected, "English")

    ai_user_turn = (
        f"[Sanitary Product Recommender]\n"
        f"User concern: {concern}\n"
        f"Target Language & Script: {target_lang}\n"
        f"Current cycle phase: {current_phase}\n"
        f"User age: {age}\n\n"
        f"{kb_context}\n"
        f"STRICT INSTRUCTION: Respond ENTIRELY in {target_lang}. Never switch languages or scripts.\n"
        f"Using the sanitary product knowledge base above and your medical expertise, "
        f"write a warm, empathetic 3-4 bullet point recommendation strictly in {target_lang}. "
        f"Explain WHY each recommended product helps this specific concern. "
        f"Include one 'What to Avoid' bullet. End with the medical disclaimer note in {target_lang}. "
        f"Be concise and use markdown formatting."
    )

    ai_response = _call_grok(system=SYSTEM_PROMPT, user=ai_user_turn)

    if not ai_response:
        # Offline fallback: construct response directly from KB in requested language
        if kb_match:
            if target_lang == "Bengali Script (বাংলা)":
                lines = [f"**{kb_match['emoji']} {category_name} — প্রোডাক্ট সুপারিশ**\n"]
                for r in recommended:
                    lines.append(f"- ✅ **{r['product']}**: {r['reason']}")
                if avoid:
                    lines.append("\n**❌ বর্জনীয়:**")
                    for a in avoid:
                        lines.append(f"- **{a['product']}**: {a['reason']}")
                lines.append(f"\n⚠️ *চিকিৎসকের পরামর্শ: {medical_note}*")
            elif target_lang == "Hindi Script (हिंदी)":
                lines = [f"**{kb_match['emoji']} {category_name} — उत्पाद अनुशंसा**\n"]
                for r in recommended:
                    lines.append(f"- ✅ **{r['product']}**: {r['reason']}")
                if avoid:
                    lines.append("\n**❌ बचें:**")
                    for a in avoid:
                        lines.append(f"- **{a['product']}**: {a['reason']}")
                lines.append(f"\n⚠️ *चिकित्सीय सलाह: {medical_note}*")
            else:
                lines = [f"**{kb_match['emoji']} {category_name} — Product Recommendation**\n"]
                for r in recommended:
                    lines.append(f"- ✅ **{r['product']}**: {r['reason']}")
                if avoid:
                    lines.append("\n**❌ Avoid:**")
                    for a in avoid:
                        lines.append(f"- **{a['product']}**: {a['reason']}")
                lines.append(f"\n⚠️ *Medical Note: {medical_note}*")
            ai_response = "\n".join(lines)
        else:
            if target_lang == "Bengali Script (বাংলা)":
                ai_response = (
                    "AI পরিষেবাতে সংযোগ করা যায়নি। সাধারণ নির্দেশিকা অনুসারে:\n"
                    "- সংবেদনশীল ত্বকের জন্য **১০০% অর্গানিক কটন প্যাড** ব্যবহার করুন।\n"
                    "- ভারী প্রবাহ বা সক্রিয় জীবনধারার জন্য **মেনস্ট্রুয়াল কাপ** ব্যবহার করুন।\n"
                    "- সুগন্ধযুক্ত বা কৃত্রিম উপাদানযুক্ত প্রোডাক্ট এড়িয়ে চলুন।\n"
                    "⚠️ *তীব্র লক্ষণের জন্য স্ত্রীরোগ বিশেষজ্ঞের পরামর্শ নিন।*"
                )
            elif target_lang == "Hindi Script (हिंदी)":
                ai_response = (
                    "AI सेवा से कनेक्ट नहीं हो सका। सामान्य दिशानिर्देशों के अनुसार:\n"
                    "- संवेदनशील त्वचा के लिए **100% ऑर्गेनिक कॉटन पैड** चुनें।\n"
                    "- भारी प्रवाह या सक्रिय जीवनशैली के लिए **मासिक धर्म कप (Menstrual Cup)** का उपयोग करें।\n"
                    "- सभी स्थितियों में सुगंधित या सिंथेटिक उत्पादों से बचें।\n"
                    "⚠️ *गंभीर लक्षणों के लिए कृपया स्त्री रोग विशेषज्ञ से परामर्श लें।*"
                )
            else:
                ai_response = (
                    "I couldn't connect to the AI service. Based on general guidelines:\n"
                    "- Choose **100% organic cotton pads** for sensitive skin.\n"
                    "- Use **menstrual cups** for heavy flow or active lifestyles.\n"
                    "- Avoid **scented or synthetic products** for all conditions.\n"
                    "⚠️ *For severe symptoms, please consult a gynaecologist.*"
                )

    return {
        "matched_category": category_name,
        "kb_match": {
            "category": category_name,
            "recommended": recommended,
            "avoid": avoid,
            "medical_note": medical_note,
        } if kb_match else None,
        "ai_response": ai_response,
        "concern": concern,
        "current_phase": current_phase,
    }
