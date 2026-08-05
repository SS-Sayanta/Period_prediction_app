"""
ml_pipeline.py — Machine Learning Inference & Biological Cycle Engine for FemCare AI.
Provides joblib deserialization, feature pre-processing, silent rule-based fallback,
and precise phase date computations.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

EXPECTED_FEATURES = [
    'age', 
    'weight_kg', 
    'stress_level', 
    'avg_sleep_hours', 
    'exercise_hours', 
    'water_intake_liters', 
    'mood_score', 
    'last_cycle_length', 
    'cycle_length_2', 
    'cycle_length_3', 
    'cycle_avg', 
    'cycle_variability'
]

MODEL_PATH = "period_model.pkl"


@dataclass
class CyclePredictionResult:
    predicted_cycle_days: float
    next_period_start: date
    next_period_end: date
    ovulation_date: date
    fertile_window_start: date
    fertile_window_end: date
    follicular_start: date
    follicular_end: date
    luteal_start: date
    luteal_end: date
    current_phase: str
    days_until_next_period: int
    current_cycle_day: int
    used_ml_model: bool
    confidence_score: float
    model_notes: str
    feature_summary: Dict[str, Any]


class ModelManager:
    """Manages ML model loading state and fallback inference."""
    _instance: Optional['ModelManager'] = None

    def __init__(self, model_file: str = MODEL_PATH):
        self.model_file = model_file
        self.model = None
        self.feature_names: List[str] = EXPECTED_FEATURES
        self.is_loaded: bool = False
        self.load_error: str = ""
        self._load_model()

    @classmethod
    def get_instance(cls, model_file: str = MODEL_PATH) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = ModelManager(model_file)
        return cls._instance

    def _load_model(self) -> None:
        if not os.path.exists(self.model_file):
            self.load_error = f"Model file '{self.model_file}' not found."
            self.is_loaded = False
            return

        try:
            loaded_obj = joblib.load(self.model_file)
            if isinstance(loaded_obj, tuple) and len(loaded_obj) >= 1:
                self.model = loaded_obj[0]
                if len(loaded_obj) >= 2 and isinstance(loaded_obj[1], (list, np.ndarray)):
                    self.feature_names = list(loaded_obj[1])
            elif hasattr(loaded_obj, 'predict'):
                self.model = loaded_obj
            else:
                raise ValueError("Unrecognized ML model structure.")

            if hasattr(self.model, 'predict'):
                self.is_loaded = True
                self.load_error = ""
                logger.info(f"Loaded ML model successfully from {self.model_file}")
            else:
                raise AttributeError("Loaded model object lacks '.predict()' method.")

        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)
            logger.error(f"Failed to load ML model: {exc}")


def calculate_derived_cycle_stats(cycle1: float, cycle2: float, cycle3: float) -> Tuple[float, float]:
    valid_cycles = [float(c) for c in [cycle1, cycle2, cycle3] if c is not None and 10.0 < float(c) < 90.0]
    if not valid_cycles:
        return 28.0, 0.0
    avg = float(np.mean(valid_cycles))
    var = float(np.std(valid_cycles, ddof=1)) if len(valid_cycles) > 1 else 0.0
    return round(avg, 2), round(var, 2)


def predict_cycle_length(
    age: int = 25,
    weight_kg: float = 60.0,
    stress_level: int = 3,
    avg_sleep_hours: float = 7.5,
    exercise_hours: float = 3.0,
    water_intake_liters: float = 2.0,
    mood_score: int = 3,
    last_cycle_length: float = 28.0,
    cycle_length_2: float = 28.0,
    cycle_length_3: float = 28.0,
) -> Tuple[float, bool, float, str, Dict[str, Any]]:
    manager = ModelManager.get_instance()
    
    cycle_avg, cycle_variability = calculate_derived_cycle_stats(
        last_cycle_length, cycle_length_2, cycle_length_3
    )

    feature_dict = {
        'age': float(age),
        'weight_kg': float(weight_kg),
        'stress_level': float(stress_level),
        'avg_sleep_hours': float(avg_sleep_hours),
        'exercise_hours': float(exercise_hours),
        'water_intake_liters': float(water_intake_liters),
        'mood_score': float(mood_score),
        'last_cycle_length': float(last_cycle_length),
        'cycle_length_2': float(cycle_length_2),
        'cycle_length_3': float(cycle_length_3),
        'cycle_avg': float(cycle_avg),
        'cycle_variability': float(cycle_variability)
    }

    if manager.is_loaded and manager.model is not None:
        try:
            input_df = pd.DataFrame([feature_dict])[manager.feature_names]
            raw_pred = manager.model.predict(input_df)[0]
            predicted_days = float(np.clip(raw_pred, 20.0, 45.0))
            
            confidence = max(0.65, min(0.98, 0.95 - (cycle_variability * 0.04) - (abs(stress_level - 3) * 0.03)))
            notes = "RandomForest ML Model Inference Success"
            
            return predicted_days, True, round(confidence, 2), notes, feature_dict

        except Exception as exc:
            logger.warning(f"ML Model inference exception: {exc}. Transitioning to Silent Moving Average Fallback.")

    # ── SILENT RULE-BASED FALLBACK ENGINE ──
    valid_cycles = [c for c in [last_cycle_length, cycle_length_2, cycle_length_3] if 15.0 < c < 60.0]
    base_days = float(np.median(valid_cycles)) if valid_cycles else 28.0
    stress_adj = (stress_level - 3) * 0.35
    predicted_days = float(np.clip(base_days + stress_adj, 21.0, 38.0))
    
    confidence = 0.75 if valid_cycles else 0.60
    notes = "Silent Moving Average Rule-Based Engine"

    return predicted_days, False, confidence, notes, feature_dict


def compute_cycle_phases(
    last_period_date: date,
    predicted_cycle_days: float,
    period_duration: int = 5,
    today_date: Optional[date] = None
) -> CyclePredictionResult:
    if today_date is None:
        today_date = date.today()

    pred_days_int = int(round(predicted_cycle_days))
    next_period_start = last_period_date + timedelta(days=pred_days_int)
    next_period_end = next_period_start + timedelta(days=period_duration - 1)

    ovulation_date = next_period_start - timedelta(days=14)
    fertile_window_start = ovulation_date - timedelta(days=4)
    fertile_window_end = ovulation_date + timedelta(days=1)

    follicular_start = last_period_date
    follicular_end = ovulation_date - timedelta(days=1)

    luteal_start = ovulation_date
    luteal_end = next_period_start - timedelta(days=1)

    days_until_next = (next_period_start - today_date).days
    current_cycle_day = (today_date - last_period_date).days + 1

    if last_period_date <= today_date < (last_period_date + timedelta(days=period_duration)):
        current_phase = "Menstrual Phase (Period)"
    elif fertile_window_start <= today_date <= fertile_window_end:
        if today_date == ovulation_date:
            current_phase = "Ovulation Day ⭐"
        else:
            current_phase = "Fertile Window 🌿"
    elif follicular_start <= today_date < ovulation_date:
        current_phase = "Follicular Phase"
    elif ovulation_date <= today_date < next_period_start:
        current_phase = "Luteal Phase"
    else:
        current_phase = "Upcoming Cycle Transition"

    pred_days, used_ml, confidence, notes, feature_dict = predict_cycle_length(
        last_cycle_length=predicted_cycle_days
    )

    return CyclePredictionResult(
        predicted_cycle_days=round(predicted_cycle_days, 1),
        next_period_start=next_period_start,
        next_period_end=next_period_end,
        ovulation_date=ovulation_date,
        fertile_window_start=fertile_window_start,
        fertile_window_end=fertile_window_end,
        follicular_start=follicular_start,
        follicular_end=follicular_end,
        luteal_start=luteal_start,
        luteal_end=luteal_end,
        current_phase=current_phase,
        days_until_next_period=days_until_next,
        current_cycle_day=current_cycle_day,
        used_ml_model=used_ml,
        confidence_score=confidence,
        model_notes=notes,
        feature_summary=feature_dict
    )
