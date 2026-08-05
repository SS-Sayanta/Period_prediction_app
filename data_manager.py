"""
data_manager.py — Thread-Safe Data Persistence & Table Formatting for FemCare AI.
Provides CSV storage routines and clean DataFrame formatting for Streamlit views.
"""

from __future__ import annotations
import os
import csv
import logging
import threading
from datetime import datetime, date
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

USER_HISTORY_FILE = "user_history.csv"
SYMPTOMS_LOG_FILE = "symptoms_log.csv"
VERIFIED_FEEDBACK_FILE = "verified_feedback.csv"
DATASET_ADVANCED_FILE = "period_dataset_advanced.csv"
DATASET_CORE_FILE = "dataset.csv"

_csv_lock = threading.Lock()

USER_HISTORY_HEADERS = [
    "user_id", "user_name", "last_period_date", "predicted_cycle_days", "recorded_at"
]

SYMPTOMS_LOG_HEADERS = [
    "user_id", "log_date", "cramps", "bloating", "mood", "energy", 
    "headache", "fatigue", "flow_level", "notes", "recorded_at"
]

VERIFIED_FEEDBACK_HEADERS = [
    "user_id", "feedback_date", "actual_cycle_length", "predicted_cycle_length",
    "error_days", "age", "weight_kg", "stress_level", "avg_sleep_hours",
    "exercise_hours", "water_intake_liters", "mood_score", "last_cycle_length",
    "cycle_length_2", "cycle_length_3", "cycle_avg", "cycle_variability", "recorded_at"
]


def init_and_clean_files() -> None:
    """Ensure CSV files exist and adhere strictly to standardized headers."""
    with _csv_lock:
        _ensure_or_repair_csv(
            USER_HISTORY_FILE, 
            USER_HISTORY_HEADERS, 
            default_row={
                "user_id": "demo-user-001",
                "user_name": "Demo User",
                "last_period_date": date.today().strftime("%Y-%m-%d"),
                "predicted_cycle_days": "28.0",
                "recorded_at": datetime.now().isoformat()
            }
        )

        _ensure_or_repair_csv(
            SYMPTOMS_LOG_FILE,
            SYMPTOMS_LOG_HEADERS,
            default_row={
                "user_id": "demo-user-001",
                "log_date": date.today().strftime("%Y-%m-%d"),
                "cramps": "2",
                "bloating": "1",
                "mood": "4",
                "energy": "3",
                "headache": "0",
                "fatigue": "1",
                "flow_level": "Medium",
                "notes": "Feeling energized today",
                "recorded_at": datetime.now().isoformat()
            }
        )

        _ensure_or_repair_csv(
            VERIFIED_FEEDBACK_FILE,
            VERIFIED_FEEDBACK_HEADERS,
            default_row=None
        )


def _ensure_or_repair_csv(file_path: str, expected_headers: List[str], default_row: Optional[Dict[str, str]] = None) -> None:
    need_reset = False
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        need_reset = True
    else:
        try:
            df = pd.read_csv(file_path)
            if list(df.columns) != expected_headers:
                need_reset = True
        except Exception:
            need_reset = True

    if need_reset:
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(expected_headers)
                if default_row:
                    writer.writerow([default_row.get(h, "") for h in expected_headers])
        except Exception as err:
            logger.error(f"Error resetting {file_path}: {err}")


def save_user_history(user_id: str, user_name: str, last_period_date: Any, predicted_cycle_days: float) -> bool:
    init_and_clean_files()
    date_str = last_period_date.strftime("%Y-%m-%d") if isinstance(last_period_date, (date, datetime)) else str(last_period_date)
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with _csv_lock:
        try:
            with open(USER_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([user_id, user_name, date_str, round(float(predicted_cycle_days), 1), recorded_at])
            return True
        except Exception as e:
            logger.error(f"Failed to save user history: {e}")
            return False


def save_symptoms_log(
    user_id: str,
    log_date: Any,
    cramps: int = 0,
    bloating: int = 0,
    mood: int = 3,
    energy: int = 3,
    headache: int = 0,
    fatigue: int = 0,
    flow_level: str = "None",
    notes: str = ""
) -> bool:
    init_and_clean_files()
    date_str = log_date.strftime("%Y-%m-%d") if isinstance(log_date, (date, datetime)) else str(log_date)
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _csv_lock:
        try:
            with open(SYMPTOMS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    user_id, date_str, cramps, bloating, mood, energy,
                    headache, fatigue, flow_level, notes, recorded_at
                ])
            return True
        except Exception as e:
            logger.error(f"Failed to save symptoms log: {e}")
            return False


def save_verified_feedback(
    user_id: str,
    feedback_date: Any,
    actual_cycle_length: float,
    predicted_cycle_length: float,
    feature_dict: Dict[str, Any]
) -> bool:
    init_and_clean_files()
    fb_date_str = feedback_date.strftime("%Y-%m-%d") if isinstance(feedback_date, (date, datetime)) else str(feedback_date)
    error_days = round(actual_cycle_length - predicted_cycle_length, 1)
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        user_id,
        fb_date_str,
        round(float(actual_cycle_length), 1),
        round(float(predicted_cycle_length), 1),
        error_days,
        feature_dict.get("age", 25),
        feature_dict.get("weight_kg", 60.0),
        feature_dict.get("stress_level", 3),
        feature_dict.get("avg_sleep_hours", 7.5),
        feature_dict.get("exercise_hours", 3.0),
        feature_dict.get("water_intake_liters", 2.0),
        feature_dict.get("mood_score", 3),
        feature_dict.get("last_cycle_length", 28.0),
        feature_dict.get("cycle_length_2", 28.0),
        feature_dict.get("cycle_length_3", 28.0),
        feature_dict.get("cycle_avg", 28.0),
        feature_dict.get("cycle_variability", 0.0),
        recorded_at
    ]

    with _csv_lock:
        try:
            with open(VERIFIED_FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return True
        except Exception as e:
            logger.error(f"Failed to save verified feedback: {e}")
            return False


def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame: round floats to 1 decimal, format dates, and truncate UUIDs."""
    if df.empty:
        return df

    out_df = df.copy()

    # Truncate user_id strings
    if "user_id" in out_df.columns:
        out_df["user_id"] = out_df["user_id"].astype(str).apply(lambda x: f"{x[:8]}..." if len(x) > 12 else x)

    # Format numeric float columns
    float_cols = out_df.select_dtypes(include=["float", "float64"]).columns
    for c in float_cols:
        out_df[c] = out_df[c].round(1)

    # Format datetime columns
    datetime_candidates = ["recorded_at", "log_date", "feedback_date", "last_period_date"]
    for c in datetime_candidates:
        if c in out_df.columns:
            out_df[c] = pd.to_datetime(out_df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")

    return out_df


def load_user_history() -> pd.DataFrame:
    init_and_clean_files()
    with _csv_lock:
        try:
            df = pd.read_csv(USER_HISTORY_FILE)
            return format_dataframe_for_display(df)
        except Exception as e:
            logger.error(f"Error loading user history: {e}")
            return pd.DataFrame(columns=USER_HISTORY_HEADERS)


def load_symptoms_log() -> pd.DataFrame:
    init_and_clean_files()
    with _csv_lock:
        try:
            df = pd.read_csv(SYMPTOMS_LOG_FILE)
            return format_dataframe_for_display(df)
        except Exception as e:
            logger.error(f"Error loading symptoms log: {e}")
            return pd.DataFrame(columns=SYMPTOMS_LOG_HEADERS)


def load_verified_feedback() -> pd.DataFrame:
    init_and_clean_files()
    with _csv_lock:
        try:
            df = pd.read_csv(VERIFIED_FEEDBACK_FILE)
            return format_dataframe_for_display(df)
        except Exception as e:
            logger.error(f"Error loading verified feedback: {e}")
            return pd.DataFrame(columns=VERIFIED_FEEDBACK_HEADERS)


def load_reference_dataset() -> Optional[pd.DataFrame]:
    for fname in [DATASET_ADVANCED_FILE, DATASET_CORE_FILE]:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname)
                return format_dataframe_for_display(df)
            except Exception as e:
                logger.error(f"Error loading dataset {fname}: {e}")
    return None
