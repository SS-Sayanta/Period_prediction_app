"""
auth_router.py — FemCare AI Authentication Router
Endpoints:
  POST /api/auth/send-otp          → send registration OTP to email
  POST /api/auth/verify-otp        → verify OTP + create account + return JWT
  POST /api/auth/login             → password login + return JWT
  POST /api/auth/forgot-password   → send password-reset OTP
  POST /api/auth/reset-password    → verify reset OTP + update password
  GET  /api/auth/me                → return current user info from JWT
"""

from __future__ import annotations

import os
import json
import random
import smtplib
import secrets
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET   = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALG      = "HS256"
JWT_EXPIRE_H = 720          # 30 days

SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")
SMTP_FROM    = os.getenv("SMTP_FROM", SMTP_USER or "noreply@femcare.ai")

USERS_FILE   = Path("data/users.json")
OTP_TTL_S    = 120          # OTP expires after 2 minutes

# ── Helpers ───────────────────────────────────────────────────────────────────
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
router   = APIRouter(prefix="/api/auth", tags=["auth"])

# In-memory OTP store: { email: {otp, expires_at, purpose} }
_otp_store: Dict[str, Dict[str, Any]] = {}


def _load_users() -> Dict[str, Any]:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_users(users: Dict[str, Any]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def _gen_otp() -> str:
    return str(random.randint(100000, 999999))


def _make_jwt(email: str, name: str) -> str:
    payload = {
        "sub":   email,
        "name":  name,
        "iat":   datetime.utcnow(),
        "exp":   datetime.utcnow() + timedelta(hours=JWT_EXPIRE_H),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _decode_jwt(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}")


def _send_email(to: str, subject: str, html_body: str) -> bool:
    """Send email via SMTP. Returns True on success, False if SMTP not configured."""
    if not SMTP_USER or not SMTP_PASS:
        print(f"[AUTH] SMTP not configured. OTP email to {to} not sent. Check console.")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_FROM
        msg["To"]      = to
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.sendmail(SMTP_FROM, [to], msg.as_string())
        return True
    except Exception as e:
        print(f"[AUTH] SMTP error: {e}")
        return False


def _otp_email_html(otp: str, purpose: str = "registration") -> str:
    action = "complete your registration" if purpose == "registration" else "reset your password"
    return f"""
<!DOCTYPE html>
<html>
<body style="background:#0b0e17;font-family:Inter,sans-serif;margin:0;padding:40px 20px;">
  <div style="max-width:480px;margin:0 auto;background:linear-gradient(135deg,#1a0a2e,#0d0d1a);
              border:1px solid rgba(236,72,153,0.25);border-radius:20px;padding:40px 36px;text-align:center;">
    <div style="font-size:42px;margin-bottom:16px;">❤️</div>
    <h1 style="color:#f9a8d4;font-size:22px;font-weight:800;margin:0 0 8px;">FemCare AI</h1>
    <p style="color:#94a3b8;font-size:14px;margin:0 0 32px;">Your OTP to {action}:</p>
    <div style="background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.3);
                border-radius:14px;padding:24px;margin-bottom:24px;">
      <span style="font-size:40px;font-weight:800;letter-spacing:10px;
                   background:linear-gradient(90deg,#ec4899,#a855f7);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{otp}</span>
    </div>
    <p style="color:#64748b;font-size:12px;">This code expires in 2 minutes. Do not share it with anyone.</p>
    <p style="color:#475569;font-size:11px;margin-top:24px;">— FemCare AI Team 💜</p>
  </div>
</body>
</html>"""


# ── Pydantic Models ───────────────────────────────────────────────────────────
class SendOTPRequest(BaseModel):
    email: str
    name:  Optional[str] = ""
    purpose: str = "registration"   # "registration" | "reset"


class VerifyOTPRequest(BaseModel):
    email:    str
    otp:      str
    password: Optional[str] = None  # required for registration


class LoginRequest(BaseModel):
    email:    str
    password: str


class ResetPasswordRequest(BaseModel):
    email:       str
    otp:         str
    new_password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/send-otp")
def send_otp(req: SendOTPRequest):
    """Generate & send a 6-digit OTP. Works for both registration and password reset."""
    users = _load_users()

    if req.purpose == "registration" and req.email in users:
        raise HTTPException(status_code=409, detail="An account with this email already exists. Please log in.")

    if req.purpose == "reset" and req.email not in users:
        raise HTTPException(status_code=404, detail="No account found with this email address.")

    otp = _gen_otp()
    _otp_store[req.email] = {
        "otp":        otp,
        "expires_at": time.time() + OTP_TTL_S,
        "purpose":    req.purpose,
        "name":       req.name or "",
    }

    subject = "🔐 FemCare AI — Your Verification Code"
    sent    = _send_email(req.email, subject, _otp_email_html(otp, req.purpose))

    # Always print to server console (fallback for dev / no SMTP)
    print(f"[AUTH] OTP for {req.email} ({req.purpose}): {otp}")

    return {
        "success": True,
        "email_sent": sent,
        "message": f"OTP sent to {req.email}. Check your inbox (or server console in dev mode).",
        # DEV-ONLY: remove in production
        "dev_otp": otp if not sent else None,
    }


@router.post("/verify-otp")
def verify_otp(req: VerifyOTPRequest):
    """Verify OTP → create account (registration) or confirm identity (reset step 1)."""
    record = _otp_store.get(req.email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP was requested for this email.")
    if time.time() > record["expires_at"]:
        _otp_store.pop(req.email, None)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if record["otp"] != req.otp.strip():
        raise HTTPException(status_code=400, detail="Incorrect OTP. Please try again.")

    _otp_store.pop(req.email, None)  # consume OTP

    if record["purpose"] == "registration":
        if not req.password or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        users = _load_users()
        users[req.email] = {
            "email":      req.email,
            "name":       record["name"],
            "password":   pwd_ctx.hash(req.password),
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_users(users)
        token = _make_jwt(req.email, record["name"])
        return {"success": True, "token": token, "name": record["name"], "email": req.email}

    # purpose == "reset" — mark email as OTP-verified for the reset step
    _otp_store[f"reset_verified_{req.email}"] = {"verified": True, "expires_at": time.time() + 300}
    return {"success": True, "message": "Identity verified. You may now set a new password."}


@router.post("/login")
def login(req: LoginRequest):
    users = _load_users()
    user  = users.get(req.email)
    if not user or not pwd_ctx.verify(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = _make_jwt(req.email, user["name"])
    return {"success": True, "token": token, "name": user["name"], "email": req.email}


@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    """Verify reset OTP + update password in one step (combined for simplicity)."""
    # Check OTP
    record = _otp_store.get(req.email)
    # Also accept the pre-verified token from a two-step flow
    pre = _otp_store.get(f"reset_verified_{req.email}")

    if not record and not pre:
        raise HTTPException(status_code=400, detail="No reset session found. Please restart the process.")

    if record:
        if time.time() > record["expires_at"]:
            _otp_store.pop(req.email, None)
            raise HTTPException(status_code=400, detail="OTP has expired.")
        if record["otp"] != req.otp.strip():
            raise HTTPException(status_code=400, detail="Incorrect OTP.")
        _otp_store.pop(req.email, None)

    if pre:
        if time.time() > pre["expires_at"]:
            _otp_store.pop(f"reset_verified_{req.email}", None)
            raise HTTPException(status_code=400, detail="Reset session expired. Please restart.")
        _otp_store.pop(f"reset_verified_{req.email}", None)

    if not req.new_password or len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    users = _load_users()
    if req.email not in users:
        raise HTTPException(status_code=404, detail="Account not found.")

    users[req.email]["password"] = pwd_ctx.hash(req.new_password)
    _save_users(users)
    return {"success": True, "message": "Password updated successfully. You can now log in."}


@router.get("/me")
def get_me(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token   = authorization.split(" ", 1)[1]
    payload = _decode_jwt(token)
    return {"email": payload["sub"], "name": payload.get("name", ""), "success": True}
