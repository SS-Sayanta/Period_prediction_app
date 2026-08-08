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

SMTP_SERVER     = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL    = os.getenv("SMTP_EMAIL", os.getenv("SENDER_EMAIL", ""))
SENDER_PASSWORD = os.getenv("SMTP_APP_PASSWORD", os.getenv("SMTP_PASSWORD", os.getenv("SENDER_PASSWORD", "")))
SMTP_FROM       = os.getenv("SMTP_FROM", SENDER_EMAIL or "noreply@femcare.ai")

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
    """Send email via Gmail SMTP (STARTTLS/SSL). Returns True on success."""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print(
            f"\n{'='*55}\n"
            f"[AUTH] ⚠️  SMTP NOT CONFIGURED — email to {to} skipped.\n"
            f"  Add to .env:\n"
            f"    SMTP_EMAIL=your-gmail@gmail.com\n"
            f"    SMTP_APP_PASSWORD=your-16-char-app-password\n"
            f"{'='*55}\n"
        )
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"FemCare AI <{SMTP_FROM}>"
        msg["To"]      = to
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        if SMTP_PORT == 465:
            # Direct SSL connection (Port 465)
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=15) as s:
                s.login(SENDER_EMAIL, SENDER_PASSWORD)
                s.sendmail(SENDER_EMAIL, [to], msg.as_string())
        else:
            # STARTTLS upgrade (Port 587 — Gmail default)
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                s.login(SENDER_EMAIL, SENDER_PASSWORD)
                s.sendmail(SENDER_EMAIL, [to], msg.as_string())

        print(f"[AUTH] ✅  Email sent successfully to {to}")
        return True
    except smtplib.SMTPAuthenticationError:
        print(
            f"[AUTH] ❌  Gmail authentication failed for '{SENDER_EMAIL}'.\n"
            f"  → Make sure you are using a Gmail App Password, NOT your regular password.\n"
            f"  → Generate one at: https://myaccount.google.com/apppasswords"
        )
        return False
    except smtplib.SMTPConnectError:
        print(f"[AUTH] ❌  Could not connect to {SMTP_SERVER}:{SMTP_PORT}. Check network or firewall.")
        return False
    except Exception as e:
        print(f"[AUTH] ❌  SMTP error: {type(e).__name__}: {e}")
        return False


def _is_phone_number(val: str) -> bool:
    # Remove spacing and separators
    if "@" in val:
        return False
    digits = [c for c in val if c.isdigit()]
    return len(digits) >= 8


def _send_sms(to: str, body: str) -> bool:
    """Send SMS via Twilio or Fast2SMS if configured, otherwise print to console."""
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY")

    # Clean the phone number
    cleaned_to = "".join(c for c in to if c.isdigit() or c == "+")
    if not cleaned_to.startswith("+") and len(cleaned_to) == 10:
        cleaned_to = f"+91{cleaned_to}"

    # Try Twilio first
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=cleaned_to
            )
            print(f"[AUTH] Twilio SMS sent to {cleaned_to}. SID: {message.sid}")
            return True
        except ImportError:
            print("[AUTH] twilio package is not installed. Skipping Twilio.")
        except Exception as e:
            print(f"[AUTH] Twilio SMS error: {e}")

    # Try Fast2SMS next
    if FAST2SMS_API_KEY:
        try:
            import requests
            url = "https://www.fast2sms.com/dev/bulkV2"
            raw_num = "".join(c for c in cleaned_to if c.isdigit())
            if len(raw_num) > 10 and raw_num.startswith("91"):
                numbers = raw_num[2:]
            else:
                numbers = raw_num
                
            payload = {
                "message": body,
                "language": "english",
                "route": "q",
                "numbers": numbers
            }
            headers = {
                'authorization': FAST2SMS_API_KEY,
                'Content-Type': "application/x-www-form-urlencoded",
                'Cache-Control': "no-cache"
            }
            response = requests.post(url, data=payload, headers=headers, timeout=10)
            res_json = response.json()
            if res_json.get("return"):
                print(f"[AUTH] Fast2SMS sent successfully to {numbers}")
                return True
            else:
                print(f"[AUTH] Fast2SMS error: {res_json.get('message')}")
        except ImportError:
            print("[AUTH] requests package is not installed. Skipping Fast2SMS.")
        except Exception as e:
            print(f"[AUTH] Fast2SMS API error: {e}")

    # Fallback print to console
    print(f"\n==========================================")
    print(f"[SMS MOCK] Sending SMS to: {cleaned_to}")
    print(f"[SMS MOCK] Content: {body}")
    print(f"==========================================\n")
    return False


def _otp_email_html(otp: str, purpose: str = "registration") -> str:
    action_label  = "Complete Registration" if purpose == "registration" else "Reset Your Password"
    action_desc   = "complete your registration" if purpose == "registration" else "reset your password"
    otp_digits    = "".join(
        f'<span style="display:inline-block;width:44px;height:52px;line-height:52px;'
        f'text-align:center;background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.4);'
        f'border-radius:10px;font-size:26px;font-weight:800;color:#f9a8d4;margin:0 4px;">'
        f'{d}</span>'
        for d in otp
    )
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FemCare AI — Verification Code</title>
</head>
<body style="margin:0;padding:0;background:#080b14;font-family:'Segoe UI',Inter,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#080b14;padding:48px 16px;">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0"
             style="background:linear-gradient(160deg,#1a0a2e 0%,#0d1020 100%);
                    border:1px solid rgba(236,72,153,0.2);border-radius:20px;
                    overflow:hidden;box-shadow:0 24px 80px rgba(168,85,247,0.15);">

        <!-- Header band -->
        <tr>
          <td style="background:linear-gradient(90deg,#7c3aed,#db2777);padding:28px 40px;text-align:center;">
            <div style="font-size:36px;line-height:1;">❤️</div>
            <h1 style="color:#fff;font-size:24px;font-weight:800;margin:10px 0 4px;
                       letter-spacing:-0.5px;">FemCare AI</h1>
            <p style="color:rgba(255,255,255,0.75);font-size:13px;margin:0;">Your Personal Health Companion</p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:40px 40px 32px;text-align:center;">
            <h2 style="color:#f9a8d4;font-size:18px;font-weight:700;margin:0 0 8px;">
              {action_label}
            </h2>
            <p style="color:#94a3b8;font-size:14px;margin:0 0 32px;line-height:1.6;">
              Use the code below to {action_desc}.<br>
              This code is valid for <strong style="color:#c084fc;">2 minutes</strong>.
            </p>

            <!-- OTP digit boxes -->
            <div style="margin:0 auto 28px;display:inline-block;">
              {otp_digits}
            </div>

            <!-- Warning box -->
            <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);
                        border-radius:10px;padding:14px 20px;margin-bottom:24px;">
              <p style="color:#fbbf24;font-size:12px;margin:0;line-height:1.6;">
                🔒 Never share this code with anyone.<br>
                FemCare AI will never ask for your OTP via phone or chat.
              </p>
            </div>

            <p style="color:#475569;font-size:12px;margin:0;">
              If you didn't request this, you can safely ignore this email.
            </p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:rgba(255,255,255,0.02);border-top:1px solid rgba(255,255,255,0.05);
                     padding:20px 40px;text-align:center;">
            <p style="color:#334155;font-size:11px;margin:0;">
              © 2026 FemCare AI &nbsp;·&nbsp; Reproductive Health Platform
              <br>This is an automated message — please do not reply.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
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

    is_phone = _is_phone_number(req.email)
    contact_type = "phone number" if is_phone else "email address"

    if req.purpose == "registration" and req.email in users:
        raise HTTPException(status_code=409, detail=f"An account with this {contact_type} already exists. Please log in.")

    if req.purpose == "reset" and req.email not in users:
        raise HTTPException(status_code=404, detail=f"No account found with this {contact_type}.")

    otp = _gen_otp()
    _otp_store[req.email] = {
        "otp":        otp,
        "expires_at": time.time() + OTP_TTL_S,
        "purpose":    req.purpose,
        "name":       req.name or "",
    }

    sent = False
    if is_phone:
        # Format phone number for UI display (+91XXXXXXXXXX)
        cleaned_num = "".join(c for c in req.email if c.isdigit() or c == "+")
        if not cleaned_num.startswith("+") and len(cleaned_num) == 10:
            cleaned_num = f"+91{cleaned_num}"
        sms_body = f"FemCare AI verification code: {otp}. Valid for 2 minutes."
        sent = _send_sms(req.email, sms_body)
        message = f"OTP sent to {cleaned_num}."
    else:
        subject = "🔐 FemCare AI — Your Verification Code"
        sent = _send_email(req.email, subject, _otp_email_html(otp, req.purpose))
        message = f"OTP sent to {req.email}. Check your inbox (or server console in dev mode)."

    # Always print the generated OTP directly in the VS Code terminal/console in bold green
    print(f"\n====================\n\033[1;32m[OTP DEBUG] Sent to {req.email}: {otp}\033[0m\n====================\n")

    return {
        "success": True,
        "email_sent": sent if not is_phone else False,
        "sms_sent": sent if is_phone else False,
        "sent": sent,
        "message": message,
        # DEV-ONLY: remove in production
        "dev_otp": otp if not sent else None,
    }


@router.post("/verify-otp")
def verify_otp(req: VerifyOTPRequest):
    """Verify OTP → create account (registration) or confirm identity (reset step 1)."""
    record = _otp_store.get(req.email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP was requested for this contact.")
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
        contact_type = "phone number" if _is_phone_number(req.email) else "email"
        raise HTTPException(status_code=401, detail=f"Invalid {contact_type} or password.")
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
