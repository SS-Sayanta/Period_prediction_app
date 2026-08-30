"""
auth_router.py — FemCare AI Email Authentication Router with MySQL Integration
Endpoints:
  POST /api/auth/register         → direct registration with email & password
  POST /api/auth/send-otp          → send 6-digit OTP to user email
  POST /api/auth/verify-otp        → verify OTP + create account in MySQL + return JWT
  POST /api/auth/login             → email & password login + return JWT
  POST /api/auth/forgot-password   → send password-reset OTP to email
  POST /api/auth/reset-password    → verify reset OTP + update password in MySQL
  GET  /api/auth/me                → return current user info from JWT / MySQL
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

import pymysql
import pymysql.cursors
import traceback
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError

# ── Config ────────────────────────────────────────────────────────────────────
JWT_SECRET   = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALG      = "HS256"
JWT_EXPIRE_H = 720          # 30 days

SMTP_SERVER     = os.getenv("SMTP_HOST", os.getenv("SMTP_SERVER", "smtp.gmail.com")).strip()
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL    = os.getenv("SMTP_USER", os.getenv("SMTP_EMAIL", os.getenv("SENDER_EMAIL", ""))).strip()
SENDER_PASSWORD = os.getenv("SMTP_PASSWORD", os.getenv("SMTP_APP_PASSWORD", os.getenv("SENDER_PASSWORD", ""))).strip()
SMTP_FROM       = os.getenv("SMTP_FROM", SENDER_EMAIL or "noreply@femcare.ai")

USERS_FILE   = Path("data/users.json")
OTP_TTL_S    = 600          # OTP expires after 10 minutes

# ── MySQL Credentials ────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME     = os.getenv("DB_NAME", "femcare_db")

# ── Password Hashing & Router Initialization ─────────────────────────────────
pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
router   = APIRouter(tags=["auth"])

# In-memory OTP store: { email: {otp, expires_at, purpose} }
_otp_store: Dict[str, Dict[str, Any]] = {}


# ── Database Helpers ─────────────────────────────────────────────────────────
def _get_db_connection():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        ssl={"ssl_mode": "PREFERRED"}
    )


def init_db() -> None:
    """Ensure MySQL database 'femcare_db', 'users' table, and migrate legacy json users."""
    try:
        conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, ssl={"ssl_mode": "PREFERRED"})
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        conn.close()

        conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    name VARCHAR(255) DEFAULT '',
                    password VARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

        if USERS_FILE.exists():
            try:
                users_data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
                with conn.cursor() as cursor:
                    for email_key, u in users_data.items():
                        cursor.execute("SELECT id FROM users WHERE email = %s", (email_key.lower().strip(),))
                        if not cursor.fetchone():
                            cursor.execute(
                                "INSERT INTO users (email, name, password) VALUES (%s, %s, %s)",
                                (u.get("email", email_key).lower().strip(), u.get("name", ""), u.get("password", ""))
                            )
            except Exception as me:
                print(f"[AUTH DB] Warning during legacy user migration: {me}")

        conn.close()
        print("[AUTH DB] [OK] MySQL database 'femcare_db' and 'users' table initialized successfully.")
    except Exception as e:
        print(f"[AUTH DB] [WARNING] MySQL initialization warning: {e}")


init_db()


def _load_users_json() -> Dict[str, Any]:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_users_json(users: Dict[str, Any]) -> None:
    try:
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[AUTH] [WARNING] Could not save users.json (safe to ignore if using MySQL): {e}")


def db_get_user(email: str) -> Optional[Dict[str, Any]]:
    """Retrieve user by email from MySQL users table (case-insensitive + trim-safe)."""
    cleaned_email = email.lower().strip()
    try:
        conn = _get_db_connection()
        with conn.cursor() as cursor:
            # Use LOWER(TRIM()) to match regardless of stored case/whitespace
            cursor.execute(
                "SELECT id, email, name, password, created_at FROM users WHERE LOWER(TRIM(email)) = %s",
                (cleaned_email,)
            )
            user = cursor.fetchone()
        conn.close()
        if user:
            # Normalise: DictCursor returns dict; tuple cursor returns tuple
            if isinstance(user, dict):
                return user
            cols = ["id", "email", "name", "password", "created_at"]
            return dict(zip(cols, user))
    except Exception as e:
        print(f"[AUTH DB] Error querying user '{cleaned_email}' from MySQL: {e}")

    # JSON file fallback (dev/offline)
    users = _load_users_json()
    return users.get(cleaned_email)


def db_create_user(email: str, name: str, password_hash: str) -> bool:
    """Insert a new email user record into MySQL users table."""
    cleaned_email = email.lower().strip()
    try:
        conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (email, name, password) VALUES (%s, %s, %s)",
                (cleaned_email, name.strip(), password_hash)
            )
        conn.close()
    except pymysql.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email address already exists.")
    except Exception as e:
        print(f"[AUTH DB] Error creating user '{cleaned_email}' in MySQL, falling back to JSON: {e}")

    # Fallback to JSON if MySQL fails or after successful MySQL insert
    users = _load_users_json()
    users[cleaned_email] = {
        "email": cleaned_email,
        "name": name.strip(),
        "password": password_hash,
        "created_at": datetime.utcnow().isoformat()
    }
    _save_users_json(users)
    return True


def db_update_password(email: str, new_password_hash: str) -> bool:
    """Update user password in MySQL users table."""
    cleaned_email = email.lower().strip()
    try:
        conn = _get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (new_password_hash, cleaned_email))
        conn.close()
    except Exception as e:
        print(f"[AUTH DB] Error updating password for '{cleaned_email}' in MySQL, falling back to JSON: {e}")

    users = _load_users_json()
    if cleaned_email in users:
        users[cleaned_email]["password"] = new_password_hash
        _save_users_json(users)
    return True


def _gen_otp() -> str:
    return str(random.randint(100000, 999999))


def _make_jwt(email: str, name: str) -> str:
    payload = {
        "sub":   email.lower().strip(),
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
            f"[AUTH] [WARNING] SMTP NOT CONFIGURED — email to {to} skipped.\n"
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
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=15) as s:
                s.login(SENDER_EMAIL, SENDER_PASSWORD)
                s.sendmail(SENDER_EMAIL, [to], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                s.login(SENDER_EMAIL, SENDER_PASSWORD)
                s.sendmail(SENDER_EMAIL, [to], msg.as_string())

        print(f"[AUTH] [OK] Email sent successfully to {to}")
        return True
    except smtplib.SMTPAuthenticationError:
        print(
            f"[AUTH] [ERROR] Gmail authentication failed for '{SENDER_EMAIL}'.\n"
            f"  -> Make sure you are using a Gmail App Password, NOT your regular password.\n"
            f"  -> Generate one at: https://myaccount.google.com/apppasswords"
        )
        return False
    except smtplib.SMTPConnectError:
        print(f"[AUTH] [ERROR] Could not connect to {SMTP_SERVER}:{SMTP_PORT}. Check network or firewall.")
        return False
    except Exception as e:
        print(f"[AUTH] [ERROR] SMTP error: {type(e).__name__}: {e}")
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

        <tr>
          <td style="background:linear-gradient(90deg,#7c3aed,#db2777);padding:28px 40px;text-align:center;">
            <div style="font-size:36px;line-height:1;">❤️</div>
            <h1 style="color:#fff;font-size:24px;font-weight:800;margin:10px 0 4px;
                       letter-spacing:-0.5px;">FemCare AI</h1>
            <p style="color:rgba(255,255,255,0.75);font-size:13px;margin:0;">Your Personal Health Companion</p>
          </td>
        </tr>

        <tr>
          <td style="padding:40px 40px 32px;text-align:center;">
            <h2 style="color:#f9a8d4;font-size:18px;font-weight:700;margin:0 0 8px;">
              {action_label}
            </h2>
            <p style="color:#94a3b8;font-size:14px;margin:0 0 32px;line-height:1.6;">
              Use the code below to {action_desc}.<br>
              This code is valid for <strong style="color:#c084fc;">10 minutes</strong>.
            </p>

            <div style="margin:0 auto 28px;display:inline-block;">
              {otp_digits}
            </div>

            <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);
                        border-radius:10px;padding:14px 20px;margin-bottom:24px;">
              <p style="color:#fbbf24;font-size:12px;margin:0;line-height:1.6;">
                [SECURE] Never share this code with anyone.<br>
                FemCare AI will never ask for your OTP via chat or third-party call.
              </p>
            </div>

            <p style="color:#475569;font-size:12px;margin:0;">
              If you didn't request this, you can safely ignore this email.
            </p>
          </td>
        </tr>

        <tr>
          <td style="background:rgba(255,255,255,0.02);border-top:1px solid rgba(255,255,255,0.05);
                     padding:20px 40px;text-align:center;">
            <p style="color:#334155;font-size:11px;margin:0;">
              © 2026 FemCare AI · Reproductive Health Platform
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
class RegisterRequest(BaseModel):
    email:    str
    password: str
    name:     Optional[str] = ""


class SendOTPRequest(BaseModel):
    email:   str
    name:    Optional[str] = ""
    purpose: str = "registration"   # "registration" | "reset"


class VerifyOTPRequest(BaseModel):
    email:    str
    otp:      str
    password: Optional[str] = None  # required for registration


class LoginRequest(BaseModel):
    email:    str
    password: str


class ResetPasswordRequest(BaseModel):
    email:        str
    otp:          str
    new_password: str


# ── Helper for Email Validation ──────────────────────────────────────────────
def _validate_email_str(email: str) -> str:
    cleaned = email.lower().strip()
    if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    return cleaned


# ── Routes ────────────────────────────────────────────────────────────────────

async def get_register_req(request: Request) -> RegisterRequest:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        data = await request.json()
        return RegisterRequest(**data)
    form = await request.form()
    return RegisterRequest(email=str(form.get("email","")), password=str(form.get("password","")), name=str(form.get("name","")))

async def get_login_req(request: Request) -> LoginRequest:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        data = await request.json()
        return LoginRequest(**data)
    form = await request.form()
    return LoginRequest(email=str(form.get("email","")), password=str(form.get("password","")))

@router.post("/auth/register")
def register(req: RegisterRequest = Depends(get_register_req)):
    """Direct user registration endpoint wired to MySQL database with bcrypt hashing."""
    email = _validate_email_str(req.email)
    
    if not req.password:
        raise HTTPException(status_code=400, detail="Password is required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    existing = db_get_user(email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email address already exists. Please log in.")

    user_name = req.name.strip() if req.name else "User"
    hashed_pwd = pwd_ctx.hash(req.password)
    db_create_user(email, user_name, hashed_pwd)

    token = _make_jwt(email, user_name)
    return {"success": True, "token": token, "name": user_name, "email": email}


@router.post("/auth/send-otp")
def send_otp(req: SendOTPRequest):
    """Generate & send a 6-digit OTP to the user's email after checking MySQL database."""
    email = _validate_email_str(req.email)
    existing = db_get_user(email)

    if req.purpose == "registration" and existing:
        raise HTTPException(status_code=409, detail="An account with this email address already exists. Please log in.")

    if req.purpose == "reset" and not existing:
        raise HTTPException(status_code=404, detail="No account found with this email address.")

    otp = _gen_otp()
    _otp_store[email] = {
        "otp":        otp,
        "expires_at": time.time() + OTP_TTL_S,
        "purpose":    req.purpose,
        "name":       req.name or "",
    }

    subject = "🔐 FemCare AI — Your Verification Code"
    sent = _send_email(email, subject, _otp_email_html(otp, req.purpose))

    if not sent:
        return {
            "success": False,
            "message": "Unable to send verification code. Please try again."
        }

    return {
        "success": True,
        "message": "Verification code sent to your email."
    }


@router.post("/auth/verify-otp")
def verify_otp(req: VerifyOTPRequest):
    """Verify OTP → create account in MySQL database (registration) or confirm identity (reset step 1)."""
    email = _validate_email_str(req.email)
    record = _otp_store.get(email)

    if not record:
        raise HTTPException(status_code=400, detail="No OTP was requested for this email.")
    if time.time() > record["expires_at"]:
        _otp_store.pop(email, None)
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    entered_otp = req.otp.strip()
    if entered_otp != record["otp"]:
        raise HTTPException(status_code=400, detail="Incorrect OTP. Please try again.")

    _otp_store.pop(email, None)  # consume OTP

    if record["purpose"] == "registration":
        if not req.password or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
        
        existing = db_get_user(email)
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email address already exists. Please log in.")

        user_name = record["name"] or "User"
        hashed_pwd = pwd_ctx.hash(req.password)
        db_create_user(email, user_name, hashed_pwd)
        token = _make_jwt(email, user_name)
        return {
            "status": "success",
            "message": "Account verified and registered successfully!",
            "token": token,
            "user": {
                "email": email,
                "name": user_name
            }
        }

    # purpose == "reset" — mark email as OTP-verified for the reset step
    _otp_store[f"reset_verified_{email}"] = {"verified": True, "expires_at": time.time() + 300}
    return {"success": True, "message": "Identity verified. You may now set a new password."}


@router.post("/auth/login")
def login(req: LoginRequest = Depends(get_login_req)):
    """Verify user email & password against MySQL database and return JWT token."""
    try:
        email = req.email.strip().lower()
        password = req.password.strip()
        user = db_get_user(email)
        
        if not user:
            return JSONResponse(status_code=401, content={"detail": "Invalid email address or password."})
            
        stored_password = user.get("password_hash") or user.get("password") or user.get("hashed_password") or ""
        is_valid = False
        try:
            is_valid = pwd_ctx.verify(password, stored_password)
        except Exception:
            # Fallback in case plain text or raw comparison was stored
            is_valid = (password == stored_password)
            
        if not is_valid:
            return JSONResponse(status_code=401, content={"detail": "Invalid email address or password."})
        
        token = _make_jwt(email, user.get("name", "User"))
        return JSONResponse(
            status_code=200, 
            content={
                "status": "success", 
                "message": "Login successful", 
                "user": {"email": email},
                "token": token,
                "name": user.get("name", "User")
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Internal server error during login."})


@router.post("/auth/reset-password")
def reset_password(req: ResetPasswordRequest):
    """Verify reset OTP + update password in MySQL database."""
    email = _validate_email_str(req.email)
    record = _otp_store.get(email)
    pre = _otp_store.get(f"reset_verified_{email}")

    if not record and not pre:
        raise HTTPException(status_code=400, detail="No reset session found. Please restart the process.")

    if record:
        if time.time() > record["expires_at"]:
            _otp_store.pop(email, None)
            raise HTTPException(status_code=400, detail="OTP has expired.")
        entered_otp = req.otp.strip()
        if entered_otp != record["otp"]:
            raise HTTPException(status_code=400, detail="Incorrect OTP.")
        _otp_store.pop(email, None)

    if pre:
        if time.time() > pre["expires_at"]:
            _otp_store.pop(f"reset_verified_{email}", None)
            raise HTTPException(status_code=400, detail="Reset session expired. Please restart.")
        _otp_store.pop(f"reset_verified_{email}", None)

    if not req.new_password or len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    user = db_get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="Account not found.")

    new_hash = pwd_ctx.hash(req.new_password)
    db_update_password(email, new_hash)
    return {"success": True, "message": "Password updated successfully. You can now log in."}


@router.get("/auth/me")
def get_me(authorization: Optional[str] = Header(None)):
    """Fetch current authenticated user info from JWT token and MySQL database."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated.")
    token   = authorization.split(" ", 1)[1]
    payload = _decode_jwt(token)
    email   = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token payload.")

    user = db_get_user(email)
    user_name = user["name"] if user and user.get("name") else payload.get("name", "")
    return {"email": email, "name": user_name, "success": True}
