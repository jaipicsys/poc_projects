import smtplib
import ssl
import random
import threading
from datetime import datetime
from email.mime.text import MIMEText
from config_utils import CONFIG, reload_email_config, CONFIG_LOCK

# Temporary OTP store {email: {"otp": str, "expires": timestamp}}
OTP_STORE = {}


def send_email(to_email, subject, body):
    """
    Sends an email using SMTP details from config_utils.CONFIG (live reloaded)
    """
    try:
        # ensure we always use the latest email config
        reload_email_config()

        with CONFIG_LOCK:
            email_cfg = CONFIG.get("email", {})

        sender = email_cfg.get("sender")
        password = email_cfg.get("password")
        smtp_server = email_cfg.get("smtp_server")
        smtp_port = email_cfg.get("smtp_port")

        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to_email

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            if email_cfg.get("use_auth", True):
                server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        print("Email error:", e)
        return False


def generate_and_store_otp(email: str, ttl: int = 120):
    """
    Generate a 6-digit OTP, store in OTP_STORE with expiry, and auto-remove after TTL seconds
    """
    otp = str(random.randint(100000, 999999))
    expires = datetime.now().timestamp() + ttl
    OTP_STORE[email] = {"otp": otp, "expires": expires}

    # Auto-delete after TTL
    def expire_otp():
        if email in OTP_STORE and OTP_STORE[email]["otp"] == otp:
            del OTP_STORE[email]

    threading.Timer(ttl, expire_otp).start()
    return otp


def verify_otp(email: str, otp: str) -> bool:
    """
    Verify OTP for email. Returns True if valid, False otherwise.
    Also handles expiry cleanup.
    """
    entry = OTP_STORE.get(email)
    if not entry:
        return False
    if datetime.now().timestamp() > entry["expires"]:
        del OTP_STORE[email]
        return False
    if entry["otp"] != otp:
        return False
    return True


def clear_otp(email: str):
    """Remove OTP after successful reset"""
    if email in OTP_STORE:
        del OTP_STORE[email]
