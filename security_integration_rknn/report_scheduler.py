# report_scheduler.py
import io
import json
import smtplib
import calendar
from io import BytesIO
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from apscheduler.schedulers.background import BackgroundScheduler
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from db import get_events_connection
from config_utils import CONFIG, CONFIG_LOCK, reload_email_config, update_report_job

# Scheduler instance (module-level)
_scheduler = None

# ---------- Helpers to build time windows ----------
def _get_period_window(period: str):
    now = datetime.now()
    if period == "daily":
        # previous day 06:00 to 22:00 (matching earlier pattern)
        prev = now - timedelta(days=1)
        start = prev.replace(hour=6, minute=0, second=0, microsecond=0)
        end = prev.replace(hour=22, minute=0, second=0, microsecond=0)
    elif period == "weekly":
        # last calendar week: previous Monday 00:00 to Sunday 23:59:59
        # calculate previous week Monday
        prev_monday = now - timedelta(days=now.weekday() + 7)
        prev_sunday = prev_monday + timedelta(days=6)
        start = prev_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = prev_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == "monthly":
        # previous month first day 00:00 to last day 23:59:59
        year = now.year
        month = now.month - 1
        if month == 0:
            month = 12
            year -= 1
        first_day = datetime(year, month, 1)
        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        start = first_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        # fallback: last 24 hours
        end = now
        start = now - timedelta(days=1)
    return start, end

# ---------- Report generation ----------
def generate_events_report_pdf(period="daily"):
    """
    Query loitering_events and perimeter_events for the given period
    and produce a PDF (BytesIO) summarizing the events.
    """
    start, end = _get_period_window(period)
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_events_connection()
    cur = conn.cursor()

    # Fetch loitering events in period
    cur.execute("""
        SELECT id, cam_id, frame_idx, track_id, event, duration, zone, timestamp, snapshot_path
        FROM loitering_events
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """, (start_str, end_str))
    loiter_rows = [dict(r) for r in cur.fetchall()]

    # Fetch perimeter events in period
    cur.execute("""
        SELECT id, cam_id, frame_idx, track_id, event, bbox, timestamp, snapshot_path
        FROM perimeter_events
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """, (start_str, end_str))
    perim_rows = [dict(r) for r in cur.fetchall()]

    conn.close()

    # build pdf
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    story = []

    title = f"{period.capitalize()} Events Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Period: {start.strftime('%d-%m-%Y %H:%M:%S')} to {end.strftime('%d-%m-%Y %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 8))

    # Loitering section
    story.append(Paragraph("Loitering Events", styles["Heading2"]))
    if loiter_rows:
        data = [["#", "Cam", "Track", "Zone", "Duration(s)", "Timestamp", "Snapshot"]]
        for i, r in enumerate(loiter_rows, start=1):
            data.append([
                str(i),
                r.get("cam_id", ""),
                str(r.get("track_id", "")),
                r.get("zone", ""),
                str(r.get("duration", "")),
                r.get("timestamp", ""),
                r.get("snapshot_path", "") or ""
            ])
        table = Table(data, colWidths=[30, 60, 50, 80, 60, 110, 120])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No loitering events in this period.", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Perimeter section
    story.append(Paragraph("Perimeter Breaches", styles["Heading2"]))
    if perim_rows:
        data = [["#", "Cam", "Track", "BBox", "Timestamp", "Snapshot"]]
        for i, r in enumerate(perim_rows, start=1):
            data.append([
                str(i),
                r.get("cam_id", ""),
                str(r.get("track_id", "")),
                str(r.get("bbox", "")),
                r.get("timestamp", ""),
                r.get("snapshot_path", "") or ""
            ])
        table = Table(data, colWidths=[30, 60, 50, 140, 110, 120])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey)
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No perimeter breaches in this period.", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ---------- Email sending (with attachment) ----------
def _send_email_with_attachment(subject: str, body: str, attachment_bytes: bytes, attachment_name: str):
    """
    Send an email with the given bytes as attachment using SMTP settings in CONFIG.
    """
    # reload live email config
    reload_email_config()
    with CONFIG_LOCK:
        email_cfg = CONFIG.get("email", {})

    smtp_server = email_cfg.get("smtp_server", "smtp.gmail.com")
    smtp_port = email_cfg.get("smtp_port", 587)
    sender = email_cfg.get("sender")
    password = email_cfg.get("password")
    recipients = email_cfg.get("recipients", []) or []
    cc_list = email_cfg.get("cc", []) or []

    if not recipients:
        print("[REPORT SCHEDULER] No recipients configured - skipping email send.")
        return False

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(attachment_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
    msg.attach(part)

    all_recipients = recipients + cc_list

    smtp_lib = email_cfg.get("smtp_lib", "SMTP")
    smtp_class = getattr(smtplib, smtp_lib, smtplib.SMTP)

    try:
        with smtp_class(smtp_server, smtp_port) as smtp:
            smtp.ehlo()
            # StartTLS attempt (safe when server supports it)
            try:
                smtp.starttls()
            except Exception:
                pass
            if email_cfg.get("use_auth", False) and password:
                try:
                    smtp.login(sender, password)
                except Exception as e:
                    print(f"[REPORT SCHEDULER] SMTP login failed: {e}")
            smtp.sendmail(sender, all_recipients, msg.as_string())
        print(f"[REPORT SCHEDULER] Email sent to: {all_recipients}")
        return True
    except Exception as e:
        print(f"[REPORT SCHEDULER] Failed to send email: {e}")
        return False

# ---------- Job functions ----------
def send_report_job(period="daily"):
    """
    Job entrypoint: generate PDF for period and email it.
    """
    print(f"[REPORT SCHEDULER] Generating {period} report...")
    try:
        pdf_buffer = generate_events_report_pdf(period)
        now = datetime.now()
        report_name = f"{period}_events_report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
        subject = f"{period.capitalize()} Events Report - {CONFIG.get('email', {}).get('site_name', '')} - {now.strftime('%d-%m-%Y')}"
        body = f"Please find attached the {period} events report for {CONFIG.get('email', {}).get('site_name', '')}.\n\nPeriod: see inside the PDF."

        sent = _send_email_with_attachment(subject, body, pdf_buffer.getvalue(), report_name)
        if not sent:
            print(f"[REPORT SCHEDULER] Failed to send {period} report email.")
    except Exception as e:
        print(f"[REPORT SCHEDULER] Error while creating/sending {period} report: {e}")

# ---------- Scheduler control ----------
def schedule_reports():
    """
    Read schedule config from config.json and install APScheduler jobs.
    Expected config keys (optional):
      - report_jobs: { "daily": true, "weekly": false, "monthly": false }
      - custom_cron or interval options can be added later
    By default it will schedule daily/weekly/monthly if turned on in config.
    """
    global _scheduler
    if _scheduler is not None:
        print("[REPORT SCHEDULER] Scheduler already running.")
        return _scheduler

    reload_email_config()
    with CONFIG_LOCK:
        jobs_cfg = CONFIG.get("report_jobs", {})

    _scheduler = BackgroundScheduler()
    # daily at 02:00 if enabled
    if jobs_cfg.get("daily", False):
        _scheduler.add_job(send_report_job, "cron", hour=2, minute=0, args=["daily"], id="daily_report", replace_existing=True)
        print("[REPORT SCHEDULER] Daily report scheduled at 02:00.")

    # weekly on Monday 01:00
    if jobs_cfg.get("weekly", False):
        _scheduler.add_job(send_report_job, "cron", day_of_week="mon", hour=1, minute=0, args=["weekly"], id="weekly_report", replace_existing=True)
        print("[REPORT SCHEDULER] Weekly report scheduled on Monday 01:00.")

    # monthly on day 1 at 02:00
    if jobs_cfg.get("monthly", False):
        _scheduler.add_job(send_report_job, "cron", day=1, hour=2, minute=0, args=["monthly"], id="monthly_report", replace_existing=True)
        print("[REPORT SCHEDULER] Monthly report scheduled on day 1 at 02:00.")

    # start the scheduler
    _scheduler.start()
    print("[REPORT SCHEDULER] Scheduler started.")
    return _scheduler

def stop_scheduler():
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
            print("[REPORT SCHEDULER] Scheduler stopped.")
        except Exception as e:
            print("[REPORT SCHEDULER] Error stopping scheduler:", e)
        _scheduler = None

def reschedule_from_config():
    """
    Reload config.json and reschedule jobs.
    Useful to call after config updates.
    """
    global _scheduler
    stop_scheduler()
    # Ensure CONFIG is reloaded by config_utils
    reload_email_config()
    schedule_reports()

# Utility to allow programmatic toggling
def enable_job(job_name: str, enabled: bool):
    """
    Update config and reschedule.
    job_name is one of: 'daily', 'weekly', 'monthly'
    """
    update_report_job(job_name, bool(enabled))
    reschedule_from_config()

# If imported and desired to start immediately, caller (app.py) should call schedule_reports()
if __name__ == "__main__":
    # allow local test: schedule according to config and keep running
    schedule_reports()
    try:
        print("[REPORT SCHEDULER] Running in standalone mode. Ctrl+C to exit.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_scheduler()
        print("[REPORT SCHEDULER] Exited.")
