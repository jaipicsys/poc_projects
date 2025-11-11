import smtplib
import json
import calendar
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
from db import get_violations_connection
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from email.mime.image import MIMEImage
from dateutil import parser
import os
from reportlab.platypus import Image as RLImage
import sqlite3
from config_utils import CONFIG, reload_email_config, is_job_enabled

def get_employee_connection():
    """Return connection to employee_master.db (whitelisted vehicles)."""
    return sqlite3.connect("employee_master.db")

# Load config
# with open("config.json") as f:
#     CONFIG = json.load(f)

# EMAIL_CONF = CONFIG.get("email", {})


def generate_pdf_report(period="daily"):
    """Generate PDF report from ANPR DB for the specified period."""
    conn = get_violations_connection()
    cur = conn.cursor()
    now = datetime.now()

    # --- Daily: Previous day 6 AM to 10 PM ---
    if period == "daily":
        prev_day = now - timedelta(days=1)
        start_time = prev_day.replace(hour=6, minute=0, second=0, microsecond=0)
        end_time = prev_day.replace(hour=22, minute=0, second=0, microsecond=0)

    # --- Weekly: Previous Monday 00:00 to Sunday 23:59 ---
    elif period == "weekly":
        prev_monday = now - timedelta(days=now.weekday() + 7)  # Last Monday
        prev_sunday = prev_monday + timedelta(days=6)  # Last Sunday
        start_time = prev_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = prev_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)

    # --- Monthly: Previous month 1st 00:00 to last day 23:59 ---
    else:  # monthly
        year = now.year
        month = now.month - 1
        if month == 0:
            month = 12
            year -= 1
        first_day = datetime(year, month, 1)
        last_day = datetime(year, month, calendar.monthrange(year, month)[1])
        start_time = first_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)

    cur.execute("""
        SELECT plate_number, in_time, out_time, vehicle_type
        FROM anpr_alerts
        WHERE
            (in_time IS NOT NULL AND datetime(in_time) BETWEEN datetime(?) AND datetime(?))
            OR
            (out_time IS NOT NULL AND datetime(out_time) BETWEEN datetime(?) AND datetime(?))
            OR
            (in_time IS NOT NULL AND datetime(in_time) BETWEEN datetime(?) AND datetime(?)
             AND out_time IS NULL)
        ORDER BY in_time DESC
    """, (start_time, end_time, start_time, end_time, start_time, end_time))

    rows = cur.fetchall()
    conn.close()

    print(f"[DEBUG] {period} query returned {len(rows)} rows between {start_time} and {end_time}")

    # --- PDF generation ---
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"{period.capitalize()} ANPR Report", styles["Title"]))
    story.append(Spacer(1, 12))

    data = [["Sl No", "Plate Number", "Vehicle Type", "In Time", "Out Time"]]
    for i, r in enumerate(rows, start=1):
        in_time = ""
        out_time = ""

        if r["in_time"]:
            try:
                in_time = parser.parse(r["in_time"]).strftime("%d-%m-%Y %H:%M")
            except Exception:
                in_time = r["in_time"]  # fallback raw

        if r["out_time"]:
            try:
                out_time = parser.parse(r["out_time"]).strftime("%d-%m-%Y %H:%M")
            except Exception:
                out_time = r["out_time"]

        data.append([
            str(i),
            r["plate_number"],
            r["vehicle_type"] or "",
            in_time,
            out_time
        ])
    # Add total count row
    data.append(["", "", "Total", f"Count: {len(rows)}", ""])
    table = Table(data, colWidths=[40, 100, 80, 140, 140])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),  # Highlight total row
    ]))

    story.append(table)
    doc.build(story)

    buffer.seek(0)
    return buffer

MEDIA_DIR = os.path.join(os.getcwd(), "media")  # adjust if needed

def generate_overstay_pdf(overstays, threshold_hours, now,
                          is_underwork=False,
                          title="Vehicle Overstay Alert Summary"):
    """Generate PDF with overstay vehicle details + images (2 vehicles per page)."""
    buffer = BytesIO()
    # Use smaller margins to utilise page space
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on {now.strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 8))

    for idx, (row, duration) in enumerate(overstays, start=1):
        plate = row["plate_number"]
        vtype = row["vehicle_type"] or "Unknown"
        entry_time = parser.parse(row["in_time"]).strftime("%d-%m-%Y %H:%M")
        exit_time = ""
        if is_underwork and row["out_time"]:
            try:
                exit_time = parser.parse(row["out_time"]).strftime("%d-%m-%Y %H:%M")
            except Exception:
                exit_time = row["out_time"]

        story.append(Paragraph(f"<b>Vehicle {idx}</b>", styles["Heading2"]))
        story.append(Paragraph(f"License Plate Number: {plate}", styles["Normal"]))
        story.append(Paragraph(f"Vehicle Type: {vtype}", styles["Normal"]))
        story.append(Paragraph(f"Entry Time: {entry_time}", styles["Normal"]))
        if is_underwork:
            story.append(Paragraph(f"Exit Time: {exit_time}", styles["Normal"]))
        story.append(Spacer(1, 4))

        # --- Images ---
        img_grid = []
        for key, label in [
            ("image_path", "In-Time Vehicle"),
            ("plate_image_path", "In-Time Plate"),
            ("out_image_path", "Out-Time Vehicle"),
            ("out_plate_image_path", "Out-Time Plate"),
        ]:
            img_file = row[key] if key in row.keys() else None
            if img_file:
                img_path = os.path.join(MEDIA_DIR, os.path.basename(img_file))
                if os.path.exists(img_path):
                    try:
                        cell = RLImage(img_path, width=130, height=95)
                    except Exception as e:
                        cell = Paragraph(f"{label} load error: {e}", styles["Normal"])
                else:
                    cell = Paragraph(f"{label} not found", styles["Normal"])
            else:
                cell = Paragraph(f"{label} missing", styles["Normal"])
            img_grid.append(cell)

        img_table = Table([img_grid[0:2], img_grid[2:4]],
                          colWidths=[150, 150],
                          rowHeights=[100, 100])
        img_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

        if idx % 2 == 0 and idx != len(overstays):
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer


def send_mail(period="daily"):
    reload_email_config()                       # ensure CONFIG["email"] is up-to-date
    if not is_job_enabled(period):
        print(f"[INFO] {period} report is disabled. Skipping...")
        return
    email_conf = CONFIG.get("email", {})
    """Send ANPR report via email with PDF attachment."""
    site_name = email_conf.get("site_name", "Unknown Site")
    company = email_conf.get("company_name", "Company")
    recipients = email_conf.get("recipients", [])
    cc_list = email_conf.get("cc", [])
    now = datetime.now()

    if period == "daily":
        report_date = (now - timedelta(days=1)).strftime("%d-%m-%Y")
    elif period == "weekly":
        prev_monday = now - timedelta(days=now.weekday() + 7)
        prev_sunday = prev_monday + timedelta(days=6)
        report_date = f"{prev_monday.strftime('%d-%m-%Y')} to {prev_sunday.strftime('%d-%m-%Y')}"
    else:  # monthly
        year = now.year
        month = now.month - 1
        if month == 0:
            month = 12
            year -= 1
        report_date = f"{calendar.month_name[month]} {year}"

    subject = f"{period.capitalize()} ANPR Report – {site_name} – {report_date}"

    body = f"""Dear Team,

Please find attached the {period} report of ANPR system activity for {site_name} on {report_date}.

Note: This alert was generated automatically by the ANPR Monitoring System. Please do not reply to this mail.

Regards,
RUAS Admin
"""

    msg = MIMEMultipart()
    msg["From"] = email_conf.get("sender")
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    pdf_buffer = generate_pdf_report(period)
    if not pdf_buffer:
        print(f"[ERROR] Failed to generate {period} PDF report")
        return

    part = MIMEBase("application", "octet-stream")
    part.set_payload(pdf_buffer.read())
    encoders.encode_base64(part)
    part.add_header(
        "Content-Disposition",
        f"attachment; filename={period}_anpr_report_{report_date}.pdf",
    )
    msg.attach(part)

    all_recipients = recipients + cc_list
    smtp_lib = email_conf.get("smtp_lib", "SMTP")
    smtp_class = getattr(smtplib, email_conf.get("smtp_lib", "SMTP"), smtplib.SMTP)

    try:
        with smtp_class(email_conf.get("smtp_server"), email_conf.get("smtp_port")) as smtp:
            smtp.ehlo()
            if smtp_lib == "SMTP":
                try:
                    smtp.starttls()
                except Exception:
                    pass

            if email_conf.get("use_auth", False) and email_conf.get("password"):
                smtp.login(email_conf.get("sender"), email_conf.get("password"))

            smtp.sendmail(msg["From"], all_recipients, msg.as_string())
            print(f"[EMAIL] {period.capitalize()} PDF report sent successfully to {all_recipients}")
    except Exception as e:
        print(f"[ERROR] Failed to send {period} report: {e}")

def send_overstay_summary():
    reload_email_config()
    if not is_job_enabled("overstay"):
        print("[INFO] Overstay report is disabled. Skipping...")
        return
    email_conf = CONFIG.get("email", {})
    """Send a summary email of vehicles overstaying till 12 midnight of previous day."""
    conn = get_violations_connection()
    cur = conn.cursor()

    cur.execute("SELECT hours FROM alerts_config WHERE type='overstay' ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    threshold_hours = 24
    threshold_seconds = threshold_hours * 3600

    now = datetime.now()
    cutoff = datetime(now.year, now.month, now.day, 0, 0, 0) - timedelta(seconds=1)

    cur.execute("""
        SELECT plate_number, vehicle_type, in_time, out_time,
               image_path, plate_image_path
        FROM anpr_alerts
        WHERE in_time IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    overstays = []
    for r in rows:
        in_time = parser.parse(r["in_time"]) if r["in_time"] else None
        out_time = parser.parse(r["out_time"]) if r["out_time"] else None
        if not in_time:
            continue
        duration_seconds = ((out_time or cutoff) - in_time).total_seconds()
        if duration_seconds > threshold_seconds and (out_time is None or out_time > cutoff):
            overstays.append((r, duration_seconds))

    if not overstays:
        print("[INFO] No overstays found for summary mail")
        return

    site_name = email_conf.get("site_name", "Unknown Site")
    company = email_conf.get("company_name", "Company")
    recipients = email_conf.get("recipients", [])
    cc_list = email_conf.get("cc", [])

    today = now.strftime("%d-%m-%Y")
    subject = f"Vehicle Overstay Alert – {site_name} – {today}"

    body = f"""Dear Team,

This is to notify you that the following vehicles have been identified as overstaying within the premises beyond 24 hours.
"""

    for row, duration in overstays:
        plate = row["plate_number"]
        vtype = row["vehicle_type"] or "Unknown"
        entry_time = parser.parse(row["in_time"]).strftime("%d-%m-%Y %H:%M")
        duration_str = f"{duration//3600} Hours, {(duration%3600)//60} Minutes"

        body += f"""
"""

    body += f"""

Action Required:
Please verify the vehicle status and take necessary action in line with security protocols.

Note: This alert was generated automatically by the ANPR Monitoring System.

Regards,
RUAS Admin
"""

    msg = MIMEMultipart()
    msg["From"] = email_conf.get("sender")
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # --- Attach PDF with details + images ---
    pdf_buffer = generate_overstay_pdf(overstays, threshold_hours, now)
    pdf_part = MIMEBase("application", "octet-stream")
    pdf_part.set_payload(pdf_buffer.read())
    encoders.encode_base64(pdf_part)
    pdf_part.add_header(
        "Content-Disposition",
        f"attachment; filename=overstay_summary_{today}.pdf",
    )
    msg.attach(pdf_part)

    all_recipients = recipients + cc_list
    smtp_lib = email_conf.get("smtp_lib", "SMTP")
    smtp_class = getattr(smtplib, email_conf.get("smtp_lib", "SMTP"), smtplib.SMTP)

    try:
        with smtp_class(email_conf.get("smtp_server"), email_conf.get("smtp_port")) as smtp:
            smtp.ehlo()
            if smtp_lib == "SMTP":
                try:
                    smtp.starttls()
                except Exception:
                    pass
            if email_conf.get("use_auth", False) and email_conf.get("password"):
                smtp.login(email_conf.get("sender"), email_conf.get("password"))

            smtp.sendmail(msg["From"], all_recipients, msg.as_string())
            print(f"[EMAIL] Overstay summary sent successfully to {all_recipients}")
    except Exception as e:
        print(f"[ERROR] Failed to send overstay summary: {e}")

def send_underwork_summary():
    reload_email_config()
    if not is_job_enabled("underwork"):
        print("[INFO] Underwork report is disabled. Skipping...")
        return
    email_conf = CONFIG.get("email", {})
    """
    Send a summary email of whitelisted employees who worked less than 8 hours today.
    Runs at 1:30 AM next day.
    """
    # Connect to the main ANPR database
    conn = get_db_connection()
    cur = conn.cursor()

    # Connect to the employee whitelist database
    emp_conn = get_employee_connection()
    emp_cur = emp_conn.cursor()

    # Fetch all whitelisted vehicle numbers
    emp_cur.execute('SELECT "Vehicle No" FROM employee_master')
    whitelist = {
        row[0].strip().replace(" ", "").upper()
        for row in emp_cur.fetchall()
    }
    emp_conn.close()

    # Get today’s date range
    today = datetime.now().date()
    prev_day = today - timedelta(days=1)
    start_day = datetime(prev_day.year, prev_day.month, prev_day.day, 0, 0, 0)
    end_day = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 59, 59)

    # Fetch all entries for today
    cur.execute("""
        SELECT plate_number, vehicle_type, in_time, out_time,
               image_path, plate_image_path, out_image_path, out_plate_image_path
        FROM anpr_alerts
        WHERE in_time IS NOT NULL
          AND datetime(in_time) BETWEEN datetime(?) AND datetime(?)
    """, (start_day, end_day))
    rows = cur.fetchall()
    conn.close()

    underworks = []

    for r in rows:
        plate = r["plate_number"].strip().replace(" ", "").upper()

        # Skip if not whitelisted
        if plate not in whitelist:
            continue

        # Skip if employee has not exited yet (no out_time)
        if not r["out_time"]:
            continue

        try:
            in_time = parser.parse(r["in_time"])
            out_time = parser.parse(r["out_time"])
            duration_seconds = (out_time - in_time).total_seconds()

            if duration_seconds < 8 * 3600:
                underworks.append((r, duration_seconds))
                print(f"[DEBUG] Underwork detected: {plate}, Duration: {duration_seconds} seconds")
        except Exception as e:
            print(f"[WARN] Skipping record {r}: {e}")
            continue

    # If no underwork found, exit
    if not underworks:
        print("[INFO] No employees found with <8 hours work today")
        return

    # Prepare email
    site_name = email_conf.get("site_name", "Unknown Site")
    recipients = email_conf.get("recipients", [])
    cc_list = email_conf.get("cc", [])
    today = prev_day.strftime("%d-%m-%Y")
    subject = f"Early Departure Alert – Employees < 8 Hours – {site_name} – {today}"

    # Email body
    body = f"""Dear Team,
The following employees exited the premises today ({today}) after working less than 8 hours.

"""

    for row, duration in underworks:
        plate = row["plate_number"]
        vtype = row["vehicle_type"] or "Unknown"
        entry_time = parser.parse(row["in_time"]).strftime("%d-%m-%Y %H:%M")
        exit_time = parser.parse(row["out_time"]).strftime("%d-%m-%Y %H:%M")
        duration_str = f"{duration//3600} Hours, {(duration%3600)//60} Minutes"
        body += f"""

"""

    body += f"""

Regards,
RUAS Admin
"""

    # Create email message
    msg = MIMEMultipart()
    msg["From"] = email_conf.get("sender")
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Generate and attach PDF
    pdf_buffer = generate_overstay_pdf(underworks, 8, prev_day, is_underwork=True, title="Early Departure Alert Summary")
    pdf_part = MIMEBase("application", "octet-stream")
    pdf_part.set_payload(pdf_buffer.read())
    encoders.encode_base64(pdf_part)
    pdf_part.add_header(
        "Content-Disposition",
        f"attachment; filename=early_departure{today}.pdf",
    )
    msg.attach(pdf_part)

    # Send email
    all_recipients = recipients + cc_list
    smtp_lib = email_conf.get("smtp_lib", "SMTP")
    smtp_class = getattr(smtplib, email_conf.get("smtp_lib", "SMTP"), smtplib.SMTP)

    try:
        with smtp_class(email_conf.get("smtp_server"), email_conf.get("smtp_port")) as smtp:
            smtp.ehlo()
            if smtp_lib == "SMTP":
                try:
                    smtp.starttls()
                except Exception:
                    pass
            if email_conf.get("use_auth", False) and email_conf.get("password"):
                smtp.login(email_conf.get("sender"), email_conf.get("password"))
            smtp.sendmail(msg["From"], all_recipients, msg.as_string())
            print(f"[EMAIL] early departure summary sent successfully to {all_recipients}")
    except Exception as e:
        print(f"[ERROR] Failed to send early departure summary: {e}")

def generate_blacklist_pdf(row):
    """Generate PDF for a single blacklist vehicle entry with images."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    now = parser.parse(row["in_time"]) if row.get("in_time") else datetime.now()
    story.append(Paragraph("Blacklist Vehicle Alert", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Detected on {now.strftime('%d-%m-%Y %H:%M')}", styles["Normal"]))
    story.append(Paragraph(f"License Plate: {row['plate_number']}", styles["Normal"]))
    story.append(Paragraph(f"Vehicle Type: {row.get('vehicle_type','Unknown')}", styles["Normal"]))
    story.append(Spacer(1, 8))

    # Images (same as overstay)
    img_grid = []
    for key, label in [("image_path", "In-Time Vehicle"), ("plate_image_path", "In-Time Plate")]:
        img_file = row.get(key)
        if img_file:
            img_path = os.path.join(MEDIA_DIR, os.path.basename(img_file))
            if os.path.exists(img_path):
                try:
                    cell = RLImage(img_path, width=130, height=95)
                except Exception as e:
                    cell = Paragraph(f"{label} load error: {e}", styles["Normal"])
            else:
                cell = Paragraph(f"{label} not found", styles["Normal"])
        else:
            cell = Paragraph(f"{label} missing", styles["Normal"])
        img_grid.append(cell)

    img_table = Table([img_grid], colWidths=[150, 150], rowHeights=[100])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(img_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

def send_blacklist_alert(row):
    reload_email_config()
    email_conf = CONFIG.get("email", {})
    recipients = email_conf.get("recipients", [])
    cc_list = email_conf.get("cc", [])
    site_name = email_conf.get("site_name", "Unknown Site")
    now = parser.parse(row["in_time"]) if row.get("in_time") else datetime.now()

    subject = f"Blacklist Alert – {row['plate_number']} detected at {site_name}"
    body = f"""
Dear Team,

Vehicle {row['plate_number']} has entered the premises at {site_name} on {now.strftime('%d-%m-%Y %H:%M')}.
Immediate action is required.

This alert was generated automatically by the ANPR Monitoring System.

Regards,
RUAS Admin
"""

    msg = MIMEMultipart()
    msg["From"] = email_conf.get("sender")
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach PDF
    pdf_buffer = generate_blacklist_pdf(row)
    pdf_part = MIMEBase("application", "octet-stream")
    pdf_part.set_payload(pdf_buffer.read())
    encoders.encode_base64(pdf_part)
    pdf_part.add_header("Content-Disposition", f"attachment; filename=blacklist_{row['plate_number']}_{now.strftime('%d%m%Y_%H%M')}.pdf")
    msg.attach(pdf_part)

    all_recipients = recipients + cc_list
    try:
        with smtplib.SMTP(email_conf.get("smtp_server"), email_conf.get("smtp_port")) as smtp:
            smtp.ehlo()
            if email_conf.get("smtp_lib", "SMTP") == "SMTP":
                try: smtp.starttls()
                except Exception: pass
            if email_conf.get("use_auth") and email_conf.get("password"):
                smtp.login(email_conf.get("sender"), email_conf.get("password"))
            smtp.sendmail(msg["From"], all_recipients, msg.as_string())
            print(f"[EMAIL] Blacklist alert sent for {row['plate_number']}")
    except Exception as e:
        print(f"[ERROR] Failed to send blacklist alert for {row['plate_number']}: {e}")

def schedule_reports():
    """Schedule daily, weekly, monthly reports."""
    scheduler = BackgroundScheduler()
    jobs_config = CONFIG.get("report_jobs", {})

    if jobs_config.get("daily", False):
        scheduler.add_job(send_mail, "cron", hour=2, minute=0, args=["daily"], id="daily_report")

    if jobs_config.get("weekly", False):
        scheduler.add_job(send_mail, "cron", day_of_week="mon", hour=1, minute=0, args=["weekly"], id="weekly_report")

    if jobs_config.get("monthly", False):
        scheduler.add_job(send_mail, "cron", day=1, hour=2, minute=0, args=["monthly"], id="monthly_report")

    if jobs_config.get("overstay", False):
        scheduler.add_job(send_overstay_summary, "cron", hour=1, minute=30, id="overstay_report")

    if jobs_config.get("underwork", False):
        scheduler.add_job(send_underwork_summary, "cron", hour=2, minute=30, id="underwork_report")

    scheduler.start()
    return scheduler
