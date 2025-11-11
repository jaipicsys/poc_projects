from flask import Blueprint, request, jsonify, send_from_directory, send_file, url_for
from db import get_violations_connection, get_auth_db_connection, get_history, get_all_camera_status
import os
from datetime import datetime
import hashlib
import jwt
import sqlite3
import re
from dateutil import parser
from employee import upload_patient_excel_to_db_from_bytes
from db import get_violations_connection, get_auth_db_connection, UPLOADED_DB, get_violations_connection
from dateutil import parser
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import requests
from config_utils import reload_email_config, update_report_job, CONFIG
import json
import pandas as pd
import io
from email_utils import send_email, generate_and_store_otp, verify_otp, clear_otp

SECRET_KEY = "picsysnexilishrbrdoddaballapur"
api_bp = Blueprint('api', __name__)
scheduler_instance = None

CAMERA_CONFIG = {
    "cam1": {"camera_name": "cam 1"},
    "cam2": {"camera_name": "cam 2"},
    "cam3": {"camera_name": "cam 3"},
    "cam4": {"camera_name": "cam 4"}
}

@api_bp.route("/camera_status", methods=["GET"])
def camera_status():
    """
    Returns only failed cameras if any, otherwise a flag indicating all are running.
    """
    try:
        data = get_all_camera_status()

        # Load names from config.json
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
            camera_names = cfg.get("camera_names", {})
        except Exception:
            camera_names = {}

        # Attach friendly names
        for item in data:
            cam_id = item.get("camera_id")
            item["name"] = camera_names.get(cam_id, cam_id)

        # Filter out failed cameras (anything not 'running')
        failed_cameras = [cam for cam in data if cam.get("status") != "running"]

        # If all are running → return simple true flag
        if not failed_cameras:
            result = {"all_running": True}
        else:
            result = {
                "all_running": False,
                "failed_cameras": failed_cameras
            }

        print("[DEBUG] Camera status summary:", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] camera_status:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/camera_history", methods=["GET"])
def camera_history():
    history = get_history(limit=20)
    return jsonify(history)

@api_bp.route('/media/<path:filename>', methods=['GET'])
def serve_media_file(filename):
    media_dir = os.path.abspath("media")
    return send_from_directory(media_dir, filename)

@api_bp.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not all([name, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    role = 0
    conn = get_auth_db_connection()
    cur = conn.cursor()

    try:
        cur.execute('INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                    (name, email, hashed_password, role))
        conn.commit()
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: users.email" in str(e):
            return jsonify({"error": "Email already registered"}), 409
        return jsonify({"error": "Database error"}), 500
    finally:
        conn.close()

    return jsonify({"message": "User registered successfully"})

@api_bp.route('/auth/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not all([email, password]):
            return jsonify({"error": "Email and Password required"}), 400

        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_auth_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, role FROM users WHERE email = ? AND password = ?",
                    (email, hashed_password))
        user = cur.fetchone()
        conn.close()

        if user:
            payload = {
                "id": user["id"],
                "name": user["name"],
                "email": email,
                "role": user["role"]
            }
            token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

            response = jsonify({"message": "Login successful"})
            response.set_cookie(
                "token", token, httponly=True, samesite="Lax", secure=False
            )
            return response
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/auth/me', methods=['GET'])
def get_me():
    token = request.cookies.get('token')

    if not token:
        return jsonify({"error": "Token missing"}), 401

    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_data = {
            "id": decoded["id"],
            "name": decoded["name"],
            "email": decoded["email"],
            "role": decoded["role"]
        }
        return jsonify(user_data)

    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

@api_bp.route('/auth/logout', methods=['POST'])
def logout_user():
    response = jsonify({"message": "Logged out successfully"})
    response.set_cookie("token", "", expires=0, httponly=True, samesite="Lax", secure=False)
    return response

@api_bp.route('/export_data', methods=['GET'])
def export_data():
    try:
        # --- 1. Parse request parameters ---
        vehicle_type = request.args.getlist('vehicleType')
        start_time = request.args.get('startTime')
        end_time = request.args.get('endTime')
        vehicle_number = request.args.get('vehicleNumber')
        whitelist_only = request.args.get('column', '').lower() == 'whitelist'
        employee_id = request.args.get('employeeId')
        output_format = request.args.get('format', 'json')
        underwork_only = request.args.get('column', '').lower() == 'underwork' 

        # --- 2. Fetch data from database ---
        conn = get_violations_connection()
        cur = conn.cursor()
        cur.execute(f"ATTACH DATABASE '{UPLOADED_DB}' AS empdb")

        query = """
            SELECT
                a.plate_number,
                a.vehicle_type,
                a.in_time,
                a.out_time,
                a.image_path,
                a.plate_image_path,
                a.out_image_path,
                a.out_plate_image_path,
                e."Emp ID" AS emp_id
            FROM anpr_alerts a
            LEFT JOIN empdb.employee_master e
            ON LOWER(a.plate_number) = LOWER(e."Vehicle No")
            WHERE 1=1
        """
        params = []
        if whitelist_only:
            query += " AND e.\"Vehicle No\" IS NOT NULL"

        if underwork_only:
            # Only include records with both in_time and out_time
            query += " AND a.in_time IS NOT NULL AND a.out_time IS NOT NULL"
            # And where the employee is whitelisted
            query += " AND e.\"Vehicle No\" IS NOT NULL"

        if employee_id:
            query += " AND e.\"Emp ID\" = ?"
            params.append(employee_id)
        if vehicle_type:
            placeholders = ",".join(["?"] * len(vehicle_type))
            query += f" AND LOWER(a.vehicle_type) IN ({placeholders})"
            params.extend([vt.lower() for vt in vehicle_type])
        if vehicle_number:
            query += " AND LOWER(a.plate_number) LIKE ?"
            params.append(f"%{vehicle_number.lower()}%")
        if start_time and end_time:
            try:
                parser.parse(start_time)
                parser.parse(end_time)
                query += " AND (a.in_time BETWEEN ? AND ? OR a.out_time BETWEEN ? AND ?)"
                params.extend([start_time, end_time, start_time, end_time])
            except ValueError:
                return jsonify({"error": "Invalid datetime format"}), 400
        query += " ORDER BY COALESCE(a.in_time, a.out_time) DESC"
        cur.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
        cur.execute("DETACH DATABASE empdb")
        conn.close()

        if underwork_only:
            filtered_rows = []
            for row in rows:
                if row.get("in_time") and row.get("out_time"):
                    try:
                        in_time = parser.parse(row["in_time"])
                        out_time = parser.parse(row["out_time"])
                        duration_seconds = (out_time - in_time).total_seconds()
                        if duration_seconds < 8 * 3600:
                            filtered_rows.append(row)
                    except Exception as e:
                        print(f"[WARN] Skipping record {row}: {e}")
            rows = filtered_rows

        # --- 3. Transform data for output ---
        for row in rows:
            if row.get("in_time"):
                dt = parser.parse(row["in_time"])
                row["in_time"] = dt.strftime("%d-%m-%Y %I:%M:%S %p")
            if row.get("out_time"):
                dt = parser.parse(row["out_time"])
                row["out_time"] = dt.strftime("%d-%m-%Y %I:%M:%S %p")
            row["fullImage"] = (
                url_for('api.serve_media_file', filename=os.path.basename(row["image_path"]), _external=True)
                if row.get("image_path") else None
            )
            row["plateImage"] = (
                url_for('api.serve_media_file', filename=os.path.basename(row["plate_image_path"]), _external=True)
                if row.get("plate_image_path") else None
            )
            row["outFullImage"] = (
                url_for('api.serve_media_file', filename=os.path.basename(row["out_image_path"]), _external=True)
                if row.get("out_image_path") else None
            )
            row["outPlateImage"] = (
                url_for('api.serve_media_file', filename=os.path.basename(row["out_plate_image_path"]), _external=True)
                if row.get("out_plate_image_path") else None
            )
            row["vehicleNumber"] = row.pop("plate_number")
            row["vehicleType"] = row.pop("vehicle_type")
            row["inTime"] = row.pop("in_time")
            row["outTime"] = row.pop("out_time")
            row["empId"] = row.pop("emp_id")
            row.pop("image_path", None)
            row.pop("plate_image_path", None)
            row.pop("out_image_path", None)
            row.pop("out_plate_image_path", None)

        # --- 4. Generate output ---
        if output_format == 'pdf':
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=30,
                leftMargin=30,
                topMargin=30,
                bottomMargin=30,
                title="Vehicle Report"
            )
            story = []
            styles = getSampleStyleSheet()

            # Custom styles with unique names
            styles.add(ParagraphStyle(
                name='ReportTitle',
                fontSize=16,
                leading=18,
                alignment=1,  # center
                spaceAfter=20,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='VehicleHeader',
                fontSize=12,
                leading=14,
                spaceAfter=5,
                textColor=colors.darkgreen,
                fontName='Helvetica-Bold'
            ))
            styles.add(ParagraphStyle(
                name='DetailText',
                fontSize=10,
                leading=12,
                spaceAfter=5,
                textColor=colors.black
            ))
            styles.add(ParagraphStyle(
                name='ImageTitle',
                fontSize=10,
                leading=12,
                spaceAfter=5,
                textColor=colors.darkblue,
                fontName='Helvetica-Bold'
            ))

            # Add report title
            story.append(Paragraph("VEHICLE ENTRY/EXIT REPORT", styles["ReportTitle"]))
            story.append(HRFlowable(width="100%", thickness=1, lineCap='round', color=colors.darkblue, spaceAfter=10))

            for i, row in enumerate(rows):
                # Vehicle header with border
                vehicle_header = Paragraph(
                    f"Vehicle #{i+1}: {row.get('vehicleNumber', 'N/A')} ({row.get('vehicleType', 'N/A')})",
                    styles["VehicleHeader"]
                )
                story.append(vehicle_header)
                story.append(Spacer(1, 5))

                # Vehicle details table
                details = [
                    ["Vehicle Number", row.get('vehicleNumber', 'N/A')],
                    ["Vehicle Type", row.get('vehicleType', 'N/A')],
                    ["In Time", row.get('inTime', 'N/A')],
                    ["Out Time", row.get('outTime', 'N/A')],
                    ["Employee ID", row.get('empId', 'N/A')],
                ]
                detail_table = Table(details, colWidths=[1.5*inch, 3*inch])
                detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BOX', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(detail_table)
                story.append(Spacer(1, 10))

                # Images section in 2x2 grid
                images = [
                    ("Entry Image", row.get("fullImage")),
                    ("Plate Image", row.get("plateImage")),
                    ("Exit Image", row.get("outFullImage")),
                    ("Exit Plate Image", row.get("outPlateImage")),
                ]

                img_cells = []
                for title, url in images:
                    if url:
                        try:
                            response = requests.get(url)
                            img = Image(BytesIO(response.content), width=2.5*inch, height=2*inch)
                            img.hAlign = 'CENTER'
                            cell = [Paragraph(title, styles["ImageTitle"]), img]
                            img_cells.append(cell)
                        except Exception as e:
                            img_cells.append([
                                Paragraph(title, styles["ImageTitle"]),
                                Paragraph(f"Could not load image: {str(e)}", styles["DetailText"])
                            ])
                    else:
                        img_cells.append([
                            Paragraph(title, styles["ImageTitle"]),
                            Paragraph("No Image Available", styles["DetailText"])
                        ])

                grid_data = [
                    [img_cells[0], img_cells[1]],
                    [img_cells[2], img_cells[3]]
                ]

                image_table = Table(grid_data, colWidths=[3*inch, 3*inch])
                image_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
                    ('BOX', (0, 0), (-1, -1), 0.25, colors.grey),
                ]))
                story.append(image_table)
                story.append(Spacer(1, 10))

                # Separator between vehicles
                if i < len(rows) - 1:
                    story.append(HRFlowable(width="100%", thickness=1, lineCap='round',
                                             color=colors.lightgrey, spaceBefore=10, spaceAfter=10))

            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name="vehicle_report.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/upload_excel', methods=['POST'])
def upload_excel():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({"error": "Only Excel files allowed"}), 400

        content = file.read()
        result = upload_patient_excel_to_db_from_bytes(content)

        if result["status"] == "success":
            return jsonify({"message": result["message"]}), 200
        else:
            return jsonify({"error": result["message"]}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/get_email_config', methods=['GET'])
def get_email_config():
    import json
    config_path = "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        if 'email' not in config:
            return jsonify({"status": "error", "message": "'email' section not found in config"}), 404
        return jsonify({"status": "success", "email_config": config['email']})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read config: {str(e)}"}), 500


@api_bp.route('/update_email_config', methods=['POST'])
def update_email_config():
    import json
    config_path = "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read config: {str(e)}"}), 500

    new_email_config = request.json
    if not new_email_config:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    if 'email' not in config:
        return jsonify({"status": "error", "message": "'email' section not found in config"}), 404

    config['email'].update(new_email_config)

    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        reload_email_config()
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to save config: {str(e)}"}), 500

    return jsonify({"status": "success", "message": "Email config updated and saved"})

from flask import request, jsonify
from db import get_employee_db_connection

# ----------------- USER MANAGEMENT -----------------

@api_bp.route('/auth/users', methods=['GET'])
def get_all_users():
    """Fetch all registered users"""
    conn = get_auth_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, role FROM users ORDER BY id ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows), 200


@api_bp.route('/auth/users/<int:user_id>/role', methods=['PUT'])
def update_user_role(user_id):
    """Update user role (e.g., 0 = user, 1 = admin, 2 = superadmin)"""
    data = request.get_json()
    new_role = data.get("role")

    if new_role is None:
        return jsonify({"error": "Role is required"}), 400

    conn = get_auth_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({"error": "User not found"}), 404

    cur.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
    conn.commit()
    conn.close()

    return jsonify({"message": f"User {user_id} role updated to {new_role}"}), 200

@api_bp.route('/auth/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user by ID"""
    conn = get_auth_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({"error": "User not found"}), 404

    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": f"User {user_id} deleted successfully"}), 200

@api_bp.route('/auth/change-password', methods=['POST'])
def change_password():
    data = request.get_json()
    email = data.get('email')
    old_password = data.get('oldPassword')
    new_password = data.get('newPassword')
    confirm_password = data.get('confirmPassword')

    # Validate input
    if not all([email, old_password, new_password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "New password and confirm password do not match"}), 400

    conn = get_auth_db_connection()
    cur = conn.cursor()

    try:
        # Verify user exists
        cur.execute("SELECT password FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Verify old password
        old_hashed = hashlib.sha256(old_password.encode()).hexdigest()
        if old_hashed != user[0]:
            return jsonify({"error": "Old password is incorrect"}), 403

        # Update password
        new_hashed = hashlib.sha256(new_password.encode()).hexdigest()
        cur.execute("UPDATE users SET password = ? WHERE email = ?", (new_hashed, email))
        conn.commit()

    except Exception as e:
        return jsonify({"error": "Database error"}), 500
    finally:
        conn.close()

    return jsonify({"message": "Password changed successfully"})

@api_bp.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get("email")
    if not email:
        return jsonify({"error": "Email required"}), 400

    conn = get_auth_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    user = cur.fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "User not found"}), 404

    otp = generate_and_store_otp(email)

    subject = "Password Reset OTP - RUAS Parking"
    body = f"Your OTP is {otp}. It is valid for 2 minutes."
    if not send_email(email, subject, body):
        return jsonify({"error": "Failed to send email"}), 500

    return jsonify({"message": "OTP sent to email"})


@api_bp.route('/auth/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("newPassword")
    confirm_password = data.get("confirmPassword")

    if not all([email, otp, new_password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400
    if new_password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    if not verify_otp(email, otp):
        return jsonify({"error": "Invalid or expired OTP"}), 400

    conn = get_auth_db_connection()
    cur = conn.cursor()
    new_hashed = hashlib.sha256(new_password.encode()).hexdigest()
    cur.execute("UPDATE users SET password = ? WHERE email = ?", (new_hashed, email))
    conn.commit()
    conn.close()

    clear_otp(email)

    return jsonify({"message": "Password reset successful"})

@api_bp.route('/alert_summary_counts', methods=['GET'])
def alert_summary_counts():
    """
    Returns a summary count of patients by their status (normal, low, critical).
    """
    try:
        conn = get_violations_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(CASE WHEN status = 'normal' THEN 1 END) as normal,
                COUNT(CASE WHEN status = 'low' THEN 1 END) as low,
                COUNT(CASE WHEN status = 'critical' THEN 1 END) as critical
            FROM iv_tracking
        """)
        result = cur.fetchone()
        conn.close()
        return jsonify({
            "normal": result["normal"] or 0,
            "low": result["low"] or 0,
            "critical": result["critical"] or 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/patients', methods=['GET'])
def get_patients():
    """
    Returns a list of ALL patients, sorted by ward, criticality, or bed.
    Example: /api/patients?sortBy=ward
    """
    try:
        sort_by = request.args.get('sortBy', 'criticality').lower()  # Default: sort by criticality
        valid_sort_fields = ['ward', 'criticality', 'bed']

        conn = get_violations_connection()
        cur = conn.cursor()

        query = """
            SELECT
                patient_id as id,
                patient_name as name,
                doctor,
                ward,
                bed_no as bed,
                fluid_type,
                time_left as left_time,
                flow_rate,
                status as criticality,
                fluid_level as glucose
            FROM iv_tracking
        """

        # Add sorting logic
        if sort_by in valid_sort_fields:
            if sort_by == 'ward':
                query += " ORDER BY CAST(ward AS INTEGER) ASC"  # Sort numerically
            elif sort_by == 'bed':
                query += " ORDER BY CAST(bed_no AS INTEGER) ASC"  # Sort numerically
            elif sort_by == 'criticality':
                query += " ORDER BY CASE status WHEN 'critical' THEN 1 WHEN 'low' THEN 2 WHEN 'normal' THEN 3 END DESC"
        else:
            query += " ORDER BY CAST(bed_no AS INTEGER) ASC"  # Default sort by bed number

        cur.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()

        # Round the glucose value for each patient
        for row in rows:
            if 'glucose' in row and row['glucose'] is not None:
                row['glucose'] = round(float(row['glucose']))

        return jsonify({"patients": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    try:
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = """
            SELECT
                patient_id AS id,
                patient_name AS name,
                doctor,
                ward,
                bed_no AS bed,
                fluid_type,
                time_left AS left_time,
                flow_rate,
                status AS criticality,
                fluid_level AS glucose,
                screenshot
            FROM iv_tracking
            WHERE patient_id = ?
        """
        cur.execute(query, (patient_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Patient not found"}), 404
        row_dict = dict(row)
        if row_dict.get("glucose") is not None:
            row_dict["glucose"] = round(float(row_dict["glucose"]))
        # Generate the full URL for the screenshot
        if row_dict.get("screenshot"):
            # Extract the relative path from the full filesystem path
            screenshot_path = row_dict["screenshot"]
            rel_path = os.path.relpath(screenshot_path, os.path.abspath("media"))
            # Generate the full URL
            row_dict["screenshot"] = url_for('api.serve_media_file', filename=rel_path, _external=True)
        return jsonify({"patients": [row_dict]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/patient/<patient_id>/history', methods=['GET'])
def get_patient_history(patient_id):
    """
    Returns the status history for a specific patient.
    Example response:
    {
        "history": [
            {
                "time": "04:31 PM",
                "filled": "19.59%",
                "timeLeft": "2h",
                "status": "Critical"
            }
        ]
    }
    """
    try:
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = """
            SELECT
                remaining_percentage,
                status,
                time_left,
                timestamp
            FROM patient_history
            WHERE patient_id = ?
            ORDER BY timestamp DESC
        """
        cur.execute(query, (patient_id,))
        rows = cur.fetchall()
        conn.close()

        history = []
        for row in rows:
            # Parse the timestamp to a readable time format
            dt = parser.parse(row["timestamp"])
            time_str = dt.strftime("%I:%M %p")  # e.g., "04:31 PM"
            # Round the remaining_percentage to 2 decimal places for display
            filled = f"{float(row['remaining_percentage']):.2f}%"
            history.append({
                "time": time_str,
                "filled": filled,
                "timeLeft": row["time_left"],
                "status": row["status"].capitalize()  # Capitalize for UI consistency
            })

        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/patient/<patient_id>/glucose', methods=['GET'])
def get_patient_glucose(patient_id):
    """
    Returns glucose level history for a specific patient, formatted for the graph.
    Example response:
    {
        "glucose": [
            { "time": "10:15", "glucose": 85, "status": "Normal" },
            { "time": "10:50", "glucose": 65, "status": "Low" },
            { "time": "10:45", "glucose": 40, "status": "Critical" }
        ]
    }
    """
    try:
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        query = """
            SELECT
                remaining_percentage AS glucose,
                status,
                timestamp
            FROM patient_history
            WHERE patient_id = ?
            ORDER BY timestamp ASC
        """
        cur.execute(query, (patient_id,))
        rows = cur.fetchall()
        conn.close()

        glucose_data = []
        for row in rows:
            dt = parser.parse(row["timestamp"])
            time_str = dt.strftime("%H:%M")  # e.g., "10:15"
            glucose_data.append({
                "time": time_str,
                "glucose": round(float(row["glucose"])),
                "status": row["status"].capitalize()
            })

        return jsonify({"glucose": glucose_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
