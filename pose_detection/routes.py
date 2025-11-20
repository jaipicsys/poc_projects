from flask import Blueprint, request, jsonify, send_from_directory, send_file, url_for
from db import get_auth_db_connection, get_history, get_all_camera_status, get_patient_alerts_connection
import os
from datetime import datetime, timedelta
import hashlib
import jwt
import sqlite3
import re
from dateutil import parser
from db import get_auth_db_connection, UPLOADED_DB, get_patient_alerts_connection
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

@api_bp.route("/update_state", methods=["POST"])
def update_alert_state():
    """
    Update the alert_type of the latest alert for a camera.
    Body:
    {
        "cam_id": "cam2",
        "new_state": "normal"   # can be anything now
    }
    """
    try:
        data = request.get_json()
        cam_id = data.get("cam_id")
        new_state = data.get("new_state")

        if not cam_id or not new_state:
            return jsonify({"error": "cam_id and new_state are required"}), 400

        conn = get_patient_alerts_connection()
        c = conn.cursor()

        # Get the latest alert for this camera
        c.execute("""
            SELECT id FROM patient_alerts
            WHERE cam_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (cam_id,))
        row = c.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "No alerts found for this camera"}), 404

        alert_id = row["id"]

        # Update state (now unrestricted)
        c.execute("""
            UPDATE patient_alerts
            SET alert_type = ?
            WHERE id = ?
        """, (new_state, alert_id))

        conn.commit()
        conn.close()

        return jsonify({
            "message": "Alert updated successfully",
            "cam_id": cam_id,
            "new_state": new_state
        })

    except Exception as e:
        print("[ERROR] update_alert_state:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/latest_image", methods=["GET"])
def get_latest_camera_alert():
    """
    Returns the latest alert for a specific camera.
    Query Param: cam_id
    Response:
    {
        "cam_id": "cam3",
        "timestamp": "...",
        "alert_type": "critical",
        "image_url": "http://..."
    }
    """
    try:
        cam_id = request.args.get("cam_id")
        if not cam_id:
            return jsonify({"error": "cam_id is required"}), 400

        conn = get_patient_alerts_connection()
        c = conn.cursor()

        # Fetch latest alert for that camera
        c.execute("""
            SELECT *
            FROM patient_alerts
            WHERE cam_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (cam_id,))

        row = c.fetchone()
        conn.close()

        if not row:
            return jsonify({"message": "No alerts found for this camera"}), 404

        # Build image URL
        image_url = (
            url_for(
                "api.serve_media_file",
                filename = row["image_path"].replace("media/", ""),
                _external=True
            )
            if row["image_path"] else None
        )

        result = {
            "cam_id": row["cam_id"],
            "timestamp": row["timestamp"],
            "alert_type": row["alert_type"],
            "image_url": image_url
        }

        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_latest_camera_alert:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/status_counts", methods=["GET"])
def get_patient_alert_status_counts():
    """
    Returns counts of alerts grouped by status (alert_type).
    Example:
    {
        "normal": 23,
        "low": 15,
        "critical": 7
    }
    """
    try:
        conn = get_patient_alerts_connection()
        c = conn.cursor()

        # Group by alert_type and count entries
        c.execute("""
            SELECT alert_type, COUNT(*) as count
            FROM patient_alerts
            GROUP BY alert_type
        """)

        rows = c.fetchall()
        conn.close()

        # Build JSON result
        result = {}
        for row in rows:
            result[row["alert_type"]] = row["count"]

        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_patient_alert_status_counts:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/patient_alerts", methods=["GET"])
def get_current_patient_alerts_api():
    """
    Returns ONLY the latest alert per camera.
    Supports sorting:
        - ?sort=timestamp     (default)
        - ?sort=criticality   (critical > low > normal)
    """
    try:
        from app import CONFIG
        from flask import request

        sort_mode = request.args.get("sort", "timestamp")

        conn = get_patient_alerts_connection()
        c = conn.cursor()

        # Fetch latest alert per camera
        c.execute("""
            SELECT cam_id, alert_type, timestamp
            FROM patient_alerts
            WHERE id IN (
                SELECT MAX(id) FROM patient_alerts GROUP BY cam_id
            )
        """)

        rows = c.fetchall()
        conn.close()

        camera_names = CONFIG.get("camera_names", {})

        data = []
        for row in rows:
            state = (row["alert_type"] or "").strip().lower()   # <-- FIX HERE

            data.append({
                "cam_id": row["cam_id"],
                "camera_name": camera_names.get(row["cam_id"], row["cam_id"]),
                "state": state,
                "timestamp": row["timestamp"]
            })

        # Sorting logic
        if sort_mode == "timestamp":
            data.sort(key=lambda x: x["timestamp"], reverse=False)

        elif sort_mode == "criticality":
            priority = {
                "critical": 1,
                "low": 2,
                "normal": 3
            }
            data.sort(key=lambda x: priority.get(x["state"], 999))

        return jsonify(data)

    except Exception as e:
        print("[ERROR] get_current_patient_alerts_api:", e)
        return jsonify({"error": str(e)}), 500


def generate_mjpeg(camera_id):
    """Serve JPEG frames from the global frame buffer."""
    while True:
        with frames_lock:
            frame = latest_frames.get(camera_id)

        if frame is None:
            time.sleep(0.03)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() + b"\r\n")

        time.sleep(0.03)

@api_bp.route("/live/<camera_id>", methods=["GET"])
def api_live_video(camera_id):
    """MJPEG Live Streaming API"""
    from app import latest_frames  # safe lazy import to avoid circular import

    if camera_id not in latest_frames:
        return jsonify({"error": "Invalid camera ID"}), 404

    return Response(
        generate_mjpeg(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@api_bp.route("/dashboard/alert_states", methods=["GET"])
def camera_states():
    try:
        # Get names from config.json
        try:
            with open("config.json", "r") as f:
                cfg = json.load(f)
            camera_names = cfg.get("camera_names", {})
            camera_ids = list(camera_names.keys())
        except Exception:
            return jsonify({"error": "camera_names missing in config.json"}), 500

        result = []

        for cam_id in camera_ids:
            state = get_latest_state_for_camera(cam_id)
            result.append({
                "camera_id": cam_id,
                "name": camera_names.get(cam_id, cam_id),
                "state": state or "unknown"
            })

        return jsonify(result)

    except Exception as e:
        print("[ERROR] camera_states:", e)
        return jsonify({"error": str(e)}), 500


# @api_bp.route("/camera_status", methods=["GET"])
# def camera_status():
#     data = get_all_camera_status()

#     # Load names from config.json
#     try:
#         with open("config.json", "r") as f:
#             cfg = json.load(f)
#         camera_names = cfg.get("camera_names", {})
#     except Exception as e:
#         camera_names = {}

#     # Attach friendly name to each record
#     for item in data:
#         cam_id = item.get("camera_id")
#         item["name"] = camera_names.get(cam_id, cam_id)

#     return jsonify(data)


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