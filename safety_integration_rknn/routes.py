from flask import Blueprint, request, jsonify, send_from_directory, send_file, url_for
from db import get_violations_connection, get_auth_db_connection, get_history, get_all_camera_status
import os
from datetime import datetime
import hashlib
import jwt
import sqlite3
import re
from dateutil import parser
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
import globals

SECRET_KEY = "picsysnexilishrbrdoddaballapur"   
api_bp = Blueprint('api', __name__)
scheduler_instance = None

CAMERA_CONFIG = {
    "cam1": {"camera_name": "cam 1"},
    "cam2": {"camera_name": "cam 2"},
    "cam3": {"camera_name": "cam 3"},
    "cam4": {"camera_name": "cam 4"}
}

@api_bp.route("/ppe/compliance-trends", methods=["GET"])
def get_ppe_compliance_trends_api():
    """
    Returns last 28 days PPE violation trends.
    Only counts actual violations: no_hardhat & no_vest

    {
        "labels": [...dates...],
        "no_hardhat": [...],
        "no_vest": [...]
    }
    """
    try:
        from datetime import datetime, timedelta, time

        conn = get_violations_connection()
        c = conn.cursor()

        labels = []
        no_hardhat_daily = []
        no_vest_daily = []

        today = datetime.now()

        for i in range(28):
            day = today - timedelta(days=(27 - i))
            start = datetime.combine(day.date(), time.min)
            end = datetime.combine(day.date(), time.max)

            labels.append(day.date().isoformat())

            # Count for each violation type
            for violation_type, arr in [
                ('no_hardhat', no_hardhat_daily),
                ('no_vest', no_vest_daily)
            ]:
                c.execute("""
                    SELECT COUNT(*)
                    FROM violations
                    WHERE violation = ?
                    AND timestamp BETWEEN ? AND ?
                """, (violation_type, start.isoformat(), end.isoformat()))
                arr.append(c.fetchone()[0] or 0)

        conn.close()

        return jsonify({
            "labels": labels,
            "no_hardhat": no_hardhat_daily,
            "no_vest": no_vest_daily
        })

    except Exception as e:
        print("[ERROR] get_ppe_compliance_trends_api:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/violation_location", methods=["GET"])
def get_violation_location_api():
    """
    Returns total violations camera-wise using config.json camera names:
    {
        "total": 265,
        "locations": {
            "Warehouse": 120,
            "Entrance": 45,
            ...
        }
    }
    """
    try:
        from app import CONFIG

        conn = get_violations_connection()
        c = conn.cursor()

        # Fetch camera names from config
        camera_names = CONFIG.get("camera_names", {})

        # PPE (only real violations)
        c.execute("""
            SELECT cam_id, COUNT(*)
            FROM violations
            WHERE violation IN ('no_hardhat', 'no_vest')
            GROUP BY cam_id
        """)
        ppe_counts = dict(c.fetchall())

        # Fire violations
        c.execute("""
            SELECT cam_id, COUNT(*)
            FROM fire_events
            GROUP BY cam_id
        """)
        fire_counts = dict(c.fetchall())

        conn.close()

        location_data = {}
        total = 0

        # Combine both tables per camera
        all_cams = set(ppe_counts.keys()) | set(fire_counts.keys())
        for cam in all_cams:
            count = ppe_counts.get(cam, 0) + fire_counts.get(cam, 0)
            total += count
            location_name = camera_names.get(cam, cam)
            location_data[location_name] = count

        return jsonify({
            #"total": total,
            "locations": location_data
        })

    except Exception as e:
        print("[ERROR] get_violation_location_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/ppe/recent_violations", methods=["GET"])
def get_violations_list_api():
    """
    Returns today's PPE violation events with image links.
    {
        "cam1": [
            {
                "violation": "no_helmet",
                "timestamp": "2025-10-16T14:22:10",
                "snapshot": "http://<server>/api/fire_media/<filename>.jpg"
            },
            ...
        ],
        "cam2": [...]
    }
    """
    try:
        from app import CONFIG
        import os
        from datetime import datetime, time

        conn = get_violations_connection()
        c = conn.cursor()

        # --- Define today's time window ---
        now = datetime.now()
        start_of_day = datetime.combine(now.date(), time.min)
        end_of_day = datetime.combine(now.date(), time.max)
        start_str = start_of_day.isoformat()
        end_str = end_of_day.isoformat()

        # --- Fetch today's violation records ---
        c.execute("""
            SELECT cam_id, violation, timestamp, snapshot_path
            FROM violations
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_str, end_str))
        rows = c.fetchall()

        conn.close()

        # --- Group by camera ---
        violation_data = {}
        for cam_id, violation, timestamp, snapshot_path in rows:
            # Build image URL
            if snapshot_path:
                rel_path = os.path.relpath(snapshot_path, "output")
                snapshot_url = url_for(
                    "api.serve_fire_media",
                    filename=rel_path,
                    _external=True
                )
            else:
                snapshot_url = None

            if cam_id not in violation_data:
                violation_data[cam_id] = []

            violation_data[cam_id].append({
                "violation": violation,
                "timestamp": timestamp,
                "snapshot": snapshot_url
            })

        return jsonify(violation_data)

    except Exception as e:
        print("[ERROR] get_violations_list_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/restart_with_test_video", methods=["POST"])
def restart_with_test_video():
    """
    Clears RTSP URLs and sets test video for cam1.
    Then calls CameraManager.restart_server() to restart the backend.
    """
    import json
    import os


    try:
        config_path = "config.json"

        # --- 1. Load config ---
        with open(config_path, "r") as f:
            config = json.load(f)

        # --- 2. Replace rtsp_urls with the test video ---
        config["rtsp_urls"] = {
            "cam1": "/home/jai/udit_backend/fire_integration/output_detected.mp4"
        }

        # --- 3. Save updated config ---
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print("[INFO] Updated config.json with test video path.")

        # --- 4. Restart the server using CameraManager's built-in logic ---
        globals.CAMERA_MANAGER.restart_server()

        # --- 5. Return response (will probably not reach here before restart) ---
        return jsonify({"message": "Config updated. Restarting server..."})

    except Exception as e:
        print("[ERROR] restart_with_test_video:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/media/fire/<path:filename>")
def serve_fire_media(filename):
    """
    Serves fire snapshot images from the output/ directory.
    Example URL:
    http://localhost:5002/api/media/fire/cam3/fire_snapshots/fire_399.jpg
    """
    from flask import send_from_directory
    import os

    base_dir = os.path.join(os.getcwd(), "output")
    return send_from_directory(base_dir, filename)

@api_bp.route("/fire/status", methods=["GET"])
def get_fire_camera_status_api():
    """
    Returns the alert/normal status of each 'fire' camera for today.
    - 'alert' if there is at least 1 fire event today
    - 'normal' if 0 events
    Example:
    {
        "status": {
            "Loading Dock": "alert",
            "Storage": "normal"
        }
    }
    """
    try:
        from app import CONFIG
        from datetime import datetime

        conn = get_violations_connection()
        c = conn.cursor()

        # --- Today's range ---
        today = datetime.now().date()
        start_str = f"{today.isoformat()}T00:00:00"
        end_str = datetime.now().isoformat()

        # --- Get only fire cameras ---
        fire_cameras = {
            cam_id: CONFIG["camera_names"][cam_id]
            for cam_id, cam_type in CONFIG.get("camera_types", {}).items()
            if cam_type == "fire"
        }

        # --- Initialize all fire cameras as 'normal' ---
        status = {cam_name: "normal" for cam_name in fire_cameras.values()}

        # --- Query today's fire counts per camera ---
        c.execute("""
            SELECT cam_id, COUNT(*)
            FROM fire_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY cam_id
        """, (start_str, end_str))
        rows = c.fetchall()
        conn.close()

        # --- Update status based on counts ---
        for cam_id, count in rows:
            if cam_id in fire_cameras:
                cam_name = fire_cameras[cam_id]
                if count > 0:
                    status[cam_name] = "alert"

        result = {"status": status}

        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_fire_camera_status_api:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/violation_count", methods=["GET"])
def get_violation_count_api():
    """
    Returns combined Fire + PPE (only no_hardhat & no_vest) violation counts:
    {
        "total_all": 1250,
        "last_28_days": 375,
        "today": 18
    }
    """
    try:
        from datetime import datetime, timedelta, time

        conn = get_violations_connection()
        c = conn.cursor()

        now = datetime.now()
        start_today = datetime.combine(now.date(), time.min)
        end_today = datetime.combine(now.date(), time.max)

        last_28_start = now - timedelta(days=28)

        # --- PPE counts (only real violations) ---
        ppe_query = """
            SELECT COUNT(*) FROM violations
            WHERE violation IN ('no_hardhat', 'no_vest')
            AND timestamp BETWEEN ? AND ?
        """

        # --- Fire counts ---
        fire_query = """
            SELECT COUNT(*) FROM fire_events
            WHERE timestamp BETWEEN ? AND ?
        """

        # Total all-time PPE
        c.execute("SELECT COUNT(*) FROM violations WHERE violation IN ('no_hardhat', 'no_vest')")
        total_ppe = c.fetchone()[0] or 0

        # Total all-time Fire
        c.execute("SELECT COUNT(*) FROM fire_events")
        total_fire = c.fetchone()[0] or 0

        total_all = total_ppe + total_fire

        # Last 28 days
        c.execute(ppe_query, (last_28_start.isoformat(), now.isoformat()))
        last_28_ppe = c.fetchone()[0] or 0
        c.execute(fire_query, (last_28_start.isoformat(), now.isoformat()))
        last_28_fire = c.fetchone()[0] or 0
        last_28_total = last_28_ppe + last_28_fire

        # Today
        c.execute(ppe_query, (start_today.isoformat(), end_today.isoformat()))
        today_ppe = c.fetchone()[0] or 0
        c.execute(fire_query, (start_today.isoformat(), end_today.isoformat()))
        today_fire = c.fetchone()[0] or 0
        today_total = today_ppe + today_fire

        conn.close()

        return jsonify({
            "total_all": total_all,
            "last_28_days": last_28_total,
            "today": today_total
        })

    except Exception as e:
        print("[ERROR] get_violation_count_api:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/dashboard/ppe_graph", methods=["GET"])
def get_ppe_violations_summary_graph():
    """
    Returns per-day counts (last 28 days including today) of each PPE violation category:
    no_mask, no_gloves, no_shoes, no_helmet.
    Output format:
    {
        "categories": ["no_mask", "no_gloves", "no_shoes", "no_helmet"],
        "data": {
            "2025-09-20": {"no_mask": 2, "no_gloves": 1, "no_shoes": 0, "no_helmet": 3},
            ...
        }
    }
    """
    try:
        from app import CONFIG
        from datetime import datetime, timedelta
        conn = get_violations_connection()
        c = conn.cursor()

        # --- Get 'no_*' categories ---
        categories = CONFIG.get("violation_classes", [])
        violation_categories = [cat for cat in categories if cat.startswith("no_")]

        # --- Date range: last 28 days (including today) ---
        now = datetime.now()
        start_date = (now - timedelta(days=27)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now.replace(hour=23, minute=59, second=59, microsecond=0)

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # --- SQL query to fetch all relevant violations ---
        placeholders = ",".join("?" * len(violation_categories))
        query = f"""
            SELECT violation, timestamp
            FROM violations
            WHERE violation IN ({placeholders})
            AND timestamp BETWEEN ? AND ?
        """
        c.execute(query, (*violation_categories, start_str, end_str))
        rows = c.fetchall()

        # --- Initialize daily summary structure ---
        summary = {}
        for i in range(28):
            day = (start_date + timedelta(days=i)).date().isoformat()
            summary[day] = {cat: 0 for cat in violation_categories}

        # --- Count occurrences per day per category ---
        for violation, timestamp in rows:
            ts_date = datetime.fromisoformat(timestamp).date().isoformat()
            if ts_date in summary:
                if violation in summary[ts_date]:
                    summary[ts_date][violation] += 1

        conn.close()

        # --- Response ---
        result = {
            "categories": violation_categories,
            "data": summary
        }

        # print("[DEBUG] PPE Violations per-day summary:", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_ppe_violations_summary_api:", e)
        return jsonify({"error": str(e)}), 500



@api_bp.route("/fire/counts_graph", methods=["GET"])
def get_fire_counts_by_location_30days_api():
    """
    Returns fire event counts by camera location (name) for the past 28 days.
    Includes only cameras of type 'fire' from CONFIG["camera_types"].
    Format:
    {
        "by_location": {"Loading Dock": 3, "Storage": 1},
        "total": 4
    }
    """
    try:
        from app import CONFIG
        from datetime import datetime, timedelta

        conn = get_violations_connection()
        c = conn.cursor()

        # --- Define date range (last 30 days) ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=28)
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

        # --- Get only fire cameras ---
        fire_cameras = {
            cam_id: CONFIG["camera_names"][cam_id]
            for cam_id, cam_type in CONFIG.get("camera_types", {}).items()
            if cam_type == "fire"
        }

        # --- Initialize all fire locations with 0 ---
        by_location = {
            cam_name: 0 for cam_name in fire_cameras.values()
        }

        # --- Query actual fire event counts ---
        c.execute("""
            SELECT cam_id, COUNT(*) 
            FROM fire_events
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY cam_id
        """, (start_str, end_str))
        rows = c.fetchall()
        conn.close()

        total = 0

        # --- Fill counts only for fire cameras ---
        for cam_id, count in rows:
            if cam_id in fire_cameras:
                cam_name = fire_cameras[cam_id]
                by_location[cam_name] = count
                total += count

        result = {
            "by_location": by_location,
            "total": total
        }

        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_fire_counts_by_location_30days_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/fire/events_today", methods=["GET"])
def get_all_fire_events_api():
    """
    Returns all recorded fire events (latest first).
    Includes:
      - camera_id
      - timestamp
      - snapshot (as full URL)
      - camera_name (if available from CONFIG)
    """
    try:
        import os
        from app import CONFIG
        from flask import url_for

        conn = get_violations_connection()
        c = conn.cursor()

        # --- Query all fire events (latest first) ---
        c.execute("""
            SELECT cam_id, timestamp, snapshot_path
            FROM fire_events
            ORDER BY timestamp DESC
        """)

        rows = c.fetchall()

        # --- Group by camera (latest event per camera) ---
        fire_data = {}
        for cam_id, timestamp, snapshot_path in rows:
            # Keep all events (not just one per camera)
            if snapshot_path:
                rel_path = os.path.relpath(snapshot_path, "output")
                snapshot_url = url_for(
                    "api.serve_fire_media",
                    filename=rel_path,
                    _external=True
                )
            else:
                snapshot_url = None

            if cam_id not in fire_data:
                fire_data[cam_id] = []

            fire_data[cam_id].append({
                "timestamp": timestamp,
                "snapshot": snapshot_url
            })

        # --- Map camera IDs to friendly names ---
        camera_names = CONFIG.get("camera_names", {})
        result = []
        for cam_id, events in fire_data.items():
            result.append({
                "camera_id": cam_id,
                "camera_name": camera_names.get(cam_id, cam_id),
                "events": events
            })

        conn.close()

        # print("[DEBUG] All fire events:", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_all_fire_events_api:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/dashboard/ppe_fire_today", methods=["GET"])
def get_today_ppe_violations_api():
    """
    Returns today's total counts:
    - PPE violations (only 'no_*' categories)
    - Fire detections (from fire_events table)
    - Recent fire image (latest saved snapshot with proper URL)
    """
    try:
        from app import CONFIG
        from datetime import datetime
        import os
        import glob
        from flask import url_for

        conn = get_violations_connection()
        c = conn.cursor()

        # --- Calculate today's time range ---
        today = datetime.now().date()
        start_str = f"{today.isoformat()}T00:00:00"
        end_str = datetime.now().isoformat()

        # --- Identify PPE violation-only categories ---
        categories = CONFIG.get("violation_classes", [])
        violation_categories = [cat for cat in categories if cat.startswith("no_")]

        # --- Count today's PPE violations ---
        total_violations = 0
        for cat in violation_categories:
            c.execute("""
                SELECT COUNT(*) FROM violations
                WHERE violation = ? AND timestamp BETWEEN ? AND ?
            """, (cat, start_str, end_str))
            total_violations += c.fetchone()[0]

        # --- Count today's Fire events ---
        c.execute("""
            SELECT COUNT(*) FROM fire_events
            WHERE timestamp BETWEEN ? AND ?
        """, (start_str, end_str))
        total_fire = c.fetchone()[0]

        conn.close()

        # --- Find the most recent fire snapshot ---
        latest_image = None
        latest_time = 0

        try:
            output_dir = CONFIG.get("output_dir", "output")
            snapshot_dirs = glob.glob(os.path.join(output_dir, "*", "fire_snapshots"))

            for folder in snapshot_dirs:
                images = glob.glob(os.path.join(folder, "*"))
                for img in images:
                    if os.path.isfile(img):
                        mtime = os.path.getmtime(img)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_image = img

            # Convert to URL using your existing route
            if latest_image:
                rel_path = os.path.relpath(latest_image, "output")
                snapshot_url = url_for(
                    "api.serve_fire_media",
                    filename=rel_path,
                    _external=True
                )
            else:
                snapshot_url = None

        except Exception as img_err:
            print("[WARN] Could not fetch latest fire image:", img_err)
            snapshot_url = None

        # --- Final Response ---
        result = {
            "ppe": total_violations,
            "fire": total_fire,
            "recent_fire_image": snapshot_url
        }

        print("[DEBUG] Today's PPE + Fire counts:", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_today_ppe_violations_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/dashboard/violations_by_camera", methods=["GET"])
def get_violations_by_camera_api():
    """
    Returns total PPE violations count grouped by camera,
    considering only 'no_*' categories (e.g. no_helmet, no_mask, etc.)
    and mapping cam_id to friendly names from config.json.
    """
    try:
        from app import CONFIG  # Use global config
        conn = get_violations_connection()
        c = conn.cursor()

        camera_names = CONFIG.get("camera_names", {})
        categories = CONFIG.get("violation_classes", [])

        # --- Filter only 'no_*' violation categories ---
        violation_categories = [cat for cat in categories if cat.startswith("no_")]

        # --- Query counts per camera for only those categories ---
        placeholders = ",".join("?" * len(violation_categories))
        query = f"""
            SELECT cam_id, COUNT(*) AS count
            FROM violations
            WHERE violation IN ({placeholders})
            GROUP BY cam_id
        """
        c.execute(query, violation_categories)
        rows = c.fetchall()
        conn.close()

        # --- Map cam_id -> friendly name ---
        result = {}
        for row in rows:
            cam_id = row["cam_id"]
            count = row["count"]
            cam_name = camera_names.get(cam_id, cam_id)  # fallback to ID if name missing
            result[cam_name] = count

        # print("[DEBUG] Violations by camera (no_* only):", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_violations_by_camera_api:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route("/dashboard/violations_summary", methods=["GET"])
def get_ppe_violations_summary_api():
    """
    Returns total PPE violations, last 28 days violations, and today's violations.
    Only counts 'no_*' categories (e.g. no_helmet, no_mask, etc.)
    """
    try:
        from app import CONFIG
        conn = get_violations_connection()
        c = conn.cursor()
        from datetime import datetime, timedelta

        # --- Get violation categories (only 'no_*') ---
        categories = CONFIG.get("violation_classes", [])
        violation_categories = [cat for cat in categories if cat.startswith("no_")]

        # --- Prepare placeholders for SQL IN clause ---
        placeholders = ",".join("?" * len(violation_categories))

        # --- Calculate time windows ---
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        last_28_days_start = now - timedelta(days=28)

        today_start_str = today_start.isoformat()
        last_28_days_start_str = last_28_days_start.isoformat()
        now_str = now.isoformat()

        # --- Total violations (all time) ---
        query_total = f"""
            SELECT COUNT(*) FROM violations
            WHERE violation IN ({placeholders})
        """
        c.execute(query_total, violation_categories)
        total_all = c.fetchone()[0]

        # --- Violations in the last 28 days ---
        query_28_days = f"""
            SELECT COUNT(*) FROM violations
            WHERE violation IN ({placeholders})
            AND timestamp BETWEEN ? AND ?
        """
        c.execute(query_28_days, (*violation_categories, last_28_days_start_str, now_str))
        total_28_days = c.fetchone()[0]

        # --- Violations today ---
        query_today = f"""
            SELECT COUNT(*) FROM violations
            WHERE violation IN ({placeholders})
            AND timestamp BETWEEN ? AND ?
        """
        c.execute(query_today, (*violation_categories, today_start_str, now_str))
        total_today = c.fetchone()[0]

        conn.close()

        result = {
            "total_all": total_all,
            "last_28_days": total_28_days,
            "today": total_today
        }

        # print("[DEBUG] PPE Violations Summary (no_* only):", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_ppe_violations_summary_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/ppe/violations_percentage", methods=["GET"])
def get_violation_percentage_api():
    """
    Returns the percentage of each violation category (only 'no_' categories)
    relative to the total number of all 'no_' violations in the last 30 days.
    """
    try:
        from app import CONFIG
        from datetime import datetime, timedelta

        conn = get_violations_connection()
        c = conn.cursor()

        categories = CONFIG.get("violation_classes", [])

        # --- Filter to include only violation categories starting with "no_" ---
        violation_categories = [cat for cat in categories if cat.startswith("no_")]

        # --- Time window: last 30 days ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # --- Count per violation category ---
        category_counts = {}
        for cat in violation_categories:
            c.execute("""
                SELECT COUNT(*) FROM violations
                WHERE violation=? AND timestamp BETWEEN ? AND ?
            """, (cat, start_str, end_str))
            category_counts[cat] = c.fetchone()[0]

        conn.close()

        # --- Compute total of all 'no_' violations ---
        total_no_violations = sum(category_counts.values())

        # --- Compute percentage of each ---
        if total_no_violations > 0:
            category_percentages = {
                cat: round((count / total_no_violations) * 100, 2)
                for cat, count in category_counts.items()
            }
        else:
            category_percentages = {cat: 0.0 for cat in violation_categories}

        # print("[DEBUG] 'no_' Violation Percentages (30 days):", category_percentages)
        return jsonify(category_percentages)

    except Exception as e:
        print("[ERROR] get_violation_percentage_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/ppe/violations_count", methods=["GET"])
def get_violation_counts_api():
    """
    Returns structured counts for each PPE item over the last 30 days:
    {
        "helmet": {"total": X, "violation": Y},
        "mask": {"total": X, "violation": Y},
        "shoes": {"total": X, "violation": Y},
        "gloves": {"total": X, "violation": Y}
    }

    Where:
      total = (helmet + no_helmet)
      violation = no_helmet
    """
    try:
        from app import CONFIG
        conn = get_violations_connection()
        c = conn.cursor()
        from datetime import datetime, timedelta

        # --- Define time window ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # --- PPE base categories ---
        base_categories = ["hardhat", "no_hardhat", "vest", "no_vest"]

        result = {}

        # --- For each PPE item ---
        for item in base_categories:
            yes_label = item
            no_label = f"no_{item}"

            # Total = both present + absent detections
            c.execute("""
                SELECT COUNT(*) FROM violations
                WHERE violation IN (?, ?) AND timestamp BETWEEN ? AND ?
            """, (yes_label, no_label, start_str, end_str))
            total = c.fetchone()[0]

            # Violations = only the 'no_' version
            c.execute("""
                SELECT COUNT(*) FROM violations
                WHERE violation = ? AND timestamp BETWEEN ? AND ?
            """, (no_label, start_str, end_str))
            violations = c.fetchone()[0]

            result[item] = {
                "total": total,
                "violation": violations
            }

        conn.close()

        # print("[DEBUG] Last 30 days PPE counts (total + violation):", result)
        return jsonify(result)

    except Exception as e:
        print("[ERROR] get_violation_counts_api:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/violations", methods=["GET"])
def list_violations():
    try:
        # Query params
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 50))
        offset = (page - 1) * limit

        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row 
        cur = conn.cursor()

        # Count query
        cur.execute("SELECT COUNT(*) as total FROM violations")
        total_count = cur.fetchone()["total"]

        # Data query
        cur.execute("""
            SELECT id, cam_id, frame_idx, track_id, violation, confidence, bbox, timestamp, snapshot_path
            FROM violations
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

        rows = [dict(row) for row in cur.fetchall()]
        conn.close()

        # Build structured response
        return jsonify({
            "page": page,
            "limit": limit,
            "total_count": total_count,
            "data": rows
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/count", methods=["GET"])
def get_total_count():
    try:
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row 
        rows = conn.execute("SELECT cam_id, track_id, violation FROM violations").fetchall()
        conn.close()

        unique_violations = set((row["cam_id"], row["track_id"], row["violation"]) for row in rows)

        return jsonify({
            "total_violations": len(unique_violations)
        })

    except Exception as e:
        print("[ERROR] /count:", e)
        return jsonify({"error": str(e)}), 500

@api_bp.route('/export_ppe_data', methods=['GET'])
def export_ppe_data():
    try:
        # --- 1. Parse optional filters ---
        cam_id = request.args.get('cam_id')           # e.g. cam1
        violation = request.args.get('violation')     # e.g. no_helmet
        start_time = request.args.get('startTime')    # ISO-like string (e.g. 2025-10-06T00:00:00)
        end_time = request.args.get('endTime')        # same format
        output_format = request.args.get('format', 'json').lower()  # json or csv

        # --- 2. Build base query ---
        query = """
            SELECT
                id,
                cam_id,
                frame_idx,
                track_id,
                violation,
                confidence,
                bbox,
                timestamp,
                snapshot_path
            FROM violations
            WHERE 1=1
        """
        params = []

        # --- 3. Apply filters dynamically ---
        if cam_id:
            query += " AND cam_id = ?"
            params.append(cam_id)

        if violation:
            query += " AND violation = ?"
            params.append(violation)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC"

        # --- 4. Execute query ---
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()

        # --- 5. Handle empty result ---
        if not rows:
            return jsonify({"message": "No PPE violations found for given filters"}), 404

        # --- 6. Output JSON or CSV ---
        if output_format == 'json':
            return jsonify({
                "count": len(rows),
                "data": rows
            })

        elif output_format == 'csv':
            # Generate CSV dynamically in memory
            import io, csv
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

            csv_data = output.getvalue()
            output.close()

            # Send as downloadable file
            from flask import Response
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={
                    "Content-Disposition": f"attachment; filename=ppe_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )

        else:
            return jsonify({"error": "Invalid format. Use 'json' or 'csv'."}), 400

    except Exception as e:
        print("[ERROR] /export_ppe_data:", e)
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
    media_dir = os.path.abspath("output")
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
        result = upload_employee_excel_to_db_from_bytes(content)

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

# === Employee Master CRUD ===

@api_bp.route("/employees", methods=["GET"])
def get_all_employees():
    """Fetch all employee records from employee_master.db."""
    conn = get_employee_db_connection()
    rows = conn.execute('SELECT rowid, * FROM employee_master').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows]), 200


@api_bp.route("/employees", methods=["POST"])
def add_employee():
    """Add a new employee record."""
    data = request.get_json()
    if not data or not data.get("Vehicle No"):
        return jsonify({"error": "Vehicle No is required"}), 400

    conn = get_employee_db_connection()
    conn.execute(
        '''
        INSERT INTO employee_master 
        ("Emp ID","Name","Department","Function / Employee Group","Employee Email","Supervisor Email","Mobile No","Vehicle No")
        VALUES (?,?,?,?,?,?,?,?)
        ''',
        (
            data.get("Emp ID"),
            data.get("Name"),
            data.get("Department"),
            data.get("Function / Employee Group"),
            data.get("Employee Email"),
            data.get("Supervisor Email"),
            data.get("Mobile No"),
            data.get("Vehicle No"),
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"message": "Employee added successfully"}), 201


@api_bp.route("/employees/<int:rowid>", methods=["DELETE"])
def delete_employee(rowid):
    """Delete an employee by rowid."""
    conn = get_employee_db_connection()
    conn.execute("DELETE FROM employee_master WHERE rowid = ?", (rowid,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Employee deleted"}), 200

@api_bp.route('/ppe/delete/<int:violation_id>', methods=['DELETE'])
def delete_ppe_violation(violation_id):
    try:
        conn = get_violations_connection()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Check if violation exists
        cur.execute("SELECT id FROM violations WHERE id = ?", (violation_id,))
        record = cur.fetchone()
        if not record:
            conn.close()
            return jsonify({"error": f"Violation {violation_id} not found"}), 404

        # Delete the violation record
        cur.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
        conn.commit()
        conn.close()

        return jsonify({
            "message": f"PPE violation {violation_id} deleted successfully"
        }), 200

    except Exception as e:
        print("[ERROR] /ppe/delete:", e)
        return jsonify({"error": str(e)}), 500


@api_bp.route('/anpr/delete/<int:alert_id>', methods=['DELETE'])
def delete_anpr_alert(alert_id):
    try:
        conn = get_violations_connection()
        cur = conn.cursor()

        # Check if record exists
        cur.execute("SELECT id FROM anpr_alerts WHERE id = ?", (alert_id,))
        record = cur.fetchone()
        if not record:
            conn.close()
            return jsonify({"error": "Record not found"}), 404

        # Delete record
        cur.execute("DELETE FROM anpr_alerts WHERE id = ?", (alert_id,))
        conn.commit()
        conn.close()

        return jsonify({"message": f"Record {alert_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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