from flask import Blueprint, request, jsonify, send_from_directory, send_file, url_for
from db import (
    get_events_connection,
    fetch_events,
    fetch_counts_per_hour,
    get_history_connection,
    get_auth_db_connection,
    init_events_db,
    init_history_db,
    init_employee_db,
    EMPLOYEE_DB,
    insert_loitering_event_async,
    insert_perimeter_event_async,
    add_history,
    get_history,
    update_camera_status,
    get_all_camera_status,
    fetch_events,
    delete_event,
    upload_employee_excel_to_db_from_bytes
)
import os
import datetime, timedelta
import hashlib
import jwt
import sqlite3
import re
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
from config_utils import reload_email_config, update_report_job, CONFIG, CONFIG_LOCK
import json
import pandas as pd
import io
from email_utils import send_email, generate_and_store_otp, verify_otp, clear_otp
import globals
import db
import csv
from io import StringIO
from flask import make_response
import base64
import cv2
import numpy as np
from threading import Thread
import os, sys, time, subprocess

SECRET_KEY = "picsysnexilishrbrdoddaballapur"   
api_bp = Blueprint('api', __name__)
scheduler_instance = None

CAMERA_CONFIG = {
    "cam1": {"camera_name": "cam 1"},
    "cam2": {"camera_name": "cam 2"},
    "cam3": {"camera_name": "cam 3"},
    "cam4": {"camera_name": "cam 4"}
}

# ---------------------------
# Helpers
# ---------------------------
def json_response(data, status=200):
    return jsonify(data), status

def get_request_arg(name, default=None, cast=None):
    v = request.args.get(name, default)
    if v is None:
        return default
    if cast:
        try:
            return cast(v)
        except:
            return default
    return v

def paginate_list(items, page=1, per_page=25):
    page = max(1, int(page))
    per_page = max(1, int(per_page))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], len(items)

# Small util to safely convert DB row dicts snapshot_path -> URL
def attach_snapshot_urls(rows, media_prefix):
    for r in rows:
        if r.get("snapshot_path"):
            # ensure no leading slashes duplicate
            path = r["snapshot_path"].lstrip("/")
            r["snapshot_url"] = f"/api/media/{media_prefix}/{path}"
    return rows

# ---------------------------
# Restart with test video
# ---------------------------
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
            "cam1": "/home/jai/work/poc_projects/loitering_detection/src/loitering_crop_input.mp4"
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

# ---------------------------
# Media Serving
# ---------------------------
@api_bp.route("/media/loitering/<path:filename>", methods=["GET"])
def get_loitering_media(filename):
    """
    Serves loitering snapshot images from anywhere inside the output directory.
    Example stored path: output/cam2/snapshots/loiter_cam2_zoneA_1_400_20251107_115652.jpg
    """
    # Absolute base directory
    base_dir = os.path.join(os.getcwd(), "output")

    # If the DB entry includes 'output/' at the start, strip it off
    if filename.startswith("output/"):
        filename = filename[len("output/"):]

    # Construct full path
    full_path = os.path.join(base_dir, filename)

    directory = os.path.dirname(full_path)
    file_name = os.path.basename(full_path)

    return send_from_directory(directory, file_name)


@api_bp.route("/media/breaching/<path:filename>", methods=["GET"])
def get_breaching_media(filename):
    """
    Serves breaching snapshot images from the output/ directory.
    """
    base_dir = os.path.join(os.getcwd(), "output")
    return send_from_directory(base_dir, filename)

# ---------------------------
# LOITERING endpoints (using loitering_events table)
# Default listing and events endpoints will use LIMIT 0,30 via db.fetch_events(limit=30)
# ---------------------------
@api_bp.route("/loitering/status", methods=["GET"])
def get_loitering_status():
    """
    Returns a simple boolean: any recent loitering events within the last X minutes.
    """
    minutes = int(request.args.get("minutes", 60))
    cutoff = (datetime.datetime.now() - datetime.timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("loitering_events", start_time=cutoff)
    status = "yes" if rows else "no"
    return json_response({"loitering": status})

@api_bp.route("/loitering/recent_snapshot", methods=["GET"])
def recent_loitering_snapshot():
    """
    Returns the most recent loitering event snapshot (timestamp + URL).
    Uses db.fetch_events to stay consistent with other APIs.
    """
    rows = db.fetch_events("loitering_events")

    if not rows:
        return json_response({"message": "No loitering events found"})

    # Get the most recent event by timestamp
    recent_event = sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)[0]
    timestamp = recent_event.get("timestamp")
    snapshot_path = recent_event.get("snapshot_path")

    snapshot_url = None
    if snapshot_path:
        # Remove any leading 'output/' since the route handles it
        clean_path = snapshot_path[len("output/"):] if snapshot_path.startswith("output/") else snapshot_path
        snapshot_url = url_for("api.get_loitering_media", filename=clean_path, _external=True)

    return json_response({
        "timestamp": timestamp,
        "snapshot_url": snapshot_url
    })

@api_bp.route("/loitering/counts_graph", methods=["GET"])
def get_loitering_counts_graph():
    """
    Returns hourly counts of loitering events for today (00:00 → 23:59).
    Produces 24 data points — one per hour.
    """
    now = datetime.datetime.now()
    today_start = datetime.datetime.combine(now.date(), datetime.time.min)
    all_loitering = db.fetch_events("loitering_events", start_time=today_start.strftime("%Y-%m-%d %H:%M:%S"))

    hourly_counts = []
    hour_labels = []

    for hour in range(24):
        hour_start = today_start + datetime.timedelta(hours=hour)
        hour_end = hour_start + datetime.timedelta(hours=1)

        count = 0
        for r in all_loitering:
            ts_str = r.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.datetime.fromisoformat(ts_str)
            except ValueError:
                try:
                    ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

            if hour_start <= ts < hour_end:
                count += 1

        hourly_counts.append(count)
        hour_labels.append(f"{hour:02d}:00")

    return json_response({
        "hours": hour_labels,
        "loitering_counts": hourly_counts
    })


@api_bp.route("/loitering/events_today", methods=["GET"])
def get_loitering_events_today():
    """
    Returns loitering events from today.
    """
    today_start = datetime.datetime.now().strftime('%Y-%m-%d') + " 00:00:00"
    rows = db.fetch_events("loitering_events", limit=30, start_time=today_start)
    rows = attach_snapshot_urls(rows, "loitering")
    return json_response({"events": rows})

@api_bp.route("/count_today", methods=["GET"])
def count_violations_today():
    """
    Returns count of loitering and breaching violations for today.
    """
    today_start = datetime.datetime.now().strftime('%Y-%m-%d') + " 00:00:00"

    # Fetch today's loitering and perimeter events
    l_today = db.fetch_events("loitering_events", start_time=today_start)
    p_today = db.fetch_events("perimeter_events", start_time=today_start)

    return json_response({
        "loitering_today": len(l_today),
        "breaching_today": len(p_today)
    })

@api_bp.route("/loitering/violations_percentage", methods=["GET"])
def get_loitering_violations_percentage():
    """
    Returns percentage distribution of loitering event types over the last N days.
    """
    days = int(request.args.get("days", 30))
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("loitering_events", start_time=cutoff)
    counts = {}
    for r in rows:
        event_type = r.get("event", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    total = sum(counts.values()) or 1
    percentages = {k: (v / total) * 100 for k, v in counts.items()}
    return json_response({"percentages": percentages})

@api_bp.route("/loitering/violations_count", methods=["GET"])
def get_loitering_violations_count():
    """
    Returns the count of each loitering violation over the last N days.
    """
    days = int(request.args.get("days", 30))
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("loitering_events", start_time=cutoff)
    counts = {}
    for r in rows:
        event_type = r.get("event", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return json_response({"counts": counts})

@api_bp.route("/loitering/violations", methods=["GET"])
def get_loitering_violations():
    """
    Returns a paginated list of loitering violations. Default returns 30 violations.
    """
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))

    rows = db.fetch_events("loitering_events", limit=per_page, offset=(page - 1) * per_page)
    rows = attach_snapshot_urls(rows, "loitering")
    return json_response({"page": page, "per_page": per_page, "total": len(rows), "items": rows})

@api_bp.route("/loitering/count", methods=["GET"])
def loitering_count():
    """
    Returns total loitering count (unlimited).
    """
    rows = db.fetch_events("loitering_events")
    return json_response({"total_loitering": len(rows)})

@api_bp.route("/export_loitering_data", methods=["GET"])
def export_loitering_data():
    fmt = request.args.get("format", "json").lower()
    rows = db.fetch_events("loitering_events")
    rows = attach_snapshot_urls(rows, "loitering")

    if fmt == "csv":
        si = StringIO()
        writer = csv.DictWriter(si, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=loitering_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    else:
        return json_response({"data": rows})

@api_bp.route("/loitering/delete/<int:event_id>", methods=["DELETE"])
def delete_loitering_event(event_id):
    """
    Delete a loitering event.
    """
    db.delete_event("loitering_events", event_id)
    return json_response({"message": "Event deleted successfully."})

# ---------------------------
# BREACHING endpoints (mapped to perimeter_events)
# Default listing and events endpoints will use LIMIT 0,30 via db.fetch_events(limit=30)
# ---------------------------
@api_bp.route("/breaching/status", methods=["GET"])
def get_breaching_status():
    """
    Returns a simple boolean: any recent breaching events within the last X minutes.
    """
    minutes = int(request.args.get("minutes", 60))
    cutoff = (datetime.datetime.now() - datetime.timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("perimeter_events", start_time=cutoff)
    status = "yes" if rows else "no"
    return json_response({"breaching": status})

@api_bp.route("/breaching/recent_snapshot", methods=["GET"])
def recent_breaching_snapshot():
    """
    Returns the most recent breaching (perimeter) event snapshot (timestamp + URL).
    Uses db.fetch_events to stay consistent with other APIs.
    """
    rows = db.fetch_events("perimeter_events")

    if not rows:
        return json_response({"message": "No breaching events found"})

    # Get the most recent event by timestamp
    recent_event = sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)[0]
    timestamp = recent_event.get("timestamp")
    snapshot_path = recent_event.get("snapshot_path")

    snapshot_url = None
    if snapshot_path:
        # Remove any leading 'output/' since the route handles it
        clean_path = snapshot_path[len("output/"):] if snapshot_path.startswith("output/") else snapshot_path
        snapshot_url = url_for("api.get_breaching_media", filename=clean_path, _external=True)

    return json_response({
        "timestamp": timestamp,
        "snapshot_url": snapshot_url
    })


@api_bp.route("/breaching/counts_graph", methods=["GET"])
def get_breaching_counts_graph():
    """
    Returns hourly counts of breaching (perimeter) events for today (00:00 → 23:59).
    Produces 24 data points — one per hour.
    """
    now = datetime.datetime.now()
    today_start = datetime.datetime.combine(now.date(), datetime.time.min)
    all_breaching = db.fetch_events("perimeter_events", start_time=today_start.strftime("%Y-%m-%d %H:%M:%S"))

    hourly_counts = []
    hour_labels = []

    for hour in range(24):
        hour_start = today_start + datetime.timedelta(hours=hour)
        hour_end = hour_start + datetime.timedelta(hours=1)

        count = 0
        for r in all_breaching:
            ts_str = r.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = datetime.datetime.fromisoformat(ts_str)
            except ValueError:
                try:
                    ts = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

            if hour_start <= ts < hour_end:
                count += 1

        hourly_counts.append(count)
        hour_labels.append(f"{hour:02d}:00")

    return json_response({
        "hours": hour_labels,
        "breaching_counts": hourly_counts
    })

@api_bp.route("/breaching/events_today", methods=["GET"])
def breaching_events_today():
    today_start = datetime.datetime.now().strftime('%Y-%m-%d') + " 00:00:00"
    rows = db.fetch_events("perimeter_events", limit=30, start_time=today_start)
    rows = attach_snapshot_urls(rows, "breaching")
    return json_response({"events": rows})

@api_bp.route("/breaching/violations_percentage", methods=["GET"])
def get_breaching_violations_percentage():
    days = int(request.args.get("days", 30))
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("perimeter_events", start_time=cutoff)
    counts = {}
    for r in rows:
        event_type = r.get("event", "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    total = sum(counts.values()) or 1
    percentages = {k: (v / total) * 100 for k, v in counts.items()}
    return json_response({"percentages": percentages})

@api_bp.route("/breaching/violations_count", methods=["GET"])
def breaching_violations_count():
    days = int(request.args.get("days", 30))
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    rows = db.fetch_events("perimeter_events", start_time=cutoff)
    counts = {}
    for r in rows:
        evt = r.get("event", "unknown")
        counts[evt] = counts.get(evt, 0) + 1
    return json_response({"counts": counts})

@api_bp.route("/breaching/violations", methods=["GET"])
def breaching_violations():
    """
    Returns a paginated list of breaching violations. Default returns 30 violations.
    """
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))

    rows = db.fetch_events("perimeter_events", limit=per_page, offset=(page - 1) * per_page)
    rows = attach_snapshot_urls(rows, "breaching")
    return json_response({"page": page, "per_page": per_page, "total": len(rows), "items": rows})

@api_bp.route("/breaching/count", methods=["GET"])
def breaching_count():
    """
    Returns total breaching count (unlimited).
    """
    rows = db.fetch_events("perimeter_events")
    return json_response({"total_breaching": len(rows)})

@api_bp.route("/export_breaching_data")
def export_breaching_data():
    fmt = request.args.get("format", "json").lower()
    rows = db.fetch_events("perimeter_events")
    rows = attach_snapshot_urls(rows, "breaching")

    if fmt == "csv":
        si = io.StringIO()
        if rows:
            writer = csv.DictWriter(si, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=breaching_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    else:
        return json_response({"data": rows})

@api_bp.route("/breaching/delete/<int:event_id>", methods=["DELETE"])
def delete_breaching_event(event_id):
    """
    Deletes a breaching event by ID.
    """
    if db.delete_event("perimeter_events", event_id):
        return json_response({"message": f"Event {event_id} deleted successfully."})
    else:
        return jsonify({"error": f"Event {event_id} not found"}), 404

# ---------------------------
# Generic violations endpoints (combined)
# ---------------------------
@api_bp.route("/violations", methods=["GET"])
def get_combined_violations():
    """
    Returns combined list of violations (loitering and breaching). Supports pagination.
    """
    typ = request.args.get("types", "both")
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 30))

    items = []
    if typ in ("both", "loitering"):
        items.extend(db.fetch_events("loitering_events", limit=per_page, offset=(page - 1) * per_page))
    if typ in ("both", "breaching"):
        items.extend(db.fetch_events("perimeter_events", limit=per_page, offset=(page - 1) * per_page))

    items.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    return json_response({"page": page, "per_page": per_page, "total": len(items), "items": items})

@api_bp.route("/total_count", methods=["GET"])
def total_violations_count():
    """
    Returns total count of loitering and breaching violations combined.
    """
    l = db.fetch_events("loitering_events")
    p = db.fetch_events("perimeter_events")
    return json_response({"loitering": len(l), "breaching": len(p), "total": len(l) + len(p)})

@api_bp.route("/count", methods=["GET"])
def count_violations():
    """
    Returns count of loitering and breaching violations separately.
    """
    l = db.fetch_events("loitering_events")
    p = db.fetch_events("perimeter_events")
    return json_response({
        "loitering": len(l),
        "breaching": len(p)
    })

# ---------------------------
# DASHBOARD endpoints
# ---------------------------
@api_bp.route("/dashboard/loitering_breach_today", methods=["GET"])
def dashboard_loitering_breach_today():
    today_start = datetime.datetime.now().strftime('%Y-%m-%d') + " 00:00:00"
    l = db.fetch_events("loitering_events", start_time=today_start)
    p = db.fetch_events("perimeter_events", start_time=today_start)
    return json_response({
        "loitering_today": len(l),
        "breaching_today": len(p)
    })



@api_bp.route("/all_counts", methods=["GET"])
def all_counts():
    """
    Returns total combined counts of all (loitering + breaching) events for:
    1. Till date (all time)
    2. Last 28 days
    3. Today
    """
    now = datetime.datetime.now()
    today_start = now.strftime("%Y-%m-%d") + " 00:00:00"
    past_28_days = (now - datetime.timedelta(days=28)).strftime("%Y-%m-%d %H:%M:%S")

    # Fetch events
    loitering_all = db.fetch_events("loitering_events")
    breaching_all = db.fetch_events("perimeter_events")

    loitering_28 = db.fetch_events("loitering_events", start_time=past_28_days)
    breaching_28 = db.fetch_events("perimeter_events", start_time=past_28_days)

    loitering_today = db.fetch_events("loitering_events", start_time=today_start)
    breaching_today = db.fetch_events("perimeter_events", start_time=today_start)

    # Calculate combined totals
    total_till_date = len(loitering_all) + len(breaching_all)
    total_28_days = len(loitering_28) + len(breaching_28)
    total_today = len(loitering_today) + len(breaching_today)

    # Response
    data = {
        "till_date_total": total_till_date,
        "last_28_days_total": total_28_days,
        "today_total": total_today
    }

    return json_response(data)


@api_bp.route("/daily_trends", methods=["GET"])
def daily_trends():
    """
    Returns daily counts of loitering and breaching events
    for the past 28 days (each value = total events in 24 hrs).
    """
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=27)

    loitering_counts = []
    breaching_counts = []
    date_labels = []

    # Preload all events once (faster than querying 56 times)
    all_loitering = db.fetch_events("loitering_events", start_time=start_date.strftime("%Y-%m-%d %H:%M:%S"))
    all_breaching = db.fetch_events("perimeter_events", start_time=start_date.strftime("%Y-%m-%d %H:%M:%S"))

    for i in range(28):
        day_start = start_date + datetime.timedelta(days=i)
        next_day = day_start + datetime.timedelta(days=1)

        # Count events falling between start and next day
        loitering_day = [
            e for e in all_loitering
            if day_start.strftime("%Y-%m-%d 00:00:00") <= e["timestamp"] < next_day.strftime("%Y-%m-%d 00:00:00")
        ]
        breaching_day = [
            e for e in all_breaching
            if day_start.strftime("%Y-%m-%d 00:00:00") <= e["timestamp"] < next_day.strftime("%Y-%m-%d 00:00:00")
        ]

        loitering_counts.append(len(loitering_day))
        breaching_counts.append(len(breaching_day))
        date_labels.append(day_start.strftime("%Y-%m-%d"))

    data = {
        "dates": date_labels,
        "loitering_counts": loitering_counts,
        "breaching_counts": breaching_counts
    }

    return json_response(data)



@api_bp.route("/count_by_location", methods=["GET"])
def count_by_location():
    """
    Returns total breaching and loitering events over the past 28 days,
    and per-location breakdown with camera names from config.json.
    """
    # Load camera names dynamically
    config_path = "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        camera_names = config.get("camera_names", {})
    except Exception as e:
        camera_names = {}
        print(f"[WARN] Could not load camera names from config.json: {e}")

    now = datetime.datetime.now()
    past_28_days = (now - datetime.timedelta(days=28)).strftime("%Y-%m-%d %H:%M:%S")

    # Fetch 28-day data
    loitering_events = db.fetch_events("loitering_events", start_time=past_28_days)
    breaching_events = db.fetch_events("perimeter_events", start_time=past_28_days)

    # Initialize per-camera counters
    per_cam = {}

    # Count loitering
    for r in loitering_events:
        cam = r.get("cam_id", "unknown")
        cam_name = camera_names.get(cam, cam)
        per_cam.setdefault(cam_name, {"loitering": 0, "breaching": 0, "total": 0})
        per_cam[cam_name]["loitering"] += 1
        per_cam[cam_name]["total"] += 1

    # Count breaching
    for r in breaching_events:
        cam = r.get("cam_id", "unknown")
        cam_name = camera_names.get(cam, cam)
        per_cam.setdefault(cam_name, {"loitering": 0, "breaching": 0, "total": 0})
        per_cam[cam_name]["breaching"] += 1
        per_cam[cam_name]["total"] += 1

    # Compute global totals
    total_breaching = len(breaching_events)
    total_loitering = len(loitering_events)

    # Sort locations by total (descending)
    sorted_per_cam = dict(sorted(per_cam.items(), key=lambda item: item[1]["total"], reverse=True))

    return json_response({
        "total_breaching_events": total_breaching,
        "total_loitering_events": total_loitering,
        "violations_by_location": sorted_per_cam
    })

@api_bp.route("/dashboard/events_by_camera", methods=["GET"])
def dashboard_events_by_camera():
    """
    Counts events per camera for both loitering and breaching.
    """
    l = db.fetch_events("loitering_events")
    p = db.fetch_events("perimeter_events")
    per_cam = {}

    for r in l:
        cam = r.get("cam_id", "unknown")
        per_cam.setdefault(cam, {"loitering": 0, "breaching": 0})
        per_cam[cam]["loitering"] += 1

    for r in p:
        cam = r.get("cam_id", "unknown")
        per_cam.setdefault(cam, {"loitering": 0, "breaching": 0})
        per_cam[cam]["breaching"] += 1

    return json_response({"by_camera": per_cam})


@api_bp.route("/dashboard/violations_summary", methods=["GET"])
def get_dashboard_violations_summary():
    """
    Provides a summary of violations, including total counts for the past 28 days and today.
    """
    today_start = datetime.datetime.now().strftime('%Y-%m-%d') + " 00:00:00"
    past_28_days = (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d %H:%M:%S')

    loitering_today = len(db.fetch_events("loitering_events", start_time=today_start))
    breaching_today = len(db.fetch_events("perimeter_events", start_time=today_start))

    loitering_past_28 = len(db.fetch_events("loitering_events", start_time=past_28_days))
    breaching_past_28 = len(db.fetch_events("perimeter_events", start_time=past_28_days))

    return json_response({
        "loitering_today": loitering_today,
        "breaching_today": breaching_today,
        "loitering_28_days": loitering_past_28,
        "breaching_28_days": breaching_past_28
    })

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

# ---------------------------
# AUTHENTICATION & USER MANAGEMENT
# ---------------------------
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
            response.set_cookie("token", token, httponly=True, secure=True, samesite="None", path="/", domain=None)
#            response.set_cookie(
#                "token", token, httponly=True, samesite="Lax", secure=False
#            )
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


# ---------------------------
# DATA EXPORT & UPLOAD
# ---------------------------
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
        conn = get_events_connection()
        cur = conn.cursor()
        cur.execute(f"ATTACH DATABASE '{EMPLOYEE_DB}' AS empdb")

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

# ---------------------------
# EMAIL CONFIG
# ---------------------------
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

# ---------------------------
# Misc: delete event generic (alternate routes)
# ---------------------------
@api_bp.route("/loitering_event/delete/<int:event_id>", methods=["DELETE"])
def delete_any_loitering_event(event_id):
    return db.delete_event("loitering_events", event_id)

@api_bp.route("/breaching_event/delete/<int:event_id>", methods=["DELETE"])
def delete_any_breaching_event(event_id):
    return db.delete_event("perimeter_events", event_id)

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


@api_bp.route("/config", methods=["GET"])
def get_camera_config():
    """Return camera-related config data for frontend editing"""
    try:
        with CONFIG_LOCK:
            with open("config.json") as f:
                cfg = json.load(f)

        camera_names = cfg.get("camera_names", {})
        rtsp_urls = cfg.get("rtsp_urls", {})
        camera_types = cfg.get("camera_types", {})
        loitering_zones = cfg.get("loitering_zones", {})
        perimeter_zones = cfg.get("perimeter_zones", {})

        # Build unified camera data
        cameras = {}
        for cam_id in set(camera_names.keys()).union(rtsp_urls.keys()):
            cam_name = camera_names.get(cam_id, cam_id)
            cam_type = camera_types.get(cam_id)
            zones = []

            if cam_type == "loitering":
                zones = loitering_zones.get(cam_id, [])
            elif cam_type == "perimeter":
                zones = perimeter_zones.get(cam_id, [])

            cameras[cam_id] = {
                "name": cam_name,
                "type": cam_type,
                "rtsp_url": rtsp_urls.get(cam_id),
                "zones": zones,
            }

        result = {
            "camera_names": camera_names,
            "rtsp_urls": rtsp_urls,
            "camera_types": camera_types,
            "cameras": cameras,
            "loitering_zones": loitering_zones,
            "perimeter_zones": perimeter_zones,
        }

        return jsonify(result), 200
    except FileNotFoundError:
        return jsonify({"error": "Config file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in config file"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to load config: {str(e)}"}), 500

@api_bp.route("/config/update", methods=["POST"])
def update_camera_config():
    """Update camera-related sections in config.json"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        with CONFIG_LOCK:
            with open("config.json") as f:
                cfg = json.load(f)

            # Update camera_names, rtsp_urls, and camera_types
            for key in ["camera_names", "rtsp_urls", "camera_types","loitering_zones","perimeter_zones"]:
                if key in data:
                    cfg[key] = data[key]

            # Update zones based on cameras[type]
            if "cameras" in data:
                for cam_id, cam_data in data["cameras"].items():
                    cam_type = cam_data.get("type")
                    zones = cam_data.get("zones", [])

                    # Update camera_types if type is provided
                    if cam_type:
                        cfg.setdefault("camera_types", {})[cam_id] = cam_type

                    # Update zones based on type
                    if cam_type == "loitering":
                        cfg.setdefault("loitering_zones", {})[cam_id] = zones
                    elif cam_type == "perimeter":
                        cfg.setdefault("perimeter_zones", {})[cam_id] = zones

            # Remove zones for deleted cameras
            for key in ["camera_names", "rtsp_urls", "camera_types"]:
                for cam_id in set(cfg.get(key, {})).difference(data.get(key, {})):
                    if cam_id in cfg.get("loitering_zones", {}):
                        del cfg["loitering_zones"][cam_id]
                    if cam_id in cfg.get("perimeter_zones", {}):
                        del cfg["perimeter_zones"][cam_id]

            # Save updated config
            with open("config.json", "w") as f:
                json.dump(cfg, f, indent=4)

            # Update memory
            CONFIG.update(cfg)

        # Restart backend gracefully
        Thread(target=restart_backend, daemon=True).start()
        return jsonify({"message": "Config updated successfully, restarting backend..."}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to update config: {str(e)}"}), 500

def restart_backend():
    """Restart the Flask backend safely"""
    print("[INFO] Restarting backend after config update...")
    time.sleep(1)
    python = sys.executable
    subprocess.Popen([python, "app.py"])
    os._exit(0)


@api_bp.route("/camera_frame/<cam_id>", methods=["GET"])
def get_camera_frame(cam_id):
    """Return a single frame from the camera RTSP URL as base64 image."""
    try:
        with open("config.json") as f:
            cfg = json.load(f)

        rtsp_url = cfg.get("rtsp_urls", {}).get(cam_id)
        if not rtsp_url:
            return jsonify({"error": f"RTSP URL not found for camera: {cam_id}"}), 404

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"error": f"Failed to open RTSP stream for camera: {cam_id}"}), 500

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({"error": f"Failed to read frame from camera: {cam_id}"}), 500

        # Draw ROI zones based on camera type
        camera_types = cfg.get("camera_types", {})
        cam_type = camera_types.get(cam_id)
        zones = []
        color = (0, 255, 0)  # Default: Green

        if cam_type == "loitering":
            zones = cfg.get("loitering_zones", {}).get(cam_id, [])
            color = (255, 0, 0)  # Blue for loitering
        elif cam_type == "perimeter":
            zones = cfg.get("perimeter_zones", {}).get(cam_id, [])
            color = (0, 0, 255)  # Red for perimeter

        for zone in zones:
            points = zone.get("points", [])
            if points:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                cv2.putText(
                    frame,
                    zone.get("name", ""),
                    (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        # Resize preview
        h, w = frame.shape[:2]
        frame_resized = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))
        _, buffer = cv2.imencode(".jpg", frame_resized)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"image": img_base64})
    except FileNotFoundError:
        return jsonify({"error": "Config file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in config file"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to fetch camera frame: {str(e)}"}), 500