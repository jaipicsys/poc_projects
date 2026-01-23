import os
import sqlite3
import datetime
import threading
import queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUTH_DB = os.path.join(BASE_DIR, 'auth.db')
UPLOADED_DB = os.path.join(BASE_DIR, 'employee_master.db')
HISTORY_DB = os.path.join(BASE_DIR, "history.db")
PATIENT_ALERTS_DB = os.path.join(BASE_DIR, "patient_alerts.db")

insert_queue = queue.Queue()
stop_event = threading.Event()

# ----------------- PATIENTS DB -----------------

def get_patient_alerts_connection():
    conn = sqlite3.connect(PATIENT_ALERTS_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_patient_alerts_db():
    conn = get_patient_alerts_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            event_type TEXT,
            alert_type TEXT,
            cam_id TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    """)
    c.execute("PRAGMA table_info(patient_alerts)")
    existing_cols = [col[1] for col in c.fetchall()]
    if "image_path" not in existing_cols:
        c.execute("ALTER TABLE patient_alerts ADD COLUMN image_path TEXT")
        print("[DB] Added missing 'image_path' column to patient_alerts.")

    conn.commit()
    conn.close()

def insert_patient_alert(person_id, event_type, alert_type, cam_id, image_path):
    conn = get_patient_alerts_connection()
    c = conn.cursor()
    ts = datetime.datetime.now().isoformat()
    c.execute("""
        INSERT INTO patient_alerts (person_id, event_type, alert_type, cam_id, timestamp, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (person_id, event_type, alert_type, cam_id, ts, image_path))
    conn.commit()
    conn.close()

# ----------------- AUTH DB -----------------

def get_auth_db_connection():
    conn = sqlite3.connect(AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_auth_db():
    conn = get_auth_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()

def get_employee_db_connection():
    conn = sqlite3.connect(UPLOADED_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_employee_db():
    conn = get_employee_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employee_master (
            "Emp ID" TEXT,
            "Name" TEXT,
            "Department" TEXT,
            "Function / Employee Group" TEXT,
            "Employee Email" TEXT,
            "Supervisor Email" TEXT,
            "Mobile No" TEXT,
            "Vehicle No" TEXT
        )
    ''')

    conn.commit()
    conn.close()

def get_history_connection():
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_history_db():
    conn = get_history_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS camera_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            status TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_history(camera_id, status):
    conn = get_history_connection()
    c = conn.cursor()
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO camera_history (camera_id, status, timestamp) VALUES (?, ?, ?)", (camera_id, status, current_time))
    conn.commit()
    conn.close()

def get_history(limit=20):
    conn = get_history_connection()
    c = conn.cursor()
    c.execute("SELECT id, camera_id, status, timestamp FROM camera_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    from config_utils import CONFIG
    camera_names = CONFIG.get("camera_names", {})
    # Formatting the timestamp if needed
    return [{
        **dict(row),
        "camera_name": camera_names.get(row["camera_id"], row["camera_id"]), 
        "timestamp": datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S').strftime('%d-%m-%Y %H:%M:%S')
    } for row in rows]

def init_camera_status_table():
    conn = get_history_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS camera_status (
            camera_id TEXT PRIMARY KEY,
            status TEXT,
            last_update TEXT
        )
    ''')
    conn.commit()
    conn.close()

def update_camera_status(camera_id, status):
    conn = get_history_connection()
    c = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO camera_status (camera_id, status, last_update)
        VALUES (?, ?, ?)
        ON CONFLICT(camera_id) DO UPDATE SET
            status=excluded.status,
            last_update=excluded.last_update
    ''', (camera_id, status, now))
    conn.commit()
    conn.close()

def get_all_camera_status():
    conn = get_history_connection()
    c = conn.cursor()
    rows = c.execute("SELECT camera_id, status, last_update FROM camera_status").fetchall()
    conn.close()
    return [dict(row) for row in rows]
