import os
import sqlite3
import datetime
import threading
import queue
import json
import hashlib
import pandas as pd
import io

# ----------------- PATHS -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_DB = os.path.join(BASE_DIR, "events.db")
HISTORY_DB = os.path.join(BASE_DIR, "history.db")
AUTH_DB = os.path.join(BASE_DIR, "auth.db")
EMPLOYEE_DB = os.path.join(BASE_DIR, "employee_master.db")

# ----------------- QUEUES & EVENTS -----------------
loitering_insert_queue = queue.Queue()
perimeter_insert_queue = queue.Queue()
loitering_stop_event = threading.Event()
perimeter_stop_event = threading.Event()
_db_threads_started = False
_db_threads_lock = threading.Lock()

# ----------------- CONNECTION HELPERS -----------------
def get_events_connection():
    conn = sqlite3.connect(EVENTS_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_history_connection():
    conn = sqlite3.connect(HISTORY_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_auth_db_connection():
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def get_employee_db_connection():
    conn = sqlite3.connect(EMPLOYEE_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ----------------- INIT TABLES -----------------
def init_events_db():
    conn = get_events_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS loitering_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT,
            frame_idx INTEGER,
            track_id INTEGER,
            event TEXT,
            duration REAL,
            zone TEXT,
            timestamp TEXT,
            snapshot_path TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS perimeter_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT,
            frame_idx INTEGER,
            track_id INTEGER,
            event TEXT,
            bbox TEXT,
            timestamp TEXT,
            snapshot_path TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

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

def init_employee_db():
    conn = get_employee_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS employee_master (
            "Emp ID" TEXT PRIMARY KEY,
            "Name" TEXT,
            "Department" TEXT,
            "Function / Employee Group" TEXT,
            "Employee Email" TEXT,
            "Supervisor Email" TEXT,
            "Mobile No" TEXT,
            "Vehicle No" TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_all_dbs():
    init_events_db()
    init_history_db()
    init_auth_db()
    init_employee_db()

# ----------------- HELPER FUNCTIONS -----------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------- EVENT HELPERS -----------------
def fetch_events(table_name, limit=None, start_time=None):
    conn = get_events_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    query = f"SELECT * FROM {table_name}"
    params = []
    if start_time:
        query += " WHERE timestamp >= ?"
        params.append(start_time)
    query += " ORDER BY timestamp DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def fetch_counts_per_hour(table_name):
    conn = get_events_connection()
    c = conn.cursor()
    c.execute(f"""
        SELECT strftime('%Y-%m-%d %H:00:00', timestamp) AS hour, COUNT(*) 
        FROM {table_name}
        GROUP BY hour
    """)
    data = c.fetchall()
    conn.close()
    return data

def delete_event(table_name, event_id):
    try:
        conn = get_events_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT id FROM {table_name} WHERE id = ?", (event_id,))
        if not cur.fetchone():
            conn.close()
            return {"status": "error", "message": f"Event {event_id} not found"}, 404
        cur.execute(f"DELETE FROM {table_name} WHERE id = ?", (event_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Event {event_id} deleted successfully"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

# ----------------- EVENTS INSERT -----------------
def insert_loitering_event_async(cam_id, frame_idx, track_id, event, duration, zone, snapshot_path, timestamp):
    loitering_insert_queue.put({
        'cam_id': cam_id,
        'frame_idx': frame_idx,
        'track_id': track_id,
        'event': event,
        'duration': duration,
        'zone': zone,
        'timestamp': timestamp,
        'snapshot_path': snapshot_path
    })

def insert_perimeter_event_async(cam_id, frame_idx, track_id, event, bbox, snapshot_path, timestamp, confidence=None):
    perimeter_insert_queue.put({
        'cam_id': cam_id,
        'frame_idx': frame_idx,
        'track_id': track_id,
        'event': event,
        'bbox': bbox,
        'timestamp': timestamp,
        'snapshot_path': snapshot_path,
        'confidence': confidence
    })

# ----------------- WORKERS -----------------
def loitering_db_worker():
    conn = get_events_connection()
    c = conn.cursor()
    while not loitering_stop_event.is_set() or not loitering_insert_queue.empty():
        try:
            data = loitering_insert_queue.get(timeout=0.5)
            c.execute("""
                INSERT INTO loitering_events (cam_id, frame_idx, track_id, event, duration, zone, timestamp, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['cam_id'], data['frame_idx'], data['track_id'],
                data['event'], data.get('duration'), data.get('zone'),
                data['timestamp'], data['snapshot_path']
            ))
            conn.commit()
            loitering_insert_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[LOITERING DB WORKER ERROR] {e}")
    conn.close()

def perimeter_db_worker():
    conn = get_events_connection()
    c = conn.cursor()
    while not perimeter_stop_event.is_set() or not perimeter_insert_queue.empty():
        try:
            data = perimeter_insert_queue.get(timeout=0.5)
            c.execute("""
                INSERT INTO perimeter_events (cam_id, frame_idx, track_id, event, bbox, timestamp, snapshot_path, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['cam_id'], data['frame_idx'], data['track_id'],
                str(data['event']),
                json.dumps(data['bbox']) if data.get('bbox') else None,
                data['timestamp'], data['snapshot_path'], data.get('confidence')
            ))
            conn.commit()
            perimeter_insert_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[PERIMETER DB WORKER ERROR] {e}")
    conn.close()

def start_db_thread():
    global _db_threads_started
    with _db_threads_lock:
        if _db_threads_started:
            return None, None
        init_all_dbs()
        l_thread = threading.Thread(target=loitering_db_worker, daemon=True, name="loitering_db_worker")
        p_thread = threading.Thread(target=perimeter_db_worker, daemon=True, name="perimeter_db_worker")
        l_thread.start()
        p_thread.start()
        _db_threads_started = True
        return l_thread, p_thread

def stop_db_thread():
    loitering_stop_event.set()
    perimeter_stop_event.set()
    loitering_insert_queue.join()
    perimeter_insert_queue.join()

# ----------------- CAMERA STATUS & HISTORY -----------------
def add_history(camera_id, status):
    conn = get_history_connection()
    c = conn.cursor()
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("""
        INSERT INTO camera_history (camera_id, status, timestamp)
        VALUES (?, ?, ?)
    """, (camera_id, status, current_time))
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

# ----------------- EMPLOYEE UPLOAD -----------------
def upload_employee_excel_to_db_from_bytes(file_content):
    expected_columns = [
        "Emp ID", "Name", "Department", "Function / Employee Group",
        "Employee Email", "Supervisor Email", "Mobile No", "Vehicle No"
    ]

    try:
        df = pd.read_excel(io.BytesIO(file_content))
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]
        if missing_cols:
            return {"status": "error", "message": f"Missing columns: {missing_cols}"}
        if extra_cols:
            return {"status": "error", "message": f"Unexpected columns: {extra_cols}"}

        df.dropna(how="all", inplace=True)

        conn = get_employee_db_connection()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS employee_master")
        columns_sql = ", ".join([f'"{col}" TEXT' for col in expected_columns])
        cursor.execute(f'CREATE TABLE employee_master ({columns_sql})')
        df.to_sql("employee_master", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Data imported successfully"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
