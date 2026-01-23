import os
import sqlite3
import datetime
import threading
import queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUTH_DB = os.path.join(BASE_DIR, 'auth.db')
UPLOADED_DB = os.path.join(BASE_DIR, 'employee_master.db')
HISTORY_DB = os.path.join(BASE_DIR, "history.db")
VIOLATIONS_DB = os.path.join(BASE_DIR, "violations.db")

insert_queue = queue.Queue()
stop_event = threading.Event()

fire_insert_queue = queue.Queue()
fire_stop_event = threading.Event()

# ----------------- Violations DB -----------------
def get_violations_connection():
    conn = sqlite3.connect(VIOLATIONS_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_violations_db():
    conn = get_violations_connection()
    c = conn.cursor()

    # Create base table if it doesn't exist
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT,
            frame_idx INTEGER,
            track_id INTEGER,
            violation TEXT,
            confidence REAL,
            bbox TEXT,
            timestamp TEXT,
            snapshot_path TEXT
        )
    """)
    # ----------------- Fire Events Table -----------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS fire_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_id TEXT,
            frame_idx INTEGER,
            track_id INTEGER,
            event TEXT,
            confidence REAL,
            bbox TEXT,
            timestamp TEXT,
            snapshot_path TEXT
        )
    """)

    # # Fetch existing columns
    # c.execute("PRAGMA table_info(violations)")
    # existing_columns = [col[1] for col in c.fetchall()]

    # # Helper function to add missing columns
    # def add_column_if_missing(col_name, col_def):
    #     if col_name not in existing_columns:
    #         c.execute(f"ALTER TABLE violations ADD COLUMN {col_name} {col_def}")
    #         print(f"[DB] Added missing '{col_name}' column.")

    # # Example: extra fields you might want in future
    # add_column_if_missing("camera_id", "TEXT")
    # add_column_if_missing("vehicle_speed", "REAL DEFAULT 0")
    # add_column_if_missing("fine_amount", "REAL DEFAULT 0")

    conn.commit()
    conn.close()

def db_worker():
    conn = get_violations_connection()
    c = conn.cursor()
    while not stop_event.is_set() or not insert_queue.empty():
        try:
            data = insert_queue.get(timeout=0.5)
            c.execute("""
                INSERT INTO violations (
                    cam_id, frame_idx, track_id, violation, confidence, bbox, timestamp, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['cam_id'], data['frame_idx'], data['track_id'], data['violation'],
                data['confidence'], str(data['bbox']),
                datetime.datetime.now().isoformat(), data['snapshot_path']
            ))
            conn.commit()
            insert_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            # Log and continue so a single bad row doesn't kill the worker
            print(f"[DB WORKER ERROR] {e}")
    conn.close()

def start_db_thread():
    worker = threading.Thread(target=db_worker, daemon=True)
    worker.start()
    f_thread = threading.Thread(target=fire_db_worker, daemon=True)
    f_thread.start()
    return worker, f_thread

def insert_violation_async(cam_id, frame_idx, track_id, violation, confidence, bbox, snapshot_path):
    insert_queue.put({
        'cam_id': cam_id,
        'frame_idx': frame_idx,
        'track_id': track_id,
        'violation': violation,
        'confidence': confidence,
        'bbox': bbox,
        'snapshot_path': snapshot_path
    })

def stop_db_thread():
    stop_event.set()
    insert_queue.join()

# ----------------- FIRE TABLE -----------------

def fire_db_worker():
    conn = get_violations_connection()
    c = conn.cursor()
    while not fire_stop_event.is_set() or not fire_insert_queue.empty():
        try:
            data = fire_insert_queue.get(timeout=0.5)
            c.execute("""
                INSERT INTO fire_events (
                    cam_id, frame_idx, track_id, event, confidence, bbox, timestamp, snapshot_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['cam_id'], data['frame_idx'], data['track_id'], data['event'],
                data['confidence'], str(data['bbox']),
                data.get('timestamp', datetime.datetime.now().isoformat()), data['snapshot_path']
            ))
            conn.commit()
            fire_insert_queue.task_done()
            print("[DEBUG] fire alert added")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[DB WORKER ERROR - FIRE] {e}")
    conn.close()

def insert_fire_event_async(cam_id, frame_idx, track_id, event, confidence, bbox, snapshot_path, timestamp=None):
    fire_insert_queue.put({
        'cam_id': cam_id,
        'frame_idx': frame_idx,
        'track_id': track_id,
        'event': event,
        'confidence': confidence,
        'bbox': bbox,
        'snapshot_path': snapshot_path,
        'timestamp': timestamp
    })


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
