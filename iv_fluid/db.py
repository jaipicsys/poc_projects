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

# ----------------- VIOLATIONS (GLUCOSE / IV FLUID TRACKING) DB -----------------
def get_violations_connection():
    conn = sqlite3.connect(VIOLATIONS_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn

def get_patient_data_by_bed(bed_no):
    """Fetch patient data from employee_master.db based on bed_no."""
    conn = get_employee_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employee_master WHERE bed_no = ?", (bed_no,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "patient_id": row["emp_id"],
            "patient_name": row["patient_name"],
            "ward": row["ward"],
            "doctor": row["doctor"],
            "fluid_type": row["fluid_type"],
            "flow_rate": row["flow_rate"],
            "prescribed_volume": row["prescribed_volume"]
        }
    else:
        print(f"[WARN] No patient data found for bed {bed_no}. Using defaults.")
        return {
            "patient_id": f"patient_{bed_no}",
            "patient_name": "Unknown",
            "ward": "WARD1",
            "doctor": "Dr. Smith",
            "fluid_type": "IV",
            "flow_rate": 1.0,
            "prescribed_volume": 1000
        }


def init_violations_db():
    """Create or update violations.db for IV fluid/glucose monitoring."""
    conn = get_violations_connection()
    c = conn.cursor()

    # Table for tracking patient fluid status
    c.execute("""
        CREATE TABLE IF NOT EXISTS iv_tracking (
            patient_id TEXT PRIMARY KEY,
            patient_name TEXT,
            bed_no TEXT,
            ward TEXT,
            doctor TEXT,
            fluid_type TEXT,
            start_time TEXT,
            flow_rate REAL,
            prescribed_volume REAL,
            remaining_percentage REAL,
            fluid_level REAL, 
            time_left TEXT,
            status TEXT,  -- normal / low / critical
            screenshot TEXT, 
            timestamp TEXT
        )
    """)

        # Table for storing simplified patient history snapshots
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            remaining_percentage REAL,
            status TEXT,
            time_left TEXT,
            timestamp TEXT
        )
    """)

    c.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_status
        ON patient_history (patient_id, status)
    """)
    
    conn.commit()
    conn.close()


def db_worker():
    """Background worker that continuously writes queued IV tracking updates."""
    print("[DB WORKER] Background thread started.")

    while not stop_event.is_set() or not insert_queue.empty():
        try:
            data = insert_queue.get(timeout=0.5)
            print(f"[DB WORKER] Writing {data['patient_id']} = {data['remaining_percentage']:.2f}% | {data['status']}")

            # round values cleanly
            data['remaining_percentage'] = round(float(data['remaining_percentage']), 2)
            data['fluid_level'] = round(float(data['fluid_level']), 2)

            # open a *fresh connection* each iteration to ensure visibility
            with get_violations_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")  # improves concurrent safety
                c = conn.cursor()

                c.execute("""
                    INSERT INTO iv_tracking (
                        patient_id, patient_name, bed_no, ward, doctor, fluid_type,
                        start_time, flow_rate, prescribed_volume,
                        remaining_percentage, fluid_level, time_left, status, screenshot, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(patient_id) DO UPDATE SET
                        patient_name = excluded.patient_name,
                        bed_no = excluded.bed_no,
                        ward = excluded.ward,
                        doctor = excluded.doctor,
                        fluid_type = excluded.fluid_type,
                        start_time = excluded.start_time,
                        flow_rate = excluded.flow_rate,
                        prescribed_volume = excluded.prescribed_volume,
                        remaining_percentage = excluded.remaining_percentage,
                        fluid_level = excluded.fluid_level,
                        time_left = excluded.time_left,
                        status = excluded.status,
                        screenshot = excluded.screenshot,
                        timestamp = excluded.timestamp
                """, (
                    data['patient_id'],
                    data['patient_name'],
                    data['bed_no'],
                    data['ward'],
                    data['doctor'],
                    data['fluid_type'],
                    data['start_time'],
                    data['flow_rate'],
                    data['prescribed_volume'],
                    data['remaining_percentage'],
                    data['fluid_level'],
                    data['time_left'],
                    data['status'],
                    data['screenshot'],
                    datetime.datetime.now().isoformat()
                ))

                conn.commit()

            insert_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            # Preserve your debug info for bad payloads
            try:
                pid = data.get('patient_id', '<unknown>') if isinstance(data, dict) else '<no-data>'
            except Exception:
                pid = '<no-data>'
            print(f"[DB WORKER ERROR] patient_id={pid} error={e}")

            # Prevent deadlock even if a task failed
            try:
                insert_queue.task_done()
            except Exception:
                pass
            continue

    print("[DB WORKER] Shutting down gracefully.")



def start_db_thread():
    worker = threading.Thread(target=db_worker, daemon=True)
    worker.start()
    print("[DB THREAD] Started database writer thread")
    return worker

def has_logged_threshold(patient_id, status):
    """Check if a given patient already has a record for this threshold."""
    conn = get_violations_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM patient_history WHERE patient_id = ? AND status = ? LIMIT 1", (patient_id, status))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def insert_glucose_status_async(
    patient_id,
    patient_name,
    bed_no,
    ward,
    doctor,
    fluid_type,
    start_time,
    flow_rate,
    prescribed_volume,
    remaining_percentage,
    fluid_level,
    time_left,
    status,
    screenshot
):
    """Add IV monitoring data asynchronously."""
    print(f"[QUEUE] Queuing update for {patient_id} ({remaining_percentage:.1f}%)")
    insert_queue.put({
        "patient_id": patient_id,
        "patient_name": patient_name,
        "bed_no": bed_no,
        "ward": ward,
        "doctor": doctor,
        "fluid_type": fluid_type,
        "start_time": start_time,
        "flow_rate": flow_rate,
        "prescribed_volume": prescribed_volume,
        "remaining_percentage": remaining_percentage,
        "fluid_level": fluid_level,
        "time_left": time_left,
        "status": status,
        "screenshot": screenshot
    })

def insert_patient_history(patient_id, remaining_percentage, status, time_left):
    """Insert a simplified history record (for trend tracking over time)."""
    try:
        conn = get_violations_connection()
        c = conn.cursor()
        remaining_percentage = round(float(remaining_percentage), 2)
        c.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_patient_status
            ON patient_history (patient_id, status)
        """)

        c.execute("""
            INSERT OR IGNORE INTO patient_history (
                patient_id, remaining_percentage, status, time_left, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            patient_id,
            remaining_percentage,
            status,
            time_left,
            datetime.datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to insert into patient_history: {e}")

def stop_db_thread():
    stop_event.set()
    insert_queue.join()

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

# ----------------- EMPLOYEE MASTER DB -----------------
def get_employee_db_connection():
    conn = sqlite3.connect(UPLOADED_DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_employee_db():
    conn = get_employee_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employee_master (
            emp_id TEXT PRIMARY KEY,
            patient_name TEXT,
            bed_no TEXT,
            ward TEXT,
            doctor TEXT,
            fluid_type TEXT,
            start_time TEXT,
            flow_rate REAL,
            prescribed_volume REAL
        )
    ''')
    conn.commit()
    conn.close()

# ----------------- HISTORY (OPTIONAL, KEEP AS IS) -----------------
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
    c.execute("INSERT INTO camera_history (camera_id, status, timestamp) VALUES (?, ?, ?)",
              (camera_id, status, current_time))
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
