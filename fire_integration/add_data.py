import random
import datetime
from db import get_violations_connection

# -------------------------------------
# 🧩 Customize your violation records here
# -------------------------------------
violation_records = [
    ("cam1", 111, 1, "no_helmet", 0.8231, "(120, 200, 420, 500)", "output/cam1/snapshots/helmet1.png"),
    ("cam1", 222, 2, "no_mask", 0.7315, "(100, 180, 400, 480)", "output/cam1/snapshots/mask2.png"),
    ("cam2", 333, 3, "no_gloves", 0.6542, "(130, 190, 390, 460)", "output/cam2/snapshots/helmet2.png"),
    ("cam3", 444, 4, "no_shoes", 0.7123, "(140, 210, 420, 520)", "output/cam3/snapshots/helmet3.png"),
    ("cam4", 555, 5, "no_mask", 0.8456, "(150, 220, 440, 530)", "output/cam4/snapshots/mask1.png"),
]

# -------------------------------------
# ⚙️ Database setup
# -------------------------------------
conn = get_violations_connection()
c = conn.cursor()

now = datetime.datetime.now()

# -------------------------------------
# 🧾 Insert each record with random timestamp
# -------------------------------------
for rec in violation_records:
    cam_id, frame_idx, track_id, violation, confidence, bbox, snapshot_path = rec

    # Generate random timestamp in last 15 days (excluding today)
    random_days_ago = random.randint(1, 15)
    random_time = now - datetime.timedelta(
        days=random_days_ago,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    timestamp = random_time.isoformat()

    try:
        c.execute("""
            INSERT INTO violations (
                cam_id, frame_idx, track_id, violation, confidence, bbox, timestamp, snapshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (cam_id, frame_idx, track_id, violation, confidence, bbox, timestamp, snapshot_path))

        conn.commit()
        print(f"✅ Inserted: {violation} for {cam_id} at {timestamp}")
    except Exception as e:
        print(f"[ERROR inserting record] {e}")

conn.close()
print("🎯 All sample violation records inserted successfully.")
