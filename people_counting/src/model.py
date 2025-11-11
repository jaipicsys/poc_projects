import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import sqlite3
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================
VIDEO_PATH = "data/input.mp4"
OUTPUT_PATH = "output/output_tracked.mp4"
DB_PATH = "output/people_count.db"

# Floor area points (entry/exit corners)
image_points = np.array([
    [280, 1070],   # bottom-left entry
    [2300, 1050],  # bottom-right exit
    [2300, 50],   # top-right exit
    [400, 400]     # top-left entry
], dtype=np.float32)

# Convert points to integer tuples
bottom_left = tuple(map(int, image_points[0]))   # entry bottom
bottom_right = tuple(map(int, image_points[1]))  # exit bottom
top_right = tuple(map(int, image_points[2]))     # exit top
top_left = tuple(map(int, image_points[3]))      # entry top

# YOLO + DeepSORT
model = YOLO("model/yolov8m.pt")
tracker = DeepSort(max_age=30)

# DB setup
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS people_count
             (timestamp TEXT, total_unique INTEGER, current_inside INTEGER,
              entered INTEGER, exited INTEGER)''')
conn.commit()

# Video setup
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# ==============================
# Counting variables
# ==============================
all_ids = set()
current_inside = 3
entered_ids = set()
exited_ids = set()
last_positions = {}  # track_id -> last x-coordinate for crossing lines

# ==============================
# Processing loop
# ==============================
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == "person" and conf > 0.4:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id

        # Use bottom-center of bounding box
        bottom_center_x = int((x1 + x2)/2)

        # Initialize last_positions for new track IDs
        if track_id not in last_positions:
            # Start outside entry line (left side) to allow first crossing
            last_positions[track_id] = bottom_center_x - 50

        prev_x = last_positions[track_id]

        # Add to total unique IDs
        all_ids.add(track_id)

        # Entry line (left)
        if prev_x < top_left[0] <= bottom_center_x:
            if track_id not in entered_ids:
                entered_ids.add(track_id)
                current_inside += 1

        # Exit line (right)
        elif prev_x > top_right[0] >= bottom_center_x:
            if track_id not in exited_ids:
                exited_ids.add(track_id)
                current_inside -= 1
                current_inside = max(current_inside, 0)

        # Update last position
        last_positions[track_id] = bottom_center_x

        # Draw box + ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        #cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1)-10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


        # Prepare count text
        text = f"Count: {current_inside}  Entered: {len(entered_ids)}  Exited: {len(exited_ids)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.3
        thickness = 6

        # Get size of the text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Set text position
        x, y = 10, 50

        # Draw black rectangle as background
        cv2.rectangle(
            frame,
            (x - 10, y - text_height - 10),             # top-left corner
            (x + text_width + 10, y + baseline + 10),   # bottom-right corner
            (0, 0, 0),                                  # black color
            -1                                          # filled
        )

        # Draw text on top of the rectangle
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 0),   # yellow text color
            thickness
        )


    # Save frame to output video
    out.write(frame)

    # Save stats to DB
    timestamp = datetime.now().isoformat()
    c.execute('INSERT INTO people_count VALUES (?,?,?,?,?)',
              (timestamp, len(all_ids), current_inside, len(entered_ids), len(exited_ids)))
    conn.commit()

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
conn.close()

print(f"Total unique people detected: {len(all_ids)}")
print(f"Total entered: {len(entered_ids)}")
print(f"Total exited: {len(exited_ids)}")
print(f"Final inside count: {current_inside}")
