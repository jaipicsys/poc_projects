import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from db import (
    insert_fire_event_async,
    start_db_thread,
    stop_db_thread,
    update_camera_status
)
import json

class FireProcessor:
    def __init__(self, config, camera_id, rtsp_url, on_done_callback=None):
        self.config = config
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.ENABLE_DISPLAY = self.config.get("enable_display", True)

        # Load config.json here (if not passed)
        if not config:
            with open("config.json") as f:
                self.config = json.load(f)

        output_dir = self.config.get("output_dir", "output")
        model_path = self.config.get("fire_model_path", "fire_model.pt")

        self.snapshot_dir = os.path.join(output_dir, camera_id, "fire_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.event_class = "fire_detected"
        self.model = YOLO(model_path)

    def process_video(self):
        """Main fire detection loop with camera status + callback integration"""
        print(f"\n[INFO] Starting FireProcessor for camera {self.camera_id}...")

        cooldown_period = 200  # seconds to ignore new detections after one fire event
        last_fire_time = 0
        first_fire_time = None  # for logging later

        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[WARN] Fire camera {self.camera_id} unavailable. Retrying in 2 minutes...")
                update_camera_status(self.camera_id, "unavailable")
                time.sleep(120)
                continue

            print(f"[INFO] Fire camera {self.camera_id} connected successfully.")
            update_camera_status(self.camera_id, "running")

            try:
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[WARN] Fire camera {self.camera_id}: failed to read frame. Reconnecting...")
                        break

                    results = self.model.predict(frame, conf=0.25, verbose=False)
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            current_time = time.time()

                            # ⏱️ Check cooldown before processing a new detection
                            if current_time - last_fire_time < cooldown_period:
                                continue  # skip detections during cooldown

                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])

                                timestamp = datetime.now().isoformat()
                                snapshot_frame = frame.copy()
                                cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                label = f"{self.event_class} {conf:.2f}"
                                cv2.putText(snapshot_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                                snapshot_path = os.path.join(self.snapshot_dir, f"fire_{frame_idx}.jpg")
                                cv2.imwrite(snapshot_path, snapshot_frame)

                                insert_fire_event_async(
                                    self.camera_id,
                                    frame_idx=frame_idx,
                                    track_id=0,
                                    event=self.event_class,
                                    confidence=conf,
                                    bbox=(x1, y1, x2, y2),
                                    snapshot_path=snapshot_path,
                                    timestamp=timestamp
                                )

                                # 🔥 Update last fire detection time
                                last_fire_time = current_time
                                first_fire_time = timestamp

                                print(f"[INFO][{self.camera_id}] Fire detected! Ignoring further detections for {cooldown_period}s")

                                # prevent multiple inserts in same frame
                                break  

                            # Draw detection boxes for visualization
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{self.event_class} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    frame_idx += 1

                    if self.ENABLE_DISPLAY:
                        window_name = f"Miss Rolls Detection - {self.camera_id}"
                        resized_frame = cv2.resize(frame, (400, 250))
                        cv2.imshow(window_name, resized_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            finally:
                cap.release()
                if self.ENABLE_DISPLAY:
                    cv2.destroyAllWindows()

                update_camera_status(self.camera_id, "stopped")

                if first_fire_time:
                    print(f"[INFO][{self.camera_id}] Last fire detected at frame {frame_idx}, time {first_fire_time}")
                else:
                    print(f"[INFO][{self.camera_id}] No fire detected.")

                print(f"[INFO] Fire camera {self.camera_id} retrying in 2 minutes...\n")
                time.sleep(120)


                # if self.on_done_callback:
                #     self.on_done_callback(self.camera_id)

                print(f"[INFO] Fire camera {self.camera_id} retrying in 2 minutes...\n")
                time.sleep(120)
