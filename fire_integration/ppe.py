import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from db import insert_violation_async, update_camera_status
from datetime import datetime
import time


class PPEProcessor:

    def __init__(self, config, camera_id: str, rtsp_url: str, on_done_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.ENABLE_DISPLAY = config.get("enable_display", True)

        # Load YOLO model
        model_path = config.get("model_path", "ppe.pt")
        self.model = YOLO(model_path)

        # Output structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = config.get("output_dir", "output")
        camera_dir = os.path.join(output_dir, camera_id)
        snapshot_dir = os.path.join(camera_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        self.snapshot_dir = snapshot_dir
        self.violation_classes = config.get("violation_classes", [])
        self.tracker = DeepSort(max_age=50, n_init=5)


    def process_video(self):
        print(f"\n[INFO] Starting processing loop for camera {self.camera_id}...")

        while True:
            cap = cv2.VideoCapture(self.rtsp_url)

            if not cap.isOpened():
                print(f"[WARN] Camera {self.camera_id} unavailable. Retrying in 2 minutes...")
                update_camera_status(self.camera_id, "unavailable")
                time.sleep(120)
                continue

            print(f"[INFO] Camera {self.camera_id} connected successfully.")
            update_camera_status(self.camera_id, "running")

            try:
                frame_idx = 0
                seen_violations = set()
                total_violations = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[WARN] Failed to read frame from {self.camera_id}. Reconnecting...")
                        break

                    # YOLO inference
                    results = self.model(frame, verbose=False)[0]

                    # Prepare detections for tracker
                    detections = [
                        (
                            [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                            float(score),
                            self.model.names[int(cls_id)].lower().replace("-", "_"),
                        )
                        for box, cls_id, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf)
                    ]

                    # DeepSort tracker update
                    tracks = self.tracker.update_tracks(detections, frame=frame)

                    # Process all tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        cls_name = track.det_class.lower().replace("-", "_") if track.det_class else "unknown"
                        score = track.det_conf if track.det_conf else 0.0
                        x1, y1, x2, y2 = map(int, track.to_ltrb())

                        # Draw bounding boxes
                        color = (0, 0, 255) if cls_name in self.violation_classes else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name.upper()} ID:{track_id} {score:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Check violation
                        key = f"{track_id}_{cls_name}"
                        if cls_name in self.violation_classes and key not in seen_violations:
                            seen_violations.add(key)
                            total_violations += 1

                            snapshot_path = os.path.join(self.snapshot_dir, f"{cls_name}_{track_id}.jpg")
                            cv2.imwrite(snapshot_path, frame)

                            insert_violation_async(
                                self.camera_id,
                                frame_idx,
                                track_id,
                                cls_name,
                                score,
                                [x1, y1, x2, y2],
                                snapshot_path,
                            )

                            print(f"[VIOLATION] {self.camera_id} | {cls_name} | Track: {track_id} | Frame: {frame_idx}")

                    # Display output window
                    frame_idx += 1
                    if self.ENABLE_DISPLAY:
                        window_name = f"PPE Detection - {self.camera_id}"
                        resized_frame = cv2.resize(frame, (640, 360))
                        cv2.imshow(window_name, resized_frame)

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            finally:
                cap.release()
                if self.ENABLE_DISPLAY:
                    cv2.destroyAllWindows()

                print(f"\n[INFO] Processing completed for camera {self.camera_id}")
                print(f"[STATS] Total frames processed: {frame_idx}")
                print(f"[STATS] Total violations: {total_violations}")

                update_camera_status(self.camera_id, "stopped")

                print(f"[INFO] Retrying camera {self.camera_id} in 2 minutes...\n")
                time.sleep(120)
