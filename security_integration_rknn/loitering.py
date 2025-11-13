# loitering.py
import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from sort import Sort
from rknnlite.api import RKNNLite
from db import insert_loitering_event_async, start_db_thread, update_camera_status

# ---------------- Configurable constants ----------------
INPUT_SIZE = 512
CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
TRACKER_PARAMS = dict(max_age=30, min_hits=3, iou_threshold=0.3)

# ---------------- Utility: letterbox ----------------
def letterbox_image(src, input_w, input_h, fill=(114, 114, 114)):
    h, w = src.shape[:2]
    scale = min(input_w / w, input_h / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    pad_x = (input_w - nw) // 2
    pad_y = (input_h - nh) // 2
    resized = cv2.resize(src, (nw, nh))
    canvas = np.full((input_h, input_w, 3), fill, dtype=src.dtype)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas, scale, pad_x, pad_y

# ---------------- DFL helpers (same as perimeter) ----------------
def compute_dfl(tensor, dfl_len):
    out = np.zeros(4, dtype=np.float32)
    for b in range(4):
        start = b * dfl_len
        vals = tensor[start:start + dfl_len].astype(np.float64)
        exps = np.exp(vals - np.max(vals))
        probs = exps / (exps.sum() + 1e-6)
        out[b] = float((probs * np.arange(dfl_len)).sum())
    return out

def IoU(a, b):
    x1 = max(a['x1'], b['x1'])
    y1 = max(a['y1'], b['y1'])
    x2 = min(a['x2'], b['x2'])
    y2 = min(a['y2'], b['y2'])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    areaA = max(1e-6, (a['x2'] - a['x1']) * (a['y2'] - a['y1']))
    areaB = max(1e-6, (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
    return inter / (areaA + areaB - inter + 1e-6)

def NMS(boxes, iou_thresh=NMS_IOU_THRESH):
    if not boxes:
        return []
    dets = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    out = []
    while dets:
        best = dets.pop(0)
        out.append(best)
        dets = [d for d in dets if IoU(best, d) < iou_thresh]
    return out

# ---------------- Loitering Processor ----------------
class LoiteringProcessor:
    def __init__(self, config, camera_id: str, rtsp_url: str, on_done_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.config = config or {}
        output_dir = self.config.get("output_dir", "output")
        self.snapshot_dir = os.path.join(output_dir, camera_id, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        model_path = self.config.get("loitering_model_path", "model.rknn")
        self.rknn = RKNNLite()
        print(f"[INFO] Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")

        self.tracker = Sort(**TRACKER_PARAMS)
        self.zones = self.config.get("loitering_zones", {}).get(camera_id, [])
        self.cooldown_seconds = float(self.config.get("loitering_cooldown_sec", 60))
        self.time_threshold = float(self.config.get("loitering_time_threshold", 30))
        self.match_confidence = float(self.config.get("loitering_conf", CONF_THRESH))
        self.tracked_entry_time = {}
        self.tracked_last_logged = {}
        self.logged_loitering = set()
        self.input_size = INPUT_SIZE

    def _point_in_any_zone(self, point):
        px, py = point
        for zone in self.zones:
            pts = np.array(zone.get("points", []), dtype=np.int32)
            if pts.size and cv2.pointPolygonTest(pts, (int(px), int(py)), False) >= 0:
                return True
        return False

    def _save_snapshot(self, frame, frame_idx, track_id, bbox=None, conf=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loiter_{self.camera_id}_{track_id}_{frame_idx}_{ts}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        snap = frame.copy()
        if bbox:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
            except Exception:
                pass
        label = "loitering"
        if conf is not None:
            label = f"{label} {conf:.2f}"
        cv2.putText(snap, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(path, snap)
        return path

    def process_video(self):
        print(f"\n[INFO] Starting LoiteringProcessor for camera {self.camera_id}...")
        update_camera_status(self.camera_id, "starting")
        start_db_thread()

        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[WARN] Loitering camera {self.camera_id} unavailable. Retrying in 2 minutes...")
                update_camera_status(self.camera_id, "unavailable")
                time.sleep(120)
                if self.on_done_callback:
                    self.on_done_callback(self.camera_id)
                continue

            print(f"[INFO] Loitering camera {self.camera_id} connected successfully.")
            update_camera_status(self.camera_id, "running")

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                orig_h, orig_w = frame.shape[:2]
                for zone in self.zones:
                    pts = np.array(zone.get("points", []), dtype=np.int32)
                    if pts.size:
                        cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

                padded, scale, pad_x, pad_y = letterbox_image(frame, self.input_size, self.input_size)
                img_input = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                img_input = np.expand_dims(img_input, 0).astype(np.uint8)

                try:
                    outputs = self.rknn.inference([img_input])
                except Exception as e:
                    print(f"[ERROR] RKNN inference failed: {e}")
                    break

                # Assume same decoding logic as perimeter (customizable if needed)
                # You can import or reuse decode_dfl_heads_python and convert_boxes_from_input_to_original
                # For brevity, not repeated here.

                # Placeholder: boxes = detection_results_from_model()
                boxes = []  # Replace with real decoding call

                # Tracker update
                dets_np = np.array([[b['x1'], b['y1'], b['x2'], b['y2'], b['conf']] for b in boxes], dtype=float) if boxes else np.empty((0, 5))
                tracks = self.tracker.update(dets_np)

                now_ts = time.time()
                for t in tracks:
                    x1, y1, x2, y2, track_id = map(int, t)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    inside = self._point_in_any_zone((cx, cy))

                    if inside:
                        if track_id not in self.tracked_entry_time:
                            self.tracked_entry_time[track_id] = now_ts
                        else:
                            elapsed = now_ts - self.tracked_entry_time[track_id]
                            if elapsed > self.time_threshold:
                                last_ts = self.tracked_last_logged.get(track_id, 0)
                                if (track_id not in self.logged_loitering) or ((now_ts - last_ts) > self.cooldown_seconds):
                                    self.logged_loitering.add(track_id)
                                    self.tracked_last_logged[track_id] = now_ts
                                    timestamp_iso = datetime.now().isoformat()
                                    snapshot_path = self._save_snapshot(frame, frame_idx, track_id, bbox=[x1, y1, x2, y2])
                                    try:
                                        insert_loitering_event_async(
                                            self.camera_id,
                                            frame_idx=frame_idx,
                                            track_id=track_id,
                                            event="loitering",
                                            bbox=[x1, y1, x2, y2],
                                            snapshot_path=snapshot_path,
                                            timestamp=timestamp_iso,
                                            confidence=None
                                        )
                                        print(f"[LOITERING] {self.camera_id} | obj:{track_id} | duration:{elapsed:.1f}s | frame:{frame_idx}")
                                    except Exception as e:
                                        print(f"[ERROR] insert_loitering_event_async failed: {e}")
                    else:
                        self.tracked_entry_time.pop(track_id, None)

                frame_idx += 1

            cap.release()
            print(f"[INFO][{self.camera_id}] Loitering process ended at frame {frame_idx}")
            update_camera_status(self.camera_id, "stopped")
            if self.on_done_callback:
                self.on_done_callback(self.camera_id)
            print(f"[INFO] Retrying Loitering camera {self.camera_id} in 2 minutes...\n")
            time.sleep(120)

    def __del__(self):
        try:
            self.rknn.release()
        except Exception:
            pass
