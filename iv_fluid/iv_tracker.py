# iv_tracker.py (updated)
import os
import cv2
import time
import numpy as np
import sqlite3
from datetime import datetime

timestamp = datetime.now().isoformat()

from collections import deque
from db import insert_glucose_status_async, update_camera_status, get_employee_db_connection, insert_patient_history, has_logged_threshold, get_patient_data_by_bed
from config_utils import CONFIG
import onnxruntime as ort
import asyncio

class IVFluidTracker:
    def __init__(self, config, camera_id: str, rtsp_url: str, bed_no: str, top_cut_percent: float = None, bottom_cut_percent: float = None, on_done_callback=None):
        self.config = config
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.bed_no = bed_no
        self.on_done_callback = on_done_callback
        cut_config = config.get("cut_percent", {})
        default_cut = config.get("default_cut_percent", {"top": 0, "bottom": 12})

        cam_cut = cut_config.get(camera_id, {})
        self.top_cut_percent = top_cut_percent if top_cut_percent is not None else cam_cut.get("top", default_cut.get("top"))
        self.bottom_cut_percent = bottom_cut_percent if bottom_cut_percent is not None else cam_cut.get("bottom", default_cut.get("bottom"))
        # annotation now means "attempt to show frames when a display is available"
        self.annotation = config.get("draw_annotations_on_display", False)
        self.FLUID_THRESHOLDS = config["fluid_level_thresholds"]
        self.level_buffer = deque(maxlen=5)
        self.last_alert = None
        self.last_box = None
        self.model_path = config.get("onnx_model_path", "iv_bag2.onnx")


        # Track whether we actually created any windows (so we can call destroy only when needed)
        self._display_window_open = False

        print(f"🚀 Loading YOLOv8 ONNX model for {self.camera_id} ...")
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.patient_data = self._get_patient_data_from_db()

    # ---------------------------------------------------------------
    def _is_display_available(self):
        """
        Returns True if a display server seems available (common on desktops),
        or False on headless servers. Environment-based check is simple and robust.
        You can also override this by setting an env var SHOW_FRAMES=1 to force display.
        """
        if os.environ.get("SHOW_FRAMES") == "1":
            return True
        return "DISPLAY" in os.environ and bool(os.environ.get("DISPLAY"))

    def maybe_show_frame(self, winname: str, frame):
        """
        Safely attempt to show a frame. Will not crash in headless environments.
        Only shows frames when both self.annotation is True and display is available.
        """
        if not self.annotation:
            return

        if not self._is_display_available():
            # Display not available; skip showing frames
            return

        try:
            cv2.imshow(winname, frame)
            self._display_window_open = True
            # waitKey is required to process window events; keep it short
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # If user presses q in the GUI, we exit the loop by raising an exception
                # The exception will be caught in your thread wrapper or camera loop.
                raise KeyboardInterrupt("User requested quit via GUI")
        except Exception as e:
            # Don't let display issues crash the whole process
            print(f"[WARN] Could not show frame for {self.camera_id}: {e}")
            self._display_window_open = False

    # -------------------------------------------------------------------
    def _get_patient_data_from_db(self):
        try:
            return get_patient_data_by_bed(self.bed_no)
        except Exception as e:
            print(f"[WARN] Failed to fetch patient data for bed {self.bed_no}: {e}")
            # fallback defaults
            return {
                "patient_id": f"patient_{self.bed_no}",
                "patient_name": "Unknown",
                "ward": "WARD1",
                "doctor": "Dr. Smith",
                "fluid_type": "IV",
                "flow_rate": 1.0,
                "prescribed_volume": 1000
            }

    # -------------------------------------------------------------------
    def detect_iv_bag_roi(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
        inp = blob.astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        out = self.session.run(None, {input_name: inp})[0][0]

        best_conf, best_box = 0, None
        for i in range(out.shape[1]):
            x, y, bw, bh, conf = out[:, i]
            if conf > best_conf:
                best_conf = conf
                cx, cy = (x / 640.0) * w, (y / 640.0) * h
                bw, bh = (bw / 640.0) * w, (bh / 640.0) * h
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                best_box = (x1, y1, x2, y2)

        if best_conf < 0.5 and self.last_box is not None:
            return self.last_box, best_conf
        else:
            self.last_box = best_box
            return best_box, best_conf

    # -------------------------------------------------------------------
    def detect_fluid_level(self, img, box):
        x1, y1, x2, y2 = box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            return img, -1

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return img, -1

        h = roi.shape[0]
        eff_top = int((self.top_cut_percent / 100.0) * h)
        eff_bottom = int((1 - (self.bottom_cut_percent / 100.0)) * h)
        eff_roi = roi[eff_top:eff_bottom, :]

        gray = cv2.cvtColor(eff_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        gray = cv2.equalizeHist(gray)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y)
        row_profile = np.sum(grad_y, axis=1)
        if np.max(row_profile) == 0:
            return img, -1

        row_profile = row_profile / (np.max(row_profile)+1e-8)
        row_profile = cv2.GaussianBlur(row_profile.astype(np.float32).reshape(-1,1), (1,9), 0).flatten()
        fluid_row_rel = len(row_profile) - np.argmax(row_profile[::-1])
        fluid_row_rel = int(np.clip(fluid_row_rel, 0, eff_roi.shape[0]-1))
        actual_fluid_row = eff_top + fluid_row_rel

        usable_height = eff_bottom - eff_top
        fill_percent = 100.0 * ((eff_bottom - fluid_row_rel) / usable_height)
        fill_percent = float(np.clip(fill_percent, 0.0, 100.0))

        # Smooth
        self.level_buffer.append(fill_percent)
        fill_percent = float(np.mean(self.level_buffer))

        annotated = img.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 2)
        y_line = y1 + actual_fluid_row
        cv2.line(annotated, (x1, y_line), (x2, y_line), (0,255,0), 2)
        cv2.putText(annotated, f"{fill_percent:.1f}%", (x1+10, y_line-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y1 + eff_top), (60, 60, 60), -1)        # top cut
        cv2.rectangle(overlay, (x1, y1 + eff_bottom), (x2, y2), (60, 60, 60), -1)     # bottom cut
        annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

        cv2.putText(annotated, f"Top Cut: {self.top_cut_percent:.0f}%", (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        cv2.putText(annotated, f"Bottom Cut: {self.bottom_cut_percent:.0f}%", (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        tick_color = (0, 255, 255)
        for p in range(0, 101, 10):
            rel = 1 - (p / 100.0)
            y_tick = int(y1 + eff_top + rel * usable_height)
            cv2.line(annotated, (x2 + 5, y_tick), (x2 + 40, y_tick), tick_color, 2)
            cv2.putText(annotated, f"{p}%", (x2 + 45, y_tick + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, tick_color, 1, cv2.LINE_AA)

        return annotated, fill_percent        

    # -------------------------------------------------------------------
    async def trigger_alert(self, level, img, fill_percent):
        print(f"[DEBUG] trigger_alert called for {self.patient_data['patient_id']} level={level} fill={fill_percent:.1f}")
        alert_colors = {
            "low": (255, 255, 0),
            "medium": (0, 165, 255),
            "critical": (0, 0, 255)
        }
        color = alert_colors.get(level, (0, 255, 0))
        cv2.putText(img, f"{level.upper()} ALERT ({fill_percent:.1f}%)",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        path = None
        media_dir = os.path.join("media", self.camera_id, "snapshots")
        os.makedirs(media_dir, exist_ok=True)
        filename = f"{self.patient_data['patient_id']}_latest.png"
        path = os.path.abspath(os.path.join(media_dir, filename))
        try:
            cv2.imwrite(path, img)
            print(f"[DB] inserting to violations.db for {self.patient_data['patient_id']} ({fill_percent:.1f}%)")
        except Exception as e:
            print(f"[WARN] Failed to write snapshot for {self.camera_id}: {e}")
            path = None
        fill_percent_rounded = round(fill_percent, 2)

        thresholds = self.config["fluid_level_thresholds"]

        # Log only when exactly near the threshold values
        if (
            abs(fill_percent - thresholds["normal"]) < 0.5 and level == "normal"
        ) or (
            abs(fill_percent - thresholds["low"]) < 0.5 and level == "low"
        ) or (
            abs(fill_percent - thresholds["critical"]) < 0.5 and level == "critical"
        ):
            if not has_logged_threshold(self.patient_data["patient_id"], level):
                fill_percent_rounded = round(fill_percent, 2)
                insert_patient_history(
                    self.patient_data["patient_id"],
                    fill_percent_rounded,
                    level,
                    "2h"
                )
                print(f"[HISTORY] Logged {level.upper()} milestone for {self.patient_data['patient_id']}")

        #update_camera_status(self.camera_id, level)
        print(f"🚨 {level.upper()} ALERT | {self.camera_id} | Bed: {self.bed_no} | Level: {fill_percent:.1f}%")

    # -------------------------------------------------------------------
    async def process_video(self):
        print(f"\n🎥 Starting IV tracking for {self.camera_id} (Bed: {self.bed_no})...")
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video stream for {self.camera_id}")
            return

        last_update_time = 0     # timestamp for 1-second throttling
        last_fill = None         # last fluid percentage
        last_status = None       # last known status level
        last_screenshot_time = 0 # throttle screenshots too

        try:
            while True:
                ret, frame = cap.read()
                #print(f"[{self.camera_id}] frame read: {ret}")
                if not ret:
                    print(f"[WARN] Frame read failed for {self.camera_id}")
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    await asyncio.sleep(1)
                    continue

                # detect IV bag and fill level
                box, conf = self.detect_iv_bag_roi(frame)
                if box:
                    annotated, fill_percent = self.detect_fluid_level(frame, box)
                    if fill_percent >= 0:
                        if not hasattr(self, "level_history"):
                            self.level_history = []
                        self.level_history.append(fill_percent)
                        if len(self.level_history) > 5:
                            self.level_history.pop(0)
                        smooth_percent = sum(self.level_history) / len(self.level_history)
    
                        now = time.time()

                        # ✅ Only update once every second
                        if now - last_update_time >= 1:
                            # ✅ Skip duplicate readings unless they differ significantly
                            if last_fill is None or abs(fill_percent - last_fill) >= 0.5:
                                last_fill = fill_percent
                                last_update_time = now

                                thresholds = self.FLUID_THRESHOLDS
                                if fill_percent <= thresholds["critical"]:
                                    level = "critical"
                                elif fill_percent <= thresholds["low"]:
                                    level = "low"
                                else:
                                    level = "normal"

                                print(f"[{self.camera_id}] 💧 Fluid Level: {fill_percent:.1f}% | Status: {level}")

                                # ✅ Queue DB update
                                insert_glucose_status_async(
                                    patient_id=self.patient_data["patient_id"],
                                    patient_name=self.patient_data["patient_name"],
                                    bed_no=self.bed_no,
                                    ward=self.patient_data["ward"],
                                    doctor=self.patient_data["doctor"],
                                    fluid_type=self.patient_data["fluid_type"],
                                    start_time=datetime.now().isoformat(),
                                    flow_rate=self.patient_data["flow_rate"],
                                    prescribed_volume=self.patient_data["prescribed_volume"],
                                    remaining_percentage=fill_percent,
                                    fluid_level=fill_percent,
                                    time_left="2h",
                                    status=level,
                                    screenshot=None
                                )

                                # ✅ Trigger alert only when status changes
                                if level != self.last_alert:
                                    self.last_alert = level
                                    await self.trigger_alert(level, annotated, fill_percent)

                                
                # Optional live preview
                self.maybe_show_frame(f"IV Monitor - {self.camera_id}", frame)

                # small sleep for smoother async loop
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print(f"[INFO] KeyboardInterrupt received, stopping camera {self.camera_id}")

        except Exception as e:
            print(f"[ERROR] Unhandled exception in process_video for {self.camera_id}: {e}")

        finally:
            cap.release()
            if self._display_window_open:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            print(f"[INFO] Completed monitoring for {self.camera_id}")
            if callable(self.on_done_callback):
                try:
                    self.on_done_callback(self.camera_id)
                except Exception as e:
                    print(f"[WARN] on_done_callback failed for {self.camera_id}: {e}")
