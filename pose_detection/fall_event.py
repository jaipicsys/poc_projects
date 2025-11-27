# fall_event.py
import os
import time as pytime
import itertools
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import datetime
import cv2
from test import (
    letterbox_resize,
    DetectBox,
    NMS,
    process,
    CLASSES,
    kpt_color,
    skeleton,
    limb_color
)
from rknn.api import RKNN
from db import insert_patient_alert, update_camera_status, add_history
MODEL_PATH_DEFAULT = "fall.rknn" 
MATCH_THRESHOLD_PX = 80
PERSON_QUEUE_MAXLEN = 300
PERSON_STALE_MS = 5000
ALERT_COOLDOWN_MS_DEFAULT = 60000
WINDOW_MS_DEFAULT = 400
DISP_THRESH_DEFAULT = 15.0
VEL_THRESH_DEFAULT = 100.0
BBOX_CONF_THRESH_DEFAULT = 0.7
KP_CONF_THRESH_DEFAULT = 0.7
DEBUG_ANNOTATE = False
def draw_room_state(frame, state):
    """
    Draws the current room state text on the video frame.
    """
    color_map = {
        "stable": (0, 255, 0),     # green
        "caution": (0, 255, 255),      # yelcaution
        "emergency": (0, 0, 255),   # red
    }
    color = color_map.get(state, (255, 255, 255))
    label = f"STATE: {state}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x, text_y = 20, 50

    # Draw background box
    cv2.rectangle(
        frame,
        (text_x - 10, text_y - text_size[1] - 10),
        (text_x + text_size[0] + 10, text_y + 10),
        (0, 0, 0), -1
    )

    # Draw text
    cv2.putText(frame, label, (text_x, text_y),
                font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def now_ms() -> int:
    return int(pytime.time() * 1000)

def mean_kp_conf(kps: np.ndarray) -> float:
    """Mean keypoint confidence considering only non-zero x,y"""
    kps = np.asarray(kps)
    # reshape if needed
    if kps.shape[0] == 3 and kps.shape[1] != 3:
        kps = kps.T
    if kps.size == 0:
        return 0.0
    nonzero_mask = ~((kps[:, 0] == 0) & (kps[:, 1] == 0))
    if not np.any(nonzero_mask):
        return 0.0
    return float(np.mean(kps[nonzero_mask, 2]))

def compute_com(keypoints: np.ndarray) -> np.ndarray:
    """Compute COM as mean of left-shoulder(5), right-shoulder(6), left-hip(11), right-hip(12)."""
    try:
        S_L, S_R, H_L, H_R = 5, 6, 11, 12
        pts = keypoints[[S_L, S_R, H_L, H_R], :2]
        valid_mask = ~np.all(pts == 0, axis=1)
        if not np.any(valid_mask):
            return np.array([0.0, 0.0])
        return pts[valid_mask].mean(axis=0)
    except Exception:
        return np.array([0.0, 0.0])

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def angle_between(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def classify_posture(kps):
    """
    kps: (K,3) array
    Returns: "standing", "sitting", "lying", or "unknown"
    """
    try:
        L_SH, R_SH = 5,6
        L_HIP, R_HIP = 11,12
        L_KNEE, R_KNEE = 13,14

        pts = {}
        for idx, name in zip([L_SH,R_SH,L_HIP,R_HIP,L_KNEE,R_KNEE],
                             ["L_SH","R_SH","L_HIP","R_HIP","L_KNEE","R_KNEE"]):
            pts[name] = kps[idx,:2]
            if np.all(pts[name] == 0):
                pts[name] = None

        shoulder_y = np.mean([pts["L_SH"][1], pts["R_SH"][1]]) if pts["L_SH"] is not None and pts["R_SH"] is not None else None
        hip_y = np.mean([pts["L_HIP"][1], pts["R_HIP"][1]]) if pts["L_HIP"] is not None and pts["R_HIP"] is not None else None
        knee_y = np.mean([pts["L_KNEE"][1], pts["R_KNEE"][1]]) if pts["L_KNEE"] is not None and pts["R_KNEE"] is not None else None

        if None in [shoulder_y, hip_y, knee_y]:
            return "unknown"

        d_sh_hip = hip_y - shoulder_y
        d_hip_knee = knee_y - hip_y

        # lying detection (body is horizontal)
        if d_sh_hip < 15 and d_hip_knee < 15:
            return "lying"

        # hip angle
        hip_angles = []
        for side in ["L","R"]:
            if pts[f"{side}_SH"] is None or pts[f"{side}_HIP"] is None or pts[f"{side}_KNEE"] is None:
                continue
            v1 = pts[f"{side}_SH"] - pts[f"{side}_HIP"]
            v2 = pts[f"{side}_KNEE"] - pts[f"{side}_HIP"]
            ang = angle_between(v1, v2)
            if ang is not None:
                hip_angles.append(ang)
        hip_angle = np.mean(hip_angles) if hip_angles else None

        if hip_angle is not None and 70 <= hip_angle <= 120:
            return "sitting"
        if d_sh_hip > 20 and d_hip_knee > 20:
            return "standing"
        return "unknown"
    except Exception:
        return "unknown"

def draw_velocity_bbox(frame, xyxy, velocity, thresh=300):
    try:
        color = (0,0,255) if velocity > thresh else (0,255,0)
        x1,y1,x2,y2 = map(int, xyxy)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    except Exception:
        pass

def now_ms():
    import time
    return int(time.time() * 1000)

class RoomStateManager:
    def __init__(self, entry_delay_ms: int = 2000, emergency_delay_ms: int = 5000):
        self.state = "stable"
        self.pending_state: Optional[str] = None
        self.pending_since: Optional[int] = None
        self.entry_delay_ms = entry_delay_ms
        self.emergency_delay_ms = emergency_delay_ms

    def update(self, persons: List[Dict], patient_posture: str, falling_detected: bool) -> str:
        """
        Update the room state based on detected persons, posture, and fall events.
        
        :param persons: list of dicts, each with a 'pid' key
        :param patient_posture: 'lying', 'sitting', 'standing', etc.
        :param falling_detected: True if a fall was detected
        :return: current room state ('stable', 'caution', or 'emergency')
        """
        now = now_ms()
        current = self.state
        desired = current
        trigger = ""

        # Count unique PIDs in the current frame
        unique_pids = set(p["pid"] for p in persons)
        num_unique_pids = len(unique_pids)

        # ---------------------------------------------------------
        # RULE 0: emergency LOCK
        # If currently emergency, remain emergency until 2+ people present
        # ---------------------------------------------------------
        if current == "emergency" and num_unique_pids < 2:
            return "emergency"

        # ---------------------------------------------------------
        # RULE 1: FALL → emergency immediately
        # ---------------------------------------------------------
        if falling_detected:
            if current != "emergency":
                print(f"[INFO] STATE CHANGED: {current} → emergency (trigger=fall_detected)")
            self.state = "emergency"
            self.pending_state = None
            self.pending_since = None
            return "emergency"

        # ---------------------------------------------------------
        # RULE 2: Posture-based normal behavior (stable / caution)
        # ---------------------------------------------------------
        if patient_posture == "lying":
            desired = "stable"
            trigger = "lying"
        elif patient_posture in ("sitting", "standing"):
            if num_unique_pids >= 2:
                desired = "stable"
                trigger = "not_alone"
            else:
                desired = "caution"
                trigger = "alone"
        else:
            desired = "stable"
            trigger = "unknown"

        # ---------------------------------------------------------
        # Determine if a delay is required (debounce)
        # ---------------------------------------------------------
        if current == "emergency" and desired != "emergency":
            needs_delay = True
        else:
            needs_delay = (
                (current == "stable" and desired == "caution") or
                (current == "caution" and desired == "stable")
            )

        # No state change needed
        if desired == current:
            self.pending_state = None
            self.pending_since = None
            return current

        # ---------------------------------------------------------
        # Delayed transition (stable↔caution or emergency exit)
        # ---------------------------------------------------------
        if needs_delay:
            delay = self.emergency_delay_ms if current == "emergency" else self.entry_delay_ms

            if self.pending_state != desired:
                self.pending_state = desired
                self.pending_since = now
            else:
                if now - self.pending_since >= delay:
                    print(f"[INFO] STATE CHANGED: {current} → {desired} (trigger={trigger})")
                    self.state = desired
                    self.pending_state = None
                    self.pending_since = None

            return self.state

        # ---------------------------------------------------------
        # Immediate transition (everything else)
        # ---------------------------------------------------------
        print(f"[INFO] STATE CHANGED: {current} → {desired} (trigger={trigger})")
        self.state = desired
        self.pending_state = None
        self.pending_since = None
        return self.state

class FallDetector:
    def __init__(
        self,
        model_path: str = MODEL_PATH_DEFAULT,
        match_threshold_px: float = MATCH_THRESHOLD_PX,
        queue_maxlen: int = PERSON_QUEUE_MAXLEN,
        stale_ms: int = PERSON_STALE_MS,
        window_ms: int = WINDOW_MS_DEFAULT,
        disp_thresh: float = DISP_THRESH_DEFAULT,
        vel_thresh: float = VEL_THRESH_DEFAULT,
        bbox_conf_thresh: float = BBOX_CONF_THRESH_DEFAULT,
        kp_conf_thresh: float = KP_CONF_THRESH_DEFAULT,
        debug: bool = False,
        write_annotated=False
    ):
        # model loaded once per detector instance
        self.rknn = RKNN(verbose=True)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model at {model_path}")
        ret = self.rknn.init_runtime(target='rk3588')  # or 'rk3568', etc.
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

        self.match_threshold_px = float(match_threshold_px)
        self.queue_maxlen = int(queue_maxlen)
        self.stale_ms = int(stale_ms)
        self.window_ms = int(window_ms)
        self.disp_thresh = float(disp_thresh)
        self.vel_thresh = float(vel_thresh)
        self.bbox_conf_thresh = float(bbox_conf_thresh)
        self.kp_conf_thresh = float(kp_conf_thresh)
        self.debug = bool(debug)
        self.write_annotated = write_annotated 

        # person state dict
        self.person_states: Dict[int, dict] = {}
        self.id_counter = itertools.count(1)

    def find_best_match(self, com: np.ndarray) -> Optional[int]:
        best_id = None
        best_dist = float("inf")
        for pid, st in self.person_states.items():
            last_com = st.get("last_com")
            if last_com is None:
                continue
            d = euclidean(com, last_com)
            if d < best_dist:
                best_dist = d
                best_id = pid
        if best_id is None or best_dist > self.match_threshold_px:
            return None
        return best_id

    def compute_velocity(self, com_current: np.ndarray, prev_entry: Optional[dict], t_current_ms: int) -> float:
        if prev_entry is None:
            return 0.0
        dt_s = (t_current_ms - prev_entry["ts_ms"]) / 1000.0
        if dt_s <= 0:
            return 0.0
        dist = euclidean(com_current, prev_entry["com"])
        return float(dist / dt_s)

    def cleanup_stale(self, nowt_ms: int):
        stale_ids = []
        for pid, st in list(self.person_states.items()):
            if nowt_ms - st.get("last_seen_ts", 0) > self.stale_ms:
                stale_ids.append(pid)
        for pid in stale_ids:
            # remove stale
            self.person_states.pop(pid, None)
            if self.debug:
                print(f"[DEBUG] Removed stale pid={pid}")

    def process_frame(self, frame) -> Tuple[List[dict], dict, Optional[np.ndarray]]:
        ts_ms = now_ms()
        alerts: List[dict] = []
        summary = {"num_people": 0, "patient_posture": "unknown", "falling_detected": False}

        # Annotated frame (only created if debug True)
        annotated = frame.copy() if self.debug else None

        letterbox_img, scale, offset_x, offset_y = letterbox_resize(frame, (640, 640), 56)
        infer_img = letterbox_img[..., ::-1]  # BGR -> RGB

        try:
            results = self.rknn.inference(inputs=[infer_img])
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] RKNN inference failed: {e}")
            return alerts, summary, None

        outputs = []
        keypoints_raw = results[3]
        for x in results[:3]:
            index, stride = 0, 0
            if x.shape[2] == 20:
                stride, index = 32, 20*4*20*4 + 20*2*20*2
            elif x.shape[2] == 40:
                stride, index = 16, 20*4*20*4
            elif x.shape[2] == 80:
                stride, index = 8, 0
            feature = x.reshape(1, 65, -1)
            outputs.extend(process(feature, keypoints_raw, index, x.shape[3], x.shape[2], stride))

        predboxes = NMS(outputs)
        keypoints_list = [box.keypoint for box in predboxes]

        seen_person_ids = set()

        for det_idx, det_box in enumerate(predboxes):
            keypoints = keypoints_list[det_idx]

            keypoints = keypoints.reshape(-1, 3)

            keypoints[...,0] = (keypoints[...,0] - offset_x) / scale
            keypoints[...,1] = (keypoints[...,1] - offset_y) / scale
            det_box.xmin = (det_box.xmin - offset_x) / scale
            det_box.ymin = (det_box.ymin - offset_y) / scale
            det_box.xmax = (det_box.xmax - offset_x) / scale
            det_box.ymax = (det_box.ymax - offset_y) / scale
            score = getattr(det_box, 'score', 1.0)
            if score < self.bbox_conf_thresh:
                continue

            mkc = mean_kp_conf(keypoints)
            if mkc < self.kp_conf_thresh:
                continue

            com = compute_com(keypoints)
            if np.all(com == 0.0):
                continue
            matched_id = self.find_best_match(com)
            if matched_id is None:
                pid = next(self.id_counter)
                self.person_states[pid] = {
                    "queue": deque(maxlen=self.queue_maxlen),
                    "last_com": com,
                    "last_seen_ts": ts_ms,
                    "last_alert_ts": 0,
                    "last_posture": "unknown"
                }
            else:
                pid = matched_id

            prev_entry = self.person_states[pid]["queue"][-1] if self.person_states[pid]["queue"] else None
            vel = self.compute_velocity(com, prev_entry, ts_ms)

            queue = self.person_states[pid]["queue"]
            oldest = None
            for e in reversed(queue):
                if ts_ms - e["ts_ms"] > self.window_ms:
                    break
                oldest = e
            down_disp = com[1] - oldest["com"][1] if oldest else 0.0

            self.person_states[pid]["queue"].append({
                "ts_ms": ts_ms,
                "com": com,
                "kps": keypoints.copy()
            })
            self.person_states[pid]["last_com"] = com
            self.person_states[pid]["last_seen_ts"] = ts_ms

            posture = classify_posture(keypoints)

            is_fall = vel > self.vel_thresh and down_disp > self.disp_thresh
            if is_fall:
                posture = "falling"
                summary["falling_detected"] = True

            self.person_states[pid]["last_posture"] = posture

            if self.debug and annotated is not None:
                try:
                    draw_velocity_bbox(annotated, [det_box.xmin, det_box.ymin, det_box.xmax, det_box.ymax], vel, thresh=self.vel_thresh)

                    x1, y1, x2, y2 = map(int, [det_box.xmin, det_box.ymin, det_box.xmax, det_box.ymax])
                    cv2.putText(
                        annotated,
                        f"PID={pid} {posture}",
                        (x1, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    for kp_idx, kp in enumerate(keypoints):
                        x, y, conf = kp
                        if conf > 0.1:  # only draw confident keypoints
                            color = kpt_color[kp_idx % len(kpt_color)]
                            cv2.circle(annotated, (tuple(x), tuple(y)), 3, color, -1)

                    for j1, j2 in skeleton:
                        x1_s, y1_s, c1 = keypoints[j1]
                        x2_s, y2_s, c2 = keypoints[j2]
                        if c1 > 0.1 and c2 > 0.1:
                            color = limb_color.get((j1, j2), (0, 255, 255))
                            
                            cv2.line(annotated, (tuple(x1_s), tuple(y1_s)), (tuple(x2_s), tuple(y2_s)), color, 2)

                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Annotate failed: {e}")

            if self.debug:
                print(f"[DEBUG] PID={pid} vel={vel:.1f} down_disp={down_disp:.1f} is_fall={is_fall}")

                print(f"[DEBUG] [{ts_ms}] PID={pid} COM=({com[0]:.1f},{com[1]:.1f}) "
                    f"vel={vel:.2f}px/s down_disp={down_disp:.1f} posture={posture} mkc={mkc:.2f}")

            if is_fall:
                alerts.append({
                    "person_id": pid,
                    "event_type": "fall",
                    "ts_ms": ts_ms,
                    "posture": posture,
                    "extra_info": {"vel": vel, "down_disp": down_disp, "mkc": mkc}
                })
            else:
                seen_person_ids.add(pid)
            summary["patient_posture"] = posture
            summary["num_people"] += 1

        if len(seen_person_ids) == 1:
            only_pid = next(iter(seen_person_ids))
            last_posture = self.person_states[only_pid].get("last_posture", "unknown")
            if last_posture in ("sitting", "standing"):
                alerts.append({
                    "person_id": only_pid,
                    "event_type": "unattended",
                    "ts_ms": ts_ms,
                    "posture": last_posture,
                    "extra_info": {}
                })

        self.cleanup_stale(ts_ms)

        return alerts, summary, annotated if self.debug else None

class FallProcessor:
    def __init__(self, config: dict, camera_id: str, rtsp_url: str, on_done_callback=None, on_frame_callback=None):
        self.on_frame_callback=on_frame_callback
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        model_path = config.get("model_path", MODEL_PATH_DEFAULT)
        self.detector = FallDetector(
            model_path=model_path,
            match_threshold_px=config.get("match_threshold_px", MATCH_THRESHOLD_PX),
            queue_maxlen=config.get("queue_maxlen", PERSON_QUEUE_MAXLEN),
            stale_ms=config.get("stale_ms", PERSON_STALE_MS),
            window_ms=config.get("window_ms", WINDOW_MS_DEFAULT),
            disp_thresh=config.get("disp_thresh", DISP_THRESH_DEFAULT),
            vel_thresh=config.get("vel_thresh", VEL_THRESH_DEFAULT),
            bbox_conf_thresh=config.get("bbox_conf_thresh", BBOX_CONF_THRESH_DEFAULT),
            kp_conf_thresh=config.get("kp_conf_thresh", KP_CONF_THRESH_DEFAULT),
            debug=config.get("debug", False)
        )
        self.state_mgr = RoomStateManager(
            entry_delay_ms=config.get("entry_delay_ms", 3000),   # normal delay
            emergency_delay_ms=config.get("emergency_delay_ms", 5000)  # 5-sec delay for emergency exit
        )

        self.retry_sleep_s = int(config.get("retry_sleep_s", 120))
        self.alert_cooldown_ms = int(config.get("alert_cooldown_ms", ALERT_COOLDOWN_MS_DEFAULT))

        self.debug = bool(config.get("debug", False))
        self.write_annotated = bool(config.get("write_annotated", False))
        self.enable_display = bool(config.get("enable_display", False))
        self.annotated_out_path = config.get("annotated_out_path", None)
        self.annotated_writer = None

        print(f"[INFO] FallProcessor created for camera={self.camera_id} RTSP={self.rtsp_url}")

    def _should_log_alert(self, pid: int, event_ts_ms: int) -> bool:
        st = self.detector.person_states.get(pid, {})
        last_alert = st.get("last_alert_ts", 0)
        if event_ts_ms - last_alert < self.alert_cooldown_ms:
            return False
        # optimistic update to avoid duplicates if called concurrently
        st["last_alert_ts"] = event_ts_ms
        return True

    def _init_annotated_writer(self, frame_shape, fps):
        if not self.annotated_out_path:
            # default path
            self.annotated_out_path = f"annotated_{self.camera_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frame_shape[:2]
        self.annotated_writer = cv2.VideoWriter(self.annotated_out_path, fourcc, float(fps), (w, h))
        print(f"[INFO] Annotated output writer opened at {self.annotated_out_path}")

    def _save_frame(self, frame, media_dir):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = f"{media_dir}/alert_{ts}.jpg"
        cv2.imwrite(img_path, frame)
        return img_path

    def process_video(self):
        print(f"\n[INFO] Starting FallProcessor loop for camera {self.camera_id}...")
        media_dir = f"media/{self.camera_id}"
        os.makedirs(media_dir, exist_ok=True)   # auto-create folder per camera

        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[WARN] Camera {self.camera_id} unavailable. Retrying in {self.retry_sleep_s}s...")
                try:
                    update_camera_status(self.camera_id, "unavailable")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] update_camera_status error: {e}")
                pytime.sleep(self.retry_sleep_s)
                continue

            print(f"[INFO] Camera {self.camera_id} connected successfully.")
            try:
                update_camera_status(self.camera_id, "running")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] update_camera_status error: {e}")

            try:
                frame_idx = 0
                total_alerts = 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

                # initialize annotated writer lazily if enabled
                if self.write_annotated:
                    # attempt to read first frame for shape if needed
                    ret_tmp, frame_tmp = cap.read()
                    if ret_tmp:
                        self._init_annotated_writer(frame_tmp.shape, fps)
                        # reset to beginning if needed (for file) - but for RTSP we continue
                        # NOTE: for RTSP, we've consumed one frame; that's okay
                    else:
                        # couldn't read, will rely on first processed frame to init writer
                        pass

                # process loop
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[WARN] Failed to read frame from {self.camera_id}. Video ended or cannot read.")
                        break

                    if frame is None:
                        continue

                    frame_idx += 1

                    # get alerts and summary from detector
                    alerts, summary, annotated = self.detector.process_frame(frame)
                    tracked_persons = [{"pid": pid} for pid in self.detector.person_states.keys()] if self.detector.person_states else []

                    current_state = self.state_mgr.update(
                        persons=tracked_persons,
                        patient_posture=summary.get("patient_posture", "unknown"),
                        falling_detected=summary.get("falling_detected", False)
                    )

                    # Count number of active tracked people
                    num_people = len(self.detector.person_states)

                    # Check if any falls detected
                    falling_detected = any(a["event_type"] == "fall" for a in alerts)

                    # Get primary patient's posture if available
                    active_pids = list(self.detector.person_states.keys())
                    if active_pids:
                        pid = active_pids[0]
                        patient_posture = self.detector.person_states[pid].get("last_posture", "unknown")
                    else:
                        patient_posture = "unknown"

                    tracked_persons = [{"pid": pid} for pid in self.detector.person_states.keys()]

                    new_state = self.state_mgr.update(
                        persons=tracked_persons,
                        patient_posture=summary.get("patient_posture", "unknown"),
                        falling_detected=summary.get("falling_detected", False)
                    )

                    # ----------------------------
                    # Log State Change
                    # ----------------------------
                    prev_state = getattr(self, "prev_state", None)

                    if prev_state is None:
                        # first frame — no previous state to compare
                        self.prev_state = new_state
                    else:
                        if prev_state != new_state:
                            try:
                                image_path = self._save_frame(frame, media_dir)
                                insert_patient_alert(
                                    person_id=0,
                                    event_type="state_change",
                                    alert_type=new_state.lower(),
                                    cam_id=self.camera_id,
                                    image_path=image_path
                                )
                                print(f"[DB] State changed: {prev_state} → {new_state}")
                            except Exception as e:
                                print(f"[DB ERROR] Failed to log state change: {e}")

                        # update stored state
                        self.prev_state = new_state

                    frame = draw_room_state(frame, new_state)

                    # Log and write alerts to DB with cooldown
                    for alert in alerts:
                        pid = alert["person_id"]
                        etype = alert["event_type"]
                        ts_ms = alert["ts_ms"]

                        if not self._should_log_alert(pid, ts_ms):
                            # skip duplicate alert due to cooldown
                            if self.debug:
                                print(f"[DEBUG] Skipping duplicate alert pid={pid} type={etype}")
                            continue

                        # Insert into DB
                        try:
                            # adopt signature from your db helpers
                            alert_type = new_state.lower()  # normal / low / emergency

                            image_path = self._save_frame(frame, media_dir)
                            insert_patient_alert(pid, etype, alert_type, self.camera_id, image_path)

                            # optionally add_history entry if desired
                            try:
                                add_history({
                                    "timestamp_iso": datetime.datetime.utcfromtimestamp(ts_ms/1000.0).isoformat(),
                                    "camera_id": self.camera_id,
                                    "person_id": pid,
                                    "event_type": etype,
                                    "posture": alert.get("posture"),
                                    "extra_info": alert.get("extra_info", {})
                                })
                            except Exception:
                                # add_history optional; ignore failures
                                pass

                            total_alerts += 1
                            print(f"[ALERT] Camera={self.camera_id} PID={pid} TYPE={etype} at {datetime.datetime.now().isoformat()}")
                        except Exception as e:
                            print(f"[DB ERROR] Failed to insert patient alert: {e}")

                    # optionally write annotated frames
                    if self.write_annotated:
                        # annotated may be None if debug disabled inside detector; fallback to original frame
                        frame_to_write = annotated if annotated is not None else frame
                        if self.annotated_writer is None:
                            # lazy init with frame shape
                            self._init_annotated_writer(frame_to_write.shape, fps)
                        try:
                            self.annotated_writer.write(frame_to_write)
                        except Exception as e:
                            if self.debug:
                                print(f"[DEBUG] Failed to write annotated frame: {e}")
                    
                    
                    # Optional live frame feed
                    if self.on_frame_callback is not None:
                        try:
                            self.on_frame_callback(self.camera_id, frame)
                        except Exception as e:
                            print(f"[WARN] on_frame callback error: {e}")
                    # ------------------------------------
                    # Optional Local Display
                    # ------------------------------------
                    if self.enable_display:
                        cv2.imshow(f"Camera - {self.camera_id}", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print(f"[INFO] Display window closed for {self.camera_id}")
                            cv2.destroyAllWindows()
                            # If you want to stop only display, do nothing else
                            # If you want to stop camera processing also, uncomment next line:
                            # break


                    # Optionally update camera heartbeat / status (already set to running)
                    # Could also send socketio events here (not implemented)

                # end while cap.isOpened()

            finally:
                cap.release()
                if self.annotated_writer is not None:
                    try:
                        self.annotated_writer.release()
                    except Exception:
                        pass
                    self.annotated_writer = None

                print(f"\n[INFO] Processing completed for camera {self.camera_id}")
                print(f"[STATS] Total frames processed: {frame_idx}")
                print(f"[STATS] Total alerts: {total_alerts}")

                try:
                    update_camera_status(self.camera_id, "stopped")
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] update_camera_status error: {e}")
                if self.enable_display:
                    try:
                        cv2.destroyAllWindows()
                    except:
                        pass
