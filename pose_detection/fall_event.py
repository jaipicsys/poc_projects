# fall_event.py
"""
Integrated fall detection + room state manager

- Per-person COM-based tracker + velocity+displacement fall detection (from script1)
- Room state manager (NORMAL / LOW / CRITICAL) + RTSP loop (from script2)
- Integrates with DB helpers: insert_patient_alert, update_camera_status, add_history
- Designed as a drop-in replacement for your existing fall_event.py structure
"""
import os
import time as pytime
import itertools
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import datetime
import cv2

# Model import (RKNN)
from rknn.api import RKNN

# DB helpers
from db import insert_patient_alert, update_camera_status, add_history

# -----------------------
# Configurable defaults
# -----------------------
MODEL_PATH_DEFAULT = "fall.rknn"      # ← change your model path here
RKNN_INPUT_SIZE = 256                 # ← adjust based on your model

MATCH_THRESHOLD_PX = 80
PERSON_QUEUE_MAXLEN = 300
PERSON_STALE_MS = 5000
ALERT_COOLDOWN_MS_DEFAULT = 60000
WINDOW_MS_DEFAULT = 400
DISP_THRESH_DEFAULT = 20.0
VEL_THRESH_DEFAULT = 400.0
BBOX_CONF_THRESH_DEFAULT = 0.7
KP_CONF_THRESH_DEFAULT = 0.7

CLASSES = ["person"]  # or other relevant classes
objectThresh = 0.3    # detection threshold
nmsThresh = 0.45      # NMS IoU threshold


DEBUG_ANNOTATE = False


# -----------------------
# RKNN preprocess / postprocess
# -----------------------
def preprocess_rknn(frame):
    img = cv2.resize(frame, (RKNN_INPUT_SIZE, RKNN_INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def decode_rknn_outputs(outputs):
    """
    Convert RKNN pose outputs into DetectBox instances and keypoints.
    Assumes outputs[0..2] are feature maps and outputs[3] is keypoints.
    """
    boxes = []
    keypoints_list = []

    keypoints_raw = outputs[3]

    for x in outputs[:3]:
        index, stride = 0, 0
        if x.shape[2] == 20:
            stride = 32
            index = 20*4*20*4 + 20*2*20*2
        elif x.shape[2] == 40:
            stride = 16
            index = 20*4*20*4
        elif x.shape[2] == 80:
            stride = 8
            index = 0

        feature = x.reshape(1, 65, -1)
        dets = process(feature, keypoints_raw, index, x.shape[3], x.shape[2], stride)
        boxes.extend(dets)
        for det in dets:
            keypoints_list.append(det.keypoint.reshape(-1,3))

    return boxes, keypoints_list

def draw_room_state(frame, state):
    """
    Draws the current room state text on the video frame.
    """
    color_map = {
        "NORMAL": (0, 255, 0),     # green
        "LOW": (0, 255, 255),      # yellow
        "CRITICAL": (0, 0, 255),   # red
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

# -----------------------
# Utilities
# -----------------------
def now_ms() -> int:
    return int(pytime.time() * 1000)

def mean_kp_conf(kps: np.ndarray) -> float:
    """Mean keypoint confidence considering only non-zero x,y"""
    if kps is None or kps.size == 0:
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

# -----------------------
# RoomStateManager
# -----------------------
class RoomStateManager:
    def __init__(self, entry_delay_ms: int = 2000):
        self.state = "NORMAL"
        self.pending_state: Optional[str] = None
        self.pending_since: Optional[int] = None
        self.entry_delay_ms = entry_delay_ms

    def update(self, num_people: int, patient_posture: str, falling_detected: bool) -> str:
        now = now_ms()
        desired = self.state
        trigger = ""

        # Rule 1: fall → CRITICAL immediately
        if falling_detected:
            desired = "CRITICAL"
            trigger = "fall_detected"
        # Rule 2: lying → NORMAL
        elif patient_posture == "lying":
            desired = "NORMAL"
            trigger = "lying"
        # Rule 3: sitting/standing => possible LOW if alone
        elif patient_posture in ("sitting", "standing"):
            if num_people == 1:
                desired = "LOW"
                trigger = "alone"
            else:
                desired = "NORMAL"
                trigger = "not_alone"
        else:
            desired = "NORMAL"
            trigger = "unknown_or_none"

        # CRITICAL persists until cleared by logic or restart
        if self.state == "CRITICAL":
            return self.state

        # Debounce only NORMAL -> LOW
        if desired != self.state:
            if desired == "LOW" and self.state == "NORMAL":
                if self.pending_state != "LOW":
                    self.pending_state = "LOW"
                    self.pending_since = now
                elif now - (self.pending_since or now) >= self.entry_delay_ms:
                    # commit
                    print(f"[INFO] STATE CHANGED: {self.state} → LOW  (trigger={trigger})")
                    self.state = "LOW"
                    self.pending_state = None
                    self.pending_since = None
            else:
                # immediate switch for all other transitions
                print(f"[INFO] STATE CHANGED: {self.state} → {desired}  (trigger={trigger})")
                self.state = desired
                self.pending_state = None
                self.pending_since = None
        else:
            # keep stable
            self.pending_state = None
            self.pending_since = None

        return self.state
#----------------------------------
# DECODE LOGIC
#---------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    # 将输入向量减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint

def IOU(x1, y1, X1, Y1, x2, y2, X2, Y2):
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(X1, X2)
    yi2 = min(Y1, Y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (X1 - x1) * (Y1 - y1)
    box2_area = (X2 - x2) * (Y2 - y2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def process(out,keypoints,index,model_w,model_h,stride,scale_w=1,scale_h=1):
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0,c,(h*model_w)+w]>objectThresh:
                    xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                    xywh_=xywh_.reshape(1,4,16,1)
                    data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                    xywh_=softmax(xywh_,2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp=xywh_.copy()
                    xywh_temp[0]=(w+0.5)-xywh_[0]
                    xywh_temp[1]=(h+0.5)-xywh_[1]
                    xywh_temp[2]=(w+0.5)+xywh_[2]
                    xywh_temp[3]=(h+0.5)+xywh_[3]

                    xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                    xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                    xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                    xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                    xywh_=xywh_*stride

                    xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint=keypoints[...,(h*model_w)+w+index] 
                    keypoint[...,0:2]=keypoint[...,0:2]//1
                    box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                    out.append(box)

    return out

def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

# -----------------------
# FallDetector (pure logic, no IO)
# -----------------------
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
        debug: bool = False
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
        """
        Run RKNN model on the frame and update internal person_states.
        Returns:
            alerts: list of {'person_id': int, 'event_type': 'fall'|'unattended', 'ts_ms': int, 'posture': str}
            summary: {'num_people': int, 'patient_posture': str, 'falling_detected': bool}
            annotated_frame: np.ndarray (if debug True) else None
        """
        ts_ms = now_ms()
        alerts: List[dict] = []
        summary = {"num_people": 0, "patient_posture": "unknown", "falling_detected": False}

        # Annotated frame if debug
        annotated = frame.copy() if self.debug else None

        # RKNN inference
        letterbox_img, aspect_ratio, offset_x, offset_y = letterbox_resize(frame, (640, 640), 56)
        infer_img = letterbox_img[..., ::-1]  # BGR → RGB

        try:
            results = self.rknn.inference(inputs=[infer_img])
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] RKNN inference failed: {e}")
            return alerts, summary, None

        # decode RKNN outputs
        predboxes, keypoints_list = decode_rknn_outputs(results)

        seen_person_ids = set()

        for det in predboxes:
            # bbox confidence filter
            if det.score < self.bbox_conf_thresh:
                continue

            keypoints = det.keypoint.reshape(-1, 3)
            kp_conf = mean_kp_conf(keypoints)
            if kp_conf < self.kp_conf_thresh:
                continue

            # compute COM
            com = compute_com(keypoints)
            if np.all(com == 0):
                continue

            # match person or assign new ID
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

            # compute velocity
            prev_entry = self.person_states[pid]["queue"][-1] if self.person_states[pid]["queue"] else None
            vel = self.compute_velocity(com, prev_entry, ts_ms)

            # compute downward displacement over window
            down_disp = 0.0
            for e in reversed(self.person_states[pid]["queue"]):
                if ts_ms - e["ts_ms"] > self.window_ms:
                    break
                down_disp = com[1] - e["com"][1]

            # update queue
            self.person_states[pid]["queue"].append({
                "ts_ms": ts_ms,
                "com": com,
                "kps": keypoints.copy()
            })
            self.person_states[pid]["last_com"] = com
            self.person_states[pid]["last_seen_ts"] = ts_ms

            # classify posture
            posture = classify_posture(keypoints)

            # fall detection: both velocity + downward displacement
            is_fall = False
            if vel > self.vel_thresh and down_disp > self.disp_thresh:
                is_fall = True
                posture = "falling"
                summary["falling_detected"] = True

            self.person_states[pid]["last_posture"] = posture
            seen_person_ids.add(pid)

            # Debug draw
            if self.debug and annotated is not None:
                bbox = np.array([
                    (det.xmin - offset_x)/aspect_ratio,
                    (det.ymin - offset_y)/aspect_ratio,
                    (det.xmax - offset_x)/aspect_ratio,
                    (det.ymax - offset_y)/aspect_ratio
                ])
                draw_velocity_bbox(annotated, bbox, vel, thresh=self.vel_thresh)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(
                    annotated,
                    f"PID={pid} {posture}",
                    (x1, max(10, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Append fall alert if detected
            if is_fall:
                alerts.append({
                    "person_id": pid,
                    "event_type": "fall",
                    "ts_ms": ts_ms,
                    "posture": posture,
                    "extra_info": {"vel": vel, "down_disp": down_disp, "mkc": kp_conf}
                })

        # unattended alert if single person sitting/standing
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

        # cleanup stale persons
        self.cleanup_stale(ts_ms)

        # summary
        if self.person_states:
            first_pid = next(iter(self.person_states))
            summary["patient_posture"] = self.person_states[first_pid].get("last_posture", "unknown")
            summary["num_people"] = len(self.person_states)

        return alerts, summary, annotated if self.debug else None


# -----------------------
# FallProcessor (per-camera processing, RTSP loop)
# -----------------------
class FallProcessor:
    def __init__(self, config: dict, camera_id: str, rtsp_url: str, on_done_callback=None, on_frame_callback=None):
        self.on_frame_callback=on_frame_callback
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.room_state_mgr = RoomStateManager(entry_delay_ms=config.get("entry_delay_ms", 1000))

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
        # Room state manager — debounced LOW entry
        self.state_mgr = RoomStateManager(entry_delay_ms=config.get("low_entry_delay_ms", 2000))

        self.retry_sleep_s = int(config.get("retry_sleep_s", 120))
        self.alert_cooldown_ms = int(config.get("alert_cooldown_ms", ALERT_COOLDOWN_MS_DEFAULT))

        self.debug = bool(config.get("debug", False))
        self.write_annotated = bool(config.get("write_annotated", False))
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

                    # update room state using summary
                    current_state = self.state_mgr.update(summary.get("num_people", 0),
                                                         summary.get("patient_posture", "unknown"),
                                                         summary.get("falling_detected", False))

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

                    # Update room state
                    new_state = self.room_state_mgr.update(num_people, patient_posture, falling_detected)
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
                                    person_id=pid,
                                    event_type=etype,
                                    alert_type=new_state.lower(),  # normal / low / critical
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
                            alert_type = new_state.lower()  # normal / low / critical

                            image_path = self._save_frame(frame, media_dir)
                            insert_patient_alert(pid, etype, alert_type, self.camera_id, image_path)

                            # optionally add_history entry if desired
                            try:
                                add_history({
                                    "timestamp_iso": datetime.datetime.utcnow().isoformat(),
                                    "camera_id": self.camera_id,
                                    "person_id": pid,
                                    "event_type": etype,
                                    "posture": posture,
                                    "extra_info": {}
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

                # # optional callback
                # if self.on_done_callback:
                #     try:
                #         self.on_done_callback(self.camera_id)
                #     except Exception:
                #         pass

                # print(f"[INFO] Retrying camera {self.camera_id} in {self.retry_sleep_s}s...\n")
                # pytime.sleep(self.retry_sleep_s)

# -----------------------
# If run as standalone for testing
# -----------------------
# if __name__ == "__main__":
#     # simple local test using a file or RTSP (adjust path)
#     TEST_RTSP = "pose_integration/test_videos/fall_test6.mp4"  # change to rtsp://... for live
#     cfg = {
#         "model_path": MODEL_PATH_DEFAULT,
#         "debug": True,
#         "write_annotated": True,
#         "annotated_out_path": f"annotated_test_{int(pytime.time())}.mp4",
#         "low_entry_delay_ms": 2000,
#         "retry_sleep_s": 10,
#         "alert_cooldown_ms": ALERT_COOLDOWN_MS_DEFAULT
#     }
#     proc = FallProcessor(cfg, camera_id="testcam", rtsp_url=TEST_RTSP)
#     proc.process_video()