"""
yolov8_pose_per_person_tracking.py

- YOLOv8n-pose -> per-person COM-based lightweight tracker
- Filters: bbox_conf >= 0.7, mean_kp_conf >= 0.7
- Per-person queues, velocity computed per-person using timestamps (no FPS dependence)
- Skeleton drawing uses Ultralytics `result.plot()` (style A)
- Video saved at same input resolution

Requirements:
- ultralytics (YOLOv8)
- opencv-python
- numpy
"""

import time
from collections import deque
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO
import itertools
import math

# -----------------------
# CONFIG (from your choices)
# -----------------------
MODEL_PATH = "yolov8n-pose.pt"
VIDEO_SOURCE = "test_videos/fall_test13.mp4"  
OUTPUT_FILENAME = "output_pose18.mp4"

# Filtering thresholds
BBOX_CONF_THRESH = 0.7          # bbox confidence filter (r.boxes.conf)
KP_CONF_THRESH = 0.7            # mean keypoint confidence filter

# Tracking / queue
MATCH_THRESHOLD_PX = 80         # match threshold in pixels (COM distance)
PERSON_QUEUE_MAXLEN = 300       # per-person queue length
PERSON_STALE_MS = 5000          # consider a person gone if not seen for 5s (cleanup)

# Video write
WRITE_FOURCC = "mp4v"
DEFAULT_OUT_FPS = 20.0

# -----------------------
# Utilities
# -----------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def mean_kp_conf(kps: np.ndarray) -> float:
    """
    kps: (K,3) -> x,y,conf
    Compute mean conf considering only non-zero keypoints (x,y != 0).
    """
    if kps is None or kps.size == 0:
        return 0.0
    nonzero_mask = ~((kps[:, 0] == 0) & (kps[:, 1] == 0))
    if not np.any(nonzero_mask):
        return 0.0
    return float(np.mean(kps[nonzero_mask, 2]))


def compute_com(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute COM as mean of left-shoulder(5), right-shoulder(6), left-hip(11), right-hip(12).
    Returns np.array([x,y]) or np.array([0.,0.]) if no valid points.
    """
    try:
        S_L, S_R, H_L, H_R = 5, 6, 11, 12
        pts = keypoints[[S_L, S_R, H_L, H_R], :2]  # (4,2)
        valid_mask = ~np.all(pts == 0, axis=1)
        if not np.any(valid_mask):
            return np.array([0.0, 0.0])
        return pts[valid_mask].mean(axis=0)
    except Exception:
        return np.array([0.0, 0.0])


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def find_best_match(com: np.ndarray, person_states: Dict[int, dict], threshold: float) -> Optional[int]:
    """
    Return the person_id with smallest COM distance < threshold, or None if no match.
    """
    best_id = None
    best_dist = float("inf")
    for pid, st in person_states.items():
        last_com = st.get("last_com")
        if last_com is None:
            continue
        d = euclidean(com, last_com)
        if d < best_dist:
            best_dist = d
            best_id = pid
    if best_id is None or best_dist > threshold:
        return None
    return best_id


def compute_velocity(com_current: np.ndarray, prev_entry: dict, t_current_ms: int) -> float:
    """
    prev_entry: {'ts_ms':int, 'com':np.array([...])}
    returns px/sec or 0.0 if not computable
    """
    if prev_entry is None:
        return 0.0
    dt_s = (t_current_ms - prev_entry["ts_ms"]) / 1000.0
    if dt_s <= 0:
        return 0.0
    dist = euclidean(com_current, prev_entry["com"])
    return float(dist / dt_s)

def draw_velocity_bbox(frame, xyxy, velocity, thresh=300):
    # velocity based color
    if velocity > thresh:
        color = (0,0,255)   # RED
    else:
        color = (0,255,0)   # GREEN

    x1,y1,x2,y2 = map(int, xyxy)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)


def angle_between(v1, v2):
    """Return angle in degrees between two vectors"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def classify_posture(kps):
    """
    kps: (17,3) numpy array x,y,conf
    Returns: "standing", "sitting", "lying", or "unknown"
    """
    try:
        # keypoints indices
        L_SH, R_SH = 5,6
        L_HIP, R_HIP = 11,12
        L_KNEE, R_KNEE = 13,14

        # extract points
        pts = {}
        for idx, name in zip([L_SH,R_SH,L_HIP,R_HIP,L_KNEE,R_KNEE],
                             ["L_SH","R_SH","L_HIP","R_HIP","L_KNEE","R_KNEE"]):
            pts[name] = kps[idx,:2]
            if np.all(pts[name]==0):
                pts[name] = None

        # vertical distances for lying/standing detection
        shoulder_y = np.mean([pts["L_SH"][1], pts["R_SH"][1]]) if pts["L_SH"] is not None and pts["R_SH"] is not None else None
        hip_y = np.mean([pts["L_HIP"][1], pts["R_HIP"][1]]) if pts["L_HIP"] is not None and pts["R_HIP"] is not None else None
        knee_y = np.mean([pts["L_KNEE"][1], pts["R_KNEE"][1]]) if pts["L_KNEE"] is not None and pts["R_KNEE"] is not None else None

        if None in [shoulder_y, hip_y, knee_y]:
            return "unknown"

        d_sh_hip = hip_y - shoulder_y
        d_hip_knee = knee_y - hip_y

        # lying detection
        if d_sh_hip < 15 and d_hip_knee < 15:
            return "lying"

        # compute hip angles left/right
        hip_angles = []
        for side in ["L","R"]:
            if pts[f"{side}_SH"] is None or pts[f"{side}_HIP"] is None or pts[f"{side}_KNEE"] is None:
                continue
            v1 = pts[f"{side}_SH"] - pts[f"{side}_HIP"]  # shoulder->hip
            v2 = pts[f"{side}_KNEE"] - pts[f"{side}_HIP"]  # knee->hip
            ang = angle_between(v1,v2)
            if ang is not None:
                hip_angles.append(ang)
        hip_angle = np.mean(hip_angles) if hip_angles else None

        # sitting detection via hip angle
        if hip_angle is not None and 70 <= hip_angle <= 120:
            return "sitting"

        # standing detection via vertical distances
        if d_sh_hip > 20 and d_hip_knee > 20:
            return "standing"

        return "unknown"

    except Exception:
        return "unknown"

# -----------------------
# Main
# -----------------------
def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {VIDEO_SOURCE}")

    # Input resolution & FPS for saving
    inp_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    inp_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    inp_fps = cap.get(cv2.CAP_PROP_FPS)
    if inp_fps is None or inp_fps <= 0 or np.isnan(inp_fps):
        inp_fps = DEFAULT_OUT_FPS

    if inp_w is None or inp_h is None:
        ret, tmp = cap.read()
        if not ret:
            raise RuntimeError("Failed to read a frame to determine resolution.")
        inp_h, inp_w = tmp.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*WRITE_FOURCC)
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, float(inp_fps), (inp_w, inp_h))

    print(f"[INFO] Input res: {inp_w}x{inp_h}  FPS(for saving): {inp_fps}")
    print(f"[INFO] Using BBOX_CONF_THRESH={BBOX_CONF_THRESH}, KP_CONF_THRESH={KP_CONF_THRESH}")
    print(f"[INFO] Match threshold: {MATCH_THRESHOLD_PX}px  id_type: incremental")

    # Person states: person_id -> { 'queue': deque, 'last_com': np.array, 'last_seen_ts': int }
    person_states: Dict[int, dict] = {}
    id_counter = itertools.count(1)  # incremental ids starting 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream / cannot fetch frame. Exiting loop.")
                break

            ts_ms = now_ms()

            results = model(frame, verbose=False)

            # annotate using ultralytics plot (skeletons)
            try:
                annotated = results[0].plot()
            except Exception:
                annotated = frame.copy()

            # results usually has single Results object per call; iterate for compatibility
            for r in results:
                if getattr(r, "keypoints", None) is None:
                    continue

                # r.boxes.conf -> shape (num_detections,)
                # r.keypoints.data -> tensor (num_detections, K, 3)
                boxes_conf = None
                try:
                    boxes_conf = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None else None
                except Exception:
                    boxes_conf = None

                try:
                    kp_array = r.keypoints.data.cpu().numpy()
                except Exception:
                    kp_array = np.asarray(r.keypoints.data)

                num_dets = kp_array.shape[0]
                # iterate over detection indices and apply bbox+kp filters
                for det_idx in range(num_dets):
                    # 1) bbox conf filter (if available)
                    if boxes_conf is None or len(boxes_conf) <= det_idx:
                        # No bounding box confidence for this detection → skip safely
                        continue
    
                    det_conf = float(boxes_conf[det_idx])
                    if det_conf < BBOX_CONF_THRESH:
                        continue
    
                    person_kps = kp_array[det_idx]  # (K,3) -> x,y,conf
                    # 2) keypoint mean conf filter
                    mkc = mean_kp_conf(person_kps)
                    if mkc < KP_CONF_THRESH:
                        continue

                    # 3) compute COM and skip if invalid
                    com = compute_com(person_kps)
                    if com[0] == 0.0 and com[1] == 0.0:
                        continue

                    # 4) match to existing person by COM distance
                    matched_id = find_best_match(com, person_states, MATCH_THRESHOLD_PX)
                    if matched_id is None:
                        # create new person id
                        new_id = next(id_counter)
                        person_states[new_id] = {
                            "queue": deque(maxlen=PERSON_QUEUE_MAXLEN),
                            "last_com": com,
                            "last_seen_ts": ts_ms
                        }
                        pid = new_id
                    else:
                        pid = matched_id
                        # update last seen/com will be done later

                    # 5) compute velocity for this person using its last entry
                    prev_entry = None
                    if person_states[pid]["queue"]:
                        prev_entry = person_states[pid]["queue"][-1]
                    vel = compute_velocity(com, prev_entry, ts_ms)

                    # compute downward displacement over last ~300ms window (not just 1 frame)
                    WINDOW_MS = 400
                    DISP_THRESH = 20 # pixels

                    # find oldest entry in last window
                    entries = person_states[pid]["queue"]
                    oldest = None
                    for e in reversed(entries):
                        if ts_ms - e["ts_ms"] > WINDOW_MS:
                            break
                        oldest = e

                    down_disp = 0.0
                    if oldest is not None:
                        down_disp = com[1] - oldest["com"][1]   # positive = going downward

                    # 6) append current detection to person's queue and update last_com/seen
                    person_states[pid]["queue"].append({
                        "ts_ms": ts_ms,
                        "com": com,
                        "kps": person_kps.copy()
                    })
                    person_states[pid]["last_com"] = com
                    person_states[pid]["last_seen_ts"] = ts_ms

                    # compute posture
                    posture_text = classify_posture(person_kps)

                    # override if velocity > 400 px/s
                    if vel > 400 and down_disp > DISP_THRESH:
                        posture_text = "falling"

                    # get bounding box coords
                    bbox = r.boxes.xyxy[det_idx].cpu().numpy()
                    x1,y1,x2,y2 = map(int, bbox)

                    # draw colored bbox (red if falling)
                    draw_velocity_bbox(annotated, bbox, vel, thresh=400)

                    # draw posture/falling text at top-left of bbox
                    cv2.putText(
                        annotated,
                        posture_text,
                        (x1, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),
                        2,
                        cv2.LINE_AA
                    )

                    # lightweight log
                    print(f"[{ts_ms}] PID={pid} COM=({com[0]:.1f},{com[1]:.1f}) vel={vel:.2f}px/s mkc={mkc:.2f}")

            # Cleanup stale persons not seen for PERSON_STALE_MS
            stale_ids = []
            nowt = ts_ms
            for pid, st in person_states.items():
                if nowt - st.get("last_seen_ts", 0) > PERSON_STALE_MS:
                    stale_ids.append(pid)
            for pid in stale_ids:
                print(f"[INFO] Removing stale person id {pid} (not seen for {PERSON_STALE_MS} ms)")
                person_states.pop(pid, None)

            # write annotated frame
            out.write(annotated)

            # # preview
            # cv2.imshow("Annotated", annotated)
            # if cv2.waitKey(1) == 27:
            #     print("[INFO] ESC pressed - exiting")
            #     break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Done.")


if __name__ == "__main__":
    main()
