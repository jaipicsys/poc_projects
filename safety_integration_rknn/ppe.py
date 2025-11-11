# ppe.py
"""
PPE Detection + SORT Tracking + RKNN Inference
Uses RKNN DFL-head decode (ported from the provided C++ sample).
Input size: 512
"""
import os
import cv2
import time
import json
import math
import numpy as np
from datetime import datetime
from sort import Sort
from rknnlite.api import RKNNLite
from db import insert_violation_async, update_camera_status

# ---------------- Configurable constants ----------------
INPUT_SIZE = 512
CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
TRACKER_PARAMS = dict(max_age=50, min_hits=3, iou_threshold=0.3)
DEFAULT_VIOLATION_CLASSES = [1, 3]  # numeric ids, overwritten by config's violation_classes (names)

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

# ---------------- DFL helpers ----------------
def compute_dfl(tensor, dfl_len):
    out = np.zeros(4, dtype=np.float32)
    for b in range(4):
        start = b * dfl_len
        vals = tensor[start:start + dfl_len].astype(np.float64)
        exps = np.exp(vals - np.max(vals))
        s = exps.sum()
        if s == 0:
            s = 1e-6
        probs = exps / s
        acc = (probs * np.arange(dfl_len)).sum()
        out[b] = float(acc)
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

# ---------------- RKNN attribute helper ----------------
def get_attr_info(rknn, idx):
    """
    Query RKNNLite for tensor info and return permissive dict.
    RKNNLite.query(index) returns a dict-like or object in many wrappers.
    """
    try:
        info = rknn.query(idx)
        if isinstance(info, dict):
            return {
                'dims': info.get('dims', None),
                'zp': info.get('zp', 0),
                'scale': info.get('scale', 1.0),
                'is_signed': info.get('is_signed', True)
            }
        else:
            return {
                'dims': getattr(info, 'dims', None),
                'zp': getattr(info, 'zp', 0),
                'scale': getattr(info, 'scale', 1.0),
                'is_signed': getattr(info, 'is_signed', True)
            }
    except Exception:
        return {'dims': None, 'zp': 0, 'scale': 1.0, 'is_signed': True}

def read_out_value(out_arr, attr, ch_index, grid_idx, grid_len, raw_quant, quant_signed):
    """
    Read a value for channel ch_index at linear grid index grid_idx.
    Handles float outputs or raw quantized int8/uint8 + (zp, scale) from attrs.
    out_arr is a numpy array (as returned by RKNNLite inference).
    """
    a = out_arr
    if raw_quant and a.dtype in (np.int8, np.uint8):
        flat = a.flatten()
        idx = ch_index * grid_len + grid_idx
        val_q = int(flat[idx])
        return (float(val_q) - float(attr.get('zp', 0))) * float(attr.get('scale', 1.0))
    else:
        # floats or already dequantized: assume channel-first (C,H,W) or (1,C,H,W)
        if a.ndim == 4 and a.shape[0] == 1:
            a2 = a[0]
        else:
            a2 = a
        if a2.ndim == 3:
            C, H, W = a2.shape
            y = grid_idx // W
            x = grid_idx % W
            return float(a2[ch_index, y, x])
        elif a2.ndim == 2:
            return float(a2[ch_index, grid_idx])
        elif a2.ndim == 1:
            idx = ch_index * grid_len + grid_idx
            return float(a2[idx])
        else:
            flat = a2.flatten()
            idx = ch_index * grid_len + grid_idx
            return float(flat[idx])

def decode_dfl_heads_python(outputs, attrs, orig_w, orig_h, input_w, input_h,
                            raw_quant, quant_signed, conf_thres=CONF_THRESH):
    """
    Decode RKNN outputs arranged as repeating groups of (box_feat, cls_map, conf_map).
    Returns list of dict boxes: {x1,y1,x2,y2,conf,class_id} in INPUT coords (not yet un-letterboxed).
    """
    boxes = []
    # outputs assumed list/tuple of numpy arrays in same order as RKNN returns
    for i in range(0, len(outputs), 3):
        if i + 2 >= len(outputs):
            break
        o_box = outputs[i]
        o_cls = outputs[i + 1]
        o_conf = outputs[i + 2]
        a_box = attrs[i]
        a_cls = attrs[i + 1]
        a_conf = attrs[i + 2]

        # infer shapes (attempt CHW)
        def infer_chw(arr):
            if arr is None:
                return None, None, None
            if isinstance(arr, dict) and 'dims' in arr:
                # attrs-style
                dims = arr['dims']
                if dims and len(dims) >= 3:
                    # assume (1,C,H,W) or (C,H,W)
                    if len(dims) == 4 and dims[0] == 1:
                        return dims[1], dims[2], dims[3]
                    if len(dims) == 3:
                        return dims[0], dims[1], dims[2]
            # array-based
            if arr is None:
                return None, None, None
            if hasattr(arr, 'ndim'):
                if arr.ndim == 4 and arr.shape[0] == 1:
                    return arr.shape[1], arr.shape[2], arr.shape[3]
                if arr.ndim == 3:
                    return arr.shape[0], arr.shape[1], arr.shape[2]
                if arr.ndim == 2:
                    return arr.shape[0], 1, arr.shape[1]
            return None, None, None

        cls_C, H, W = infer_chw(o_cls)
        if cls_C is None or H is None or W is None:
            # try reading dims from attrs as fallback
            dims = a_cls.get('dims') if isinstance(a_cls, dict) else getattr(a_cls, 'dims', None)
            if dims and len(dims) >= 4:
                cls_C = dims[1]; H = dims[2]; W = dims[3]
            else:
                continue
        grid_len = H * W

        box_ch, _, _ = infer_chw(o_box)
        if box_ch is None:
            dims = a_box.get('dims') if isinstance(a_box, dict) else getattr(a_box, 'dims', None)
            if dims and len(dims) >= 2:
                box_ch = dims[1]
            else:
                continue
        dfl_len = max(1, box_ch // 4)
        stride = int(input_h // H)

        # prepare arrays for reading if they're RKNNLite outputs (numpy arrays)
        o_box_arr = o_box
        o_cls_arr = o_cls
        o_conf_arr = o_conf

        for y in range(H):
            for x in range(W):
                gidx = y * W + x
                obj_conf = read_out_value(o_conf_arr, a_conf, 0, gidx, grid_len, raw_quant, quant_signed)
                if obj_conf < conf_thres:
                    continue

                best_cls = 0
                best_score = 0.0
                for c in range(cls_C):
                    cls_score = read_out_value(o_cls_arr, a_cls, c, gidx, grid_len, raw_quant, quant_signed)
                    if cls_score > best_score:
                        best_score = cls_score
                        best_cls = c

                final_conf = float(obj_conf * best_score)
                if final_conf < conf_thres:
                    continue

                # read 4*dfl_len channel values for this cell
                before = np.zeros(4 * dfl_len, dtype=np.float32)
                for k in range(4 * dfl_len):
                    before[k] = read_out_value(o_box_arr, a_box, k, gidx, grid_len, raw_quant, quant_signed)

                box_dec = compute_dfl(before, dfl_len)

                # Convert to input coords
                x1 = (-box_dec[0] + x + 0.5) * stride
                y1 = (-box_dec[1] + y + 0.5) * stride
                x2 = ( box_dec[2] + x + 0.5) * stride
                y2 = ( box_dec[3] + y + 0.5) * stride

                boxes.append({
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'conf': float(final_conf),
                    'class_id': int(best_cls)
                })
    return boxes

def convert_boxes_from_input_to_original(boxes, input_w, input_h, orig_w, orig_h, scale, pad_x, pad_y):
    for b in boxes:
        x1 = (b['x1'] - pad_x) / scale
        y1 = (b['y1'] - pad_y) / scale
        x2 = (b['x2'] - pad_x) / scale
        y2 = (b['y2'] - pad_y) / scale
        b['x1'] = max(0, min(x1, orig_w - 1))
        b['y1'] = max(0, min(y1, orig_h - 1))
        b['x2'] = max(0, min(x2, orig_w - 1))
        b['y2'] = max(0, min(y2, orig_h - 1))

# ---------------- PPE Processor ----------------
class PPEProcessor:
    def __init__(self, config: dict, camera_id: str, rtsp_url: str, on_done_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.config = config or {}
        self.enable_display = self.config.get("enable_display", False)
        self.violation_classes_names = self.config.get("violation_classes", ["no_hardhat", "no_vest"])

        # Prepare output dirs
        output_dir = self.config.get("output_dir", "output")
        camera_dir = os.path.join(output_dir, camera_id)
        snapshot_dir = os.path.join(camera_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshot_dir = snapshot_dir

        # RKNN model (read from config.model_path)
        model_path = self.config.get("ppe_model_path")
        if not model_path:
            raise ValueError("[CONFIG ERROR] Missing 'model_path' in config.json for PPE model")
        self.rknn = RKNNLite()
        print(f"[INFO] Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")

        # tracker
        self.tracker = Sort(**TRACKER_PARAMS)

        # class names (string names)
        self.class_names = self.config.get("class_names", ["hardhat", "no_hardhat", "vest", "no_vest"])

        # counters & diagnostics
        self.missing_streams = 0
        self.missing_reasons = {'open_failed': 0, 'frame_read_failed': 0, 'inference_error': 0}
        self.inference_times_ms = []

        # local constants
        self.input_size = INPUT_SIZE

    def process_video(self):
        print(f"\n[INFO] Starting PPE processing for camera {self.camera_id}...")
        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[WARN] Camera {self.camera_id} unavailable (cannot open).")
                self.missing_streams += 1
                self.missing_reasons['open_failed'] += 1
                update_camera_status(self.camera_id, "unavailable")
                time.sleep(120)
                continue

            print(f"[INFO] Camera {self.camera_id} connected successfully.")
            update_camera_status(self.camera_id, "running")

            try:
                frame_idx = 0
                total_violations = 0
                seen_violations = set()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"[WARN] Failed to read frame from {self.camera_id}")
                        self.missing_streams += 1
                        self.missing_reasons['frame_read_failed'] += 1
                        break

                    orig_h, orig_w = frame.shape[:2]

                    # preprocess (letterbox)
                    padded, scale, pad_x, pad_y = letterbox_image(frame, INPUT_SIZE, INPUT_SIZE)
                    img_input = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                    img_input = np.expand_dims(img_input, 0).astype(np.uint8)

                    # inference + timing
                    try:
                        t0 = time.time()
                        outputs = self.rknn.inference(inputs=[img_input])
                        inf_ms = (time.time() - t0) * 1000.0
                        self.inference_times_ms.append(inf_ms)
                    except Exception as e:
                        print(f"[ERROR] Inference error for {self.camera_id}: {e}")
                        self.missing_reasons['inference_error'] += 1
                        self.missing_streams += 1
                        break

                    # build attrs & detect raw-quant
                    attrs = [get_attr_info(self.rknn, i) for i in range(len(outputs))]
                    raw_quant = any([o.dtype in (np.int8, np.uint8) for o in outputs])
                    quant_signed = True
                    if attrs and attrs[0]:
                        quant_signed = bool(attrs[0].get('is_signed', True))

                    # decode heads
                    boxes_in_input = decode_dfl_heads_python(
                        outputs, attrs,
                        orig_w=orig_w, orig_h=orig_h,
                        input_w=INPUT_SIZE, input_h=INPUT_SIZE,
                        raw_quant=raw_quant, quant_signed=quant_signed,
                        conf_thres=CONF_THRESH
                    )

                    # convert boxes to original coords
                    convert_boxes_from_input_to_original(boxes_in_input,
                                                        input_w=INPUT_SIZE, input_h=INPUT_SIZE,
                                                        orig_w=orig_w, orig_h=orig_h,
                                                        scale=scale, pad_x=pad_x, pad_y=pad_y)

                    final_boxes = NMS(boxes_in_input, iou_thresh=NMS_IOU_THRESH)

                    # prepare detection arrays for SORT
                    if final_boxes:
                        dets_np = np.array([[b['x1'], b['y1'], b['x2'], b['y2'], b['conf']] for b in final_boxes], dtype=float)
                    else:
                        dets_np = np.empty((0, 5))

                    # tracker update
                    try:
                        if dets_np.size > 0:
                            tracks = self.tracker.update(dets_np)
                        else:
                            tracks = np.empty((0, 5))
                    except Exception as e:
                        print(f"[WARN] Tracker.update raised: {e}")
                        tracks = np.empty((0, 5))

                    # map tracks to detections + violation logic
                    for t in tracks:
                        x1, y1, x2, y2, track_id = map(int, t)
                        # find matching detection class (highest IoU)
                        det_class = None
                        best_iou = 0.0
                        for b in final_boxes:
                            a = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                            iou_v = IoU(a, {'x1': b['x1'], 'y1': b['y1'], 'x2': b['x2'], 'y2': b['y2']})
                            if iou_v > best_iou:
                                best_iou = iou_v
                                det_class = b.get('class_id', None)

                        cls_name = f"class_{det_class}" if det_class is None else (self.class_names[det_class] if det_class < len(self.class_names) else f"class_{det_class}")
                        score = float(np.mean([bb['conf'] for bb in final_boxes])) if final_boxes else 0.0

                        # check violation by name mapping
                        if det_class is not None and (self.class_names[det_class] in self.violation_classes_names):
                            key = f"{track_id}_{det_class}"
                            if key not in seen_violations:
                                seen_violations.add(key)
                                total_violations += 1
                                timestamp = datetime.now().isoformat()
                                snapshot_path = os.path.join(self.snapshot_dir, f"{cls_name}_{track_id}_{frame_idx}.jpg")
                                # draw detection class on frame
                                label = f"{cls_name} {score:.2f}"
                                cv2.putText(frame, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                                # save annotated snapshot for violations
                                snapshot_frame = frame.copy()
                                cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(snapshot_frame, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                cv2.imwrite(snapshot_path, snapshot_frame)


                                insert_violation_async(
                                    self.camera_id,
                                    frame_idx,
                                    int(track_id),
                                    cls_name,
                                    float(score),
                                    [int(x1), int(y1), int(x2), int(y2)],
                                    snapshot_path,
                                )

                                print(f"[VIOLATION] {self.camera_id} | {cls_name} | Track: {track_id} | Frame: {frame_idx}")

                        # annotate frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{track_id}", (x1, max(10, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    frame_idx += 1

                    # display
                    if self.enable_display:
                        win = f"PPE - {self.camera_id}"
                        small = cv2.resize(frame, (960, 540))
                        cv2.imshow(win, small)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

            finally:
                cap.release()
                if self.enable_display:
                    cv2.destroyAllWindows()

                avg_inf = float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0
                print(f"\n[INFO] Completed run for camera {self.camera_id}")
                print(f"[STATS] frames processed (this run): {frame_idx if 'frame_idx' in locals() else 0}")
                print(f"[STATS] total violations (this run): {total_violations}")
                print(f"[STATS] missing_streams (accum): {self.missing_streams}")
                print(f"[STATS] missing_reasons: {self.missing_reasons}")
                print(f"[STATS] avg inference time: {avg_inf:.2f} ms (across runs)\n")

                update_camera_status(self.camera_id, "stopped")
                print(f"[INFO] Retrying camera {self.camera_id} in 2 minutes...\n")
                time.sleep(120)

    def __del__(self):
        try:
            self.rknn.release()
        except Exception:
            pass
