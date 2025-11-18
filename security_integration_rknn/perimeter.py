# perimeter.py
import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from typing import Optional
from sort import Sort
from rknnlite.api import RKNNLite
from db import insert_perimeter_event_async, start_db_thread, update_camera_status

# ---------------- Configurable constants ----------------
INPUT_SIZE = 512
CONF_THRESH = 0.3
NMS_IOU_THRESH = 0.4
TRACKER_PARAMS = dict(max_age=90, min_hits=1, iou_threshold=0.2)
OUTPUT_DIR_DEFAULT = "output"

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

def get_attr_info(rknn, idx):
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
    a = out_arr
    if raw_quant and a.dtype in (np.int8, np.uint8):
        flat = a.flatten()
        idx = ch_index * grid_len + grid_idx
        val_q = int(flat[idx])
        return (float(val_q) - float(attr.get('zp', 0))) * float(attr.get('scale', 1.0))
    else:
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
    boxes = []
    for i in range(0, len(outputs), 3):
        if i + 2 >= len(outputs):
            break
        o_box = outputs[i]
        o_cls = outputs[i + 1]
        o_conf = outputs[i + 2]
        a_box = attrs[i]
        a_cls = attrs[i + 1]
        a_conf = attrs[i + 2]

        def infer_chw(arr):
            if arr is None:
                return None, None, None
            if isinstance(arr, dict) and 'dims' in arr:
                dims = arr['dims']
                if dims and len(dims) >= 3:
                    if len(dims) == 4 and dims[0] == 1:
                        return dims[1], dims[2], dims[3]
                    if len(dims) == 3:
                        return dims[0], dims[1], dims[2]
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

        o_box_arr, o_cls_arr, o_conf_arr = o_box, o_cls, o_conf

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
                before = np.zeros(4 * dfl_len, dtype=np.float32)
                for k in range(4 * dfl_len):
                    before[k] = read_out_value(o_box_arr, a_box, k, gidx, grid_len, raw_quant, quant_signed)
                box_dec = compute_dfl(before, dfl_len)
                x1 = (-box_dec[0] + x + 0.5) * stride
                y1 = (-box_dec[1] + y + 0.5) * stride
                x2 = ( box_dec[2] + x + 0.5) * stride
                y2 = ( box_dec[3] + y + 0.5) * stride
                boxes.append({'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                              'conf': float(final_conf), 'class_id': int(best_cls)})
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

# ---------------- Perimeter Processor ----------------
class PerimeterProcessor:
    def __init__(self, config: Optional[dict], camera_id: str, rtsp_url: str, on_done_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.config = config or {}

        # Directories
        output_dir = self.config.get("output_dir", OUTPUT_DIR_DEFAULT)
        self.snapshot_dir = os.path.join(output_dir, camera_id, "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # RKNN
        model_path = self.config.get("perimeter_model_path", "model.rknn")
        self.rknn = RKNNLite()
        print(f"[INFO] Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Failed to initialize RKNN runtime")

        # Tracker & state
        self.tracker = Sort(**TRACKER_PARAMS)
        self.logged_track_ids = set()  # log each track once

        # Zones & confidence
        self.zones = self.config.get("perimeter_zones", {}).get(camera_id, [])
        self.match_confidence = float(self.config.get("perimeter_conf", CONF_THRESH))
        self.input_size = INPUT_SIZE
        self.event_class = "perimeter_breach"

    # ---------------- Helpers ----------------
    def _point_in_any_zone(self, point):
        px, py = point
        for zone in self.zones:
            pts = np.array(zone.get("points", []), dtype=np.int32)
            if pts.size == 0:
                continue
            if cv2.pointPolygonTest(pts, (int(px), int(py)), False) >= 0:
                return True
        return False

    def _save_snapshot(self, frame, frame_idx, track_id, bbox=None, conf=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"perimeter_{self.camera_id}_{track_id}_{frame_idx}_{ts}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        snap = frame.copy()
        if bbox:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 2)
            except:
                pass
        label = self.event_class
        if conf is not None:
            label += f" {conf:.2f}"
        cv2.putText(snap, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(path, snap)
        return path

    # ---------------- Main Loop ----------------
    def process_video(self):
        print(f"[INFO] Starting PerimeterProcessor for camera {self.camera_id}...")
        update_camera_status(self.camera_id, "starting")
        start_db_thread()

        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"[ERROR] Camera {self.camera_id} cannot be opened")
            update_camera_status(self.camera_id, "stopped")
            return

        print(f"[INFO] Perimeter camera {self.camera_id} connected successfully.")
        update_camera_status(self.camera_id, "running")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_idx += 1
                continue
            orig_h, orig_w = frame.shape[:2]

            # Optional: draw zones
            for zone in self.zones:
                pts = np.array(zone.get("points", []), dtype=np.int32)
                if pts.size:
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

            # Prepare RKNN input
            padded, scale, pad_x, pad_y = letterbox_image(frame, self.input_size, self.input_size)
            img_input = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_input, 0).astype(np.uint8)

            # RKNN inference
            try:
                outputs = self.rknn.inference([img_input])
                attrs = [None] * len(outputs)
                boxes = decode_dfl_heads_python(
                    outputs, attrs,
                    orig_w, orig_h,
                    self.input_size, self.input_size,
                    raw_quant=True,
                    quant_signed=True,
                    conf_thres=self.match_confidence
                )
                convert_boxes_from_input_to_original(boxes, self.input_size, self.input_size, orig_w, orig_h, scale, pad_x, pad_y)
                boxes = NMS(boxes, NMS_IOU_THRESH)
                boxes = [b for b in boxes if b['class_id'] == 0]
            except Exception as e:
                print(f"[ERROR] decode failed at frame {frame_idx}: {e}")
                boxes = []

            # SORT tracking
            dets_np = np.array([[b['x1'], b['y1'], b['x2'], b['y2'], b['conf']] for b in boxes], dtype=float) if boxes else np.empty((0,5))
            tracks = self.tracker.update(dets_np)

            for t in tracks:
                x1, y1, x2, y2, track_id = map(int, t)

                # Corner-based perimeter detection
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                if any(self._point_in_any_zone(p) for p in corners):
                    if track_id not in self.logged_track_ids:
                        self.logged_track_ids.add(track_id)

                        snapshot_path = self._save_snapshot(frame, frame_idx, track_id, bbox=[x1, y1, x2, y2])

                        try:
                            insert_perimeter_event_async(
                                self.camera_id,
                                frame_idx,
                                track_id,
                                self.event_class,
                                bbox=[x1, y1, x2, y2],
                                snapshot_path=snapshot_path,
                                timestamp=datetime.now().isoformat(),
                                confidence=float(np.max([b['conf'] for b in boxes])) if boxes else None
                            )
                            print(f"[PERIMETER] {self.camera_id} | track:{track_id} | frame:{frame_idx} | INSIDE ROI")
                        except Exception as e:
                            print(f"[ERROR] insert_perimeter_event_async failed: {e}")

            frame_idx += 1

        cap.release()
        update_camera_status(self.camera_id, "stopped")
        if self.on_done_callback:
            self.on_done_callback(self.camera_id)

    def __del__(self):
        try:
            self.rknn.release()
        except Exception:
            pass
