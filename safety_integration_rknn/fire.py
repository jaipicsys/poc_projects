import os
import cv2
import time
import json
import numpy as np
from datetime import datetime
from rknnlite.api import RKNNLite
from db import insert_fire_event_async, update_camera_status

# ---------------- Config ----------------
INPUT_SIZE = 512
CONF_THRESH = 0.25
NMS_IOU_THRESH = 0.45
COOLDOWN_PERIOD = 200  # seconds between fire events

# ---------------- Utilities ----------------
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

# ---------------- RKNN Helpers ----------------
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
            C,H,W = a2.shape
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

def compute_dfl(tensor, dfl_len):
    out = np.zeros(4, dtype=np.float32)
    for b in range(4):
        start = b * dfl_len
        vals = tensor[start:start + dfl_len].astype(np.float64)
        exps = np.exp(vals - np.max(vals))
        probs = exps / (exps.sum() + 1e-6)
        out[b] = float((probs * np.arange(dfl_len)).sum())
    return out

def decode_dfl_heads_python(outputs, attrs, orig_w, orig_h, input_w, input_h,
                            raw_quant, quant_signed, conf_thres=CONF_THRESH):
    boxes = []
    for i in range(0, len(outputs), 3):
        if i+2 >= len(outputs): break
        o_box, o_cls, o_conf = outputs[i:i+3]
        a_box, a_cls, a_conf = attrs[i:i+3]

        def infer_chw(arr):
            if arr is None: return None,None,None
            if isinstance(arr, dict) and 'dims' in arr:
                dims = arr['dims']
                if dims:
                    if len(dims)==4 and dims[0]==1: return dims[1], dims[2], dims[3]
                    if len(dims)==3: return dims[0], dims[1], dims[2]
            if hasattr(arr,'ndim'):
                if arr.ndim==4 and arr.shape[0]==1: return arr.shape[1], arr.shape[2], arr.shape[3]
                if arr.ndim==3: return arr.shape[0], arr.shape[1], arr.shape[2]
            return None,None,None

        cls_C,H,W = infer_chw(o_cls)
        if cls_C is None: continue
        grid_len = H*W
        box_ch,_,_ = infer_chw(o_box)
        if box_ch is None: continue
        dfl_len = max(1, box_ch//4)
        stride = int(input_h//H)

        for y in range(H):
            for x in range(W):
                gidx = y*W + x
                obj_conf = read_out_value(o_conf,a_conf,0,gidx,grid_len,raw_quant,quant_signed)
                if obj_conf<conf_thres: continue
                best_score=0.0
                for c in range(cls_C):
                    cls_score=read_out_value(o_cls,a_cls,c,gidx,grid_len,raw_quant,quant_signed)
                    if cls_score>best_score:
                        best_score=cls_score
                final_conf = float(obj_conf*best_score)
                if final_conf<conf_thres: continue
                before=np.zeros(4*dfl_len,dtype=np.float32)
                for k in range(4*dfl_len):
                    before[k]=read_out_value(o_box,a_box,k,gidx,grid_len,raw_quant,quant_signed)
                box_dec=compute_dfl(before,dfl_len)
                x1=(-box_dec[0]+x+0.5)*stride
                y1=(-box_dec[1]+y+0.5)*stride
                x2=( box_dec[2]+x+0.5)*stride
                y2=( box_dec[3]+y+0.5)*stride
                boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'conf':final_conf})
    return boxes

def convert_boxes_from_input_to_original(boxes, input_w, input_h, orig_w, orig_h, scale, pad_x, pad_y):
    for b in boxes:
        x1 = (b['x1']-pad_x)/scale
        y1 = (b['y1']-pad_y)/scale
        x2 = (b['x2']-pad_x)/scale
        y2 = (b['y2']-pad_y)/scale
        b['x1'] = max(0,min(x1,orig_w-1))
        b['y1'] = max(0,min(y1,orig_h-1))
        b['x2'] = max(0,min(x2,orig_w-1))
        b['y2'] = max(0,min(y2,orig_h-1))

# ---------------- Fire Processor ----------------
class FireProcessor:
    def __init__(self, config, camera_id, rtsp_url, on_done_callback=None):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.on_done_callback = on_done_callback
        self.config = config or {}
        self.enable_display = self.config.get("enable_display", True)
        self.event_class = "fire_detected"

        # Output dirs
        output_dir = self.config.get("output_dir", "output")
        self.snapshot_dir = os.path.join(output_dir, camera_id, "fire_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # RKNN
        model_path = self.config.get("fire_model_path")
        if not model_path:
            raise ValueError("[CONFIG] Missing 'fire_model_path'")
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0: raise RuntimeError("Failed to load RKNN")
        ret = self.rknn.init_runtime()
        if ret != 0: raise RuntimeError("Failed to init RKNN runtime")

    def process_video(self):
        print(f"[INFO] Starting FireProcessor for camera {self.camera_id}...")
        last_fire_time = 0
        first_fire_time = None

        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[WARN] Fire camera {self.camera_id} unavailable. Retrying in 2 minutes...")
                update_camera_status(self.camera_id, "unavailable")
                time.sleep(120)
                continue

            print(f"[INFO] Fire camera {self.camera_id} connected.")
            update_camera_status(self.camera_id, "running")

            try:
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    orig_h, orig_w = frame.shape[:2]
                    padded, scale, pad_x, pad_y = letterbox_image(frame, INPUT_SIZE, INPUT_SIZE)
                    img_input = np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB),0).astype(np.uint8)

                    # RKNN inference
                    try:
                        outputs = self.rknn.inference([img_input])
                    except Exception as e:
                        print(f"[ERROR] Inference failed: {e}")
                        break

                    # attrs + quant
                    attrs = [get_attr_info(self.rknn,i) for i in range(len(outputs))]
                    raw_quant = any([o.dtype in (np.int8,np.uint8) for o in outputs])
                    quant_signed = True
                    if attrs and attrs[0]: quant_signed = bool(attrs[0].get('is_signed',True))

                    # decode
                    boxes_in_input = decode_dfl_heads_python(
                        outputs, attrs, orig_w, orig_h,
                        INPUT_SIZE, INPUT_SIZE,
                        raw_quant, quant_signed, CONF_THRESH
                    )
                    convert_boxes_from_input_to_original(boxes_in_input, INPUT_SIZE, INPUT_SIZE,
                                                        orig_w, orig_h, scale, pad_x, pad_y)

                    # cooldown & save
                    current_time = time.time()
                    if boxes_in_input and current_time - last_fire_time >= COOLDOWN_PERIOD:
                        for b in boxes_in_input:
                            x1,y1,x2,y2 = map(int,[b['x1'],b['y1'],b['x2'],b['y2']])
                            conf = float(b['conf'])
                            timestamp = datetime.now().isoformat()
                            snapshot_frame = frame.copy()
                            cv2.rectangle(snapshot_frame,(x1,y1),(x2,y2),(0,0,255),2)
                            cv2.putText(snapshot_frame,f"{self.event_class} {conf:.2f}",(x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
                            snapshot_path = os.path.join(self.snapshot_dir,f"fire_{frame_idx}.jpg")
                            cv2.imwrite(snapshot_path,snapshot_frame)

                            insert_fire_event_async(
                                self.camera_id,
                                frame_idx,
                                track_id=0,
                                event=self.event_class,
                                confidence=conf,
                                bbox=(x1,y1,x2,y2),
                                snapshot_path=snapshot_path,
                                timestamp=timestamp
                            )

                            last_fire_time = current_time
                            first_fire_time = timestamp
                            print(f"[INFO][{self.camera_id}] Fire detected! Ignoring further detections for {COOLDOWN_PERIOD}s")
                            break  # one per frame

                    # draw for display
                    for b in boxes_in_input:
                        x1,y1,x2,y2 = map(int,[b['x1'],b['y1'],b['x2'],b['y2']])
                        conf = float(b['conf'])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                        cv2.putText(frame,f"{self.event_class} {conf:.2f}",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

                    frame_idx +=1
                    if self.enable_display:
                        win = f"Fire - {self.camera_id}"
                        cv2.imshow(win,cv2.resize(frame,(640,360)))
                        if cv2.waitKey(1)&0xFF==ord('q'): break

            finally:
                cap.release()
                if self.enable_display:
                    cv2.destroyAllWindows()
                update_camera_status(self.camera_id,"stopped")
                if first_fire_time:
                    print(f"[INFO][{self.camera_id}] Last fire at frame {frame_idx}, time {first_fire_time}")
                else:
                    print(f"[INFO][{self.camera_id}] No fire detected.")
                print(f"[INFO] Retrying in 2 minutes...")
                time.sleep(120)

    def __del__(self):
        try: self.rknn.release()
        except Exception: pass
