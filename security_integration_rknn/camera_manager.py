import os
import sys
import signal
import subprocess
import json
from threading import Thread
from db import add_history
from loitering import LoiteringProcessor
from perimeter import PerimeterProcessor

class CameraManager:
    def __init__(self, camera_config):
        with open("config.json") as f:
            self.config = json.load(f)

        self.processors = {}
        self.threads = {}

        camera_types = self.config.get("camera_types", {})

        for cam_id, rtsp_url in camera_config.items():
            cam_type = camera_types.get(cam_id, "loitering")  # default to loitering if not specified

            if cam_type == "loitering":
                processor = LoiteringProcessor(
                    config=self.config,
                    camera_id=cam_id,
                    rtsp_url=rtsp_url,
                    on_done_callback=self.on_camera_done
                )
            elif cam_type == "perimeter":
                processor = PerimeterProcessor(
                    config=self.config,
                    camera_id=cam_id,
                    rtsp_url=rtsp_url,
                    on_done_callback=self.on_camera_done
                )
            else:
                print(f"[WARN] Unknown type '{cam_type}' for camera {cam_id}, skipping.")
                continue

            self.processors[cam_id] = processor
            self.threads[cam_id] = Thread(target=processor.process_video, daemon=True)

    def on_camera_done(self, camera_id):
        print(f"[INFO] Camera {camera_id} processing completed. Restarting its thread...")
        add_history(camera_id, "completed")

        processor = self.processors[camera_id]
        thread = Thread(target=processor.process_video, daemon=True)
        self.threads[camera_id] = thread
        thread.start()

    def restart_server(self):
        import time
        print("[INFO] Restarting server in 1 second...")
        time.sleep(1)

        python = sys.executable
        subprocess.Popen([python, "app.py"])
        os._exit(0)

    def start_all(self):
        for cam_id, thread in self.threads.items():
            thread.start()
            print(f"[INFO] Camera thread {cam_id} started")
