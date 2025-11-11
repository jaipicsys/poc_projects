import os
import sys
import signal
import subprocess
import json
import asyncio
from threading import Thread
from iv_tracker import IVFluidTracker
from db import add_history

class CameraManager:
    def __init__(self, camera_config, bed_map):
        with open("config.json") as f:
            self.config = json.load(f)
        self.processors = {}
        self.threads = {}
        cut_config = self.config.get("cut_percent", {})
        default_cut = self.config.get("default_cut_percent", {"top": 0, "bottom": 12})
        for cam_id, rtsp_url in camera_config.items():
            bed_no = bed_map.get(cam_id, "UNKNOWN")

            cam_cut = cut_config.get(cam_id, {})
            top_cut = cam_cut.get("top", default_cut.get("top"))
            bottom_cut = cam_cut.get("bottom", default_cut.get("bottom"))
            processor = IVFluidTracker(
                config=self.config,
                camera_id=cam_id,
                rtsp_url=rtsp_url,
                bed_no=bed_no,
                on_done_callback=self.on_camera_done,
                top_cut_percent=top_cut,
                bottom_cut_percent=bottom_cut
            )
            self.processors[cam_id] = processor
            thread = Thread(target=lambda p=processor: asyncio.run(p.process_video()), daemon=True)
            self.threads[cam_id] = thread

    def on_camera_done(self, camera_id):
        print(f"[INFO] Camera {camera_id} processing completed. Restarting its thread...")
        add_history(camera_id, "completed")
        processor = self.processors[camera_id]
        thread = Thread(target=lambda p=processor: asyncio.run(p.process_video()), daemon=True)
        self.threads[camera_id] = thread
        thread.start()

    def start_all(self):
        for cam_id, thread in self.threads.items():
            thread.start()
            print(f"[INFO] Camera thread {cam_id} started")
