import os
import sys
import signal
import subprocess
import json
from threading import Thread
from db import add_history
from fall_event import FallProcessor


class CameraManager:
    def __init__(self, camera_config, on_frame_callback=None):
        with open("config.json") as f:
            self.config = json.load(f)
        self.processors = {}
        self.threads = {}
        self.on_frame_callback = on_frame_callback  # <-- new addition

        for cam_id, rtsp_url in camera_config.items():
            processor = FallProcessor(
                config=self.config,
                camera_id=cam_id,
                rtsp_url=rtsp_url,
                on_done_callback=self.on_camera_done,
                on_frame_callback=self.on_frame_callback,   # <-- pass callback here
            )
            self.processors[cam_id] = processor
            thread = Thread(target=processor.process_video, daemon=True)
            self.threads[cam_id] = thread


    def on_camera_done(self, camera_id):
        print(f"[INFO] Camera {camera_id} processing completed. Restarting server...")
        add_history(camera_id, "completed")
        self.restart_server()

    # def on_camera_done(self, camera_id):
    #     print(f"[INFO] Camera {camera_id} processing completed. Restarting its thread...")
    #     add_history(camera_id, "completed")

    #     # Restart only that camera
    #     processor = self.processors[camera_id]
    #     thread = Thread(target=processor.process_video, daemon=True)
    #     self.threads[camera_id] = thread
    #     thread.start()

    def restart_server(self):
        import os
        import sys
        import subprocess
        import time

        print("[INFO] Restarting server in 1 second...")
        time.sleep(1)  # Give a moment for cleanup

        # Spawn a new subprocess to start the server
        python = sys.executable
        subprocess.Popen([python, "app.py"])

        # Exit the current process
        os._exit(0)


        # Restart the server
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def start_all(self):

        # Start all camera threads
        for cam_id, thread in self.threads.items():
            thread.start()
            print(f"[INFO] Camera thread {cam_id} started")
