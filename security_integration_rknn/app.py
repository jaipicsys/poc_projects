import os
import json
from flask import Flask
from threading import Thread
from routes import api_bp
from db import (
    start_db_thread,
    stop_db_thread,
    init_history_db,
    init_auth_db,
    init_camera_status_table
)
from flask_cors import CORS
from camera_manager import CameraManager
from report_scheduler import schedule_reports

# Load config.json
with open("config.json") as f:
    CONFIG = json.load(f)

import warnings
warnings.filterwarnings("ignore")


FLASK_PORT = CONFIG.get("flask_port", 5000)  # fallback to 5000 if not set
camera_config = CONFIG.get("rtsp_urls", {})

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(api_bp, url_prefix='/api')

def start_scheduler():
    """Start the report scheduler in a background thread"""
    schedule_reports()  # starts APScheduler

if __name__ == '__main__':
    # Initialize DBs relevant to history and events
    init_history_db()
    init_auth_db()
    init_camera_status_table()
    # Start async DB workers
    start_db_thread()

    # Start background scheduler
    Thread(target=start_scheduler, daemon=True).start()

    # Start camera processing using CameraManager
    manager = CameraManager(camera_config)
    manager.start_all()

    try:
        app.run(host='0.0.0.0', port=FLASK_PORT)
    except KeyboardInterrupt:
        print("\n[INFO] Received keyboard interrupt. Stopping server...")
        stop_db_thread()
        os._exit(0)
