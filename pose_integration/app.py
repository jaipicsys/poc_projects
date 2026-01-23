import os
import json
import time
import threading
import cv2
from flask import Flask, Response, render_template_string
from routes import api_bp
from fall_event import FallProcessor
from db import (
    init_patient_alerts_db, init_auth_db,
    init_history_db, init_camera_status_table
)
from flask_cors import CORS
from camera_manager import CameraManager

# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
with open("config.json") as f:
    CONFIG = json.load(f)

FLASK_PORT = CONFIG.get("flask_port", 5000)
camera_config = CONFIG.get("rtsp_urls", {})

# ---------------------------------------------------------
# Flask setup
# ---------------------------------------------------------
app = Flask(__name__)
CORS(
    app,
    supports_credentials=True,
    origins=["*"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)
app.register_blueprint(api_bp, url_prefix='/api')

# ---------------------------------------------------------
# Global frame store for MJPEG streaming
# ---------------------------------------------------------
latest_frames = {}          # camera_id -> latest annotated frame
frames_lock = threading.Lock()

def on_frame(camera_id, frame):
    """Callback from each FallProcessor when a new annotated frame is ready."""
    with frames_lock:
        latest_frames[camera_id] = frame.copy()

# ---------------------------------------------------------
# MJPEG generator and routes
# ---------------------------------------------------------
def mjpeg_generator(camera_id):
    """Yields JPEG frames for streaming to browser."""
    while True:
        with frames_lock:
            frame = latest_frames.get(camera_id)
        if frame is None:
            time.sleep(0.03)
            continue
        ok, jpg = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    """Simple HTML page showing all available camera feeds."""
    cams = list(camera_config.keys())
    html = """
    <html><head><title>Pose Streams</title></head>
    <body style="background:#111;color:#eee;font-family:sans-serif">
      <h2>Live Camera Streams</h2>
      {% for cam in cams %}
        <div><h3>{{cam}}</h3>
        <img src="/video_feed/{{cam}}" style="max-width:90vw;border:2px solid #333"></div><hr>
      {% endfor %}
      <p>Press Ctrl+C in the server console to stop.</p>
    </body></html>
    """
    return render_template_string(html, cams=cams)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Route that streams MJPEG feed for a given camera."""
    return Response(mjpeg_generator(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == '__main__':
    # Initialize all databases
    init_auth_db()
    init_history_db()
    init_camera_status_table()
    init_patient_alerts_db()

    # Start camera threads with callback for live frame updates
    manager = CameraManager(camera_config, on_frame_callback=on_frame)
    manager.start_all()

    print(f"[INFO] Flask server running on port {FLASK_PORT}")
    print(f"[INFO] Visit http://localhost:{FLASK_PORT}/ to view live feeds")

    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt — stopping server and cameras...")
        manager.stop_all()
        os._exit(0)
