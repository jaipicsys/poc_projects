# config_utils.py
import json, threading

CONFIG_LOCK = threading.Lock()

with open("config.json") as f:
    CONFIG = json.load(f)

def reload_email_config():
    global CONFIG
    with CONFIG_LOCK:
        try:
            with open("config.json") as f:
                new_cfg = json.load(f)
            if "email" in new_cfg:
                CONFIG["email"] = new_cfg["email"]
            if "report_jobs" in new_cfg:
                CONFIG["report_jobs"] = new_cfg["report_jobs"]
        except Exception as e:
            print(f"[WARN] Cannot reload email config: {e}")
            
def update_report_job(job, enabled):
    """Update job state in config.json and memory CONFIG"""
    global CONFIG
    with CONFIG_LOCK:
        try:
            with open("config.json") as f:
                cfg = json.load(f)
            if "report_jobs" not in cfg:
                cfg["report_jobs"] = {}
            cfg["report_jobs"][job] = enabled
            with open("config.json", "w") as f:
                json.dump(cfg, f, indent=4)
            CONFIG["report_jobs"] = cfg["report_jobs"]
        except Exception as e:
            print(f"[ERROR] Failed to update job state: {e}")
            
def is_job_enabled(job_type):
    """Check if a job is enabled dynamically"""
    with CONFIG_LOCK:
        return CONFIG["report_jobs"].get(job_type, False)