import cv2, os, json, time, numpy as np

# 1. Load config.json
with open("config.json") as f:
    config = json.load(f)

output_dir = config.get("output_dir", "output")
camera_ids = list(config["rtsp_urls"].keys())  # ["cam1", "cam2", "cam3", "cam4"]

# 2. Main display loop
while True:
    frames = []
    for cam_id in camera_ids:
        img_path = os.path.join(output_dir, cam_id, "latest.jpg")
        if os.path.exists(img_path):
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.resize(frame, (600, 400))
                cv2.putText(frame, cam_id, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                frames.append(frame)
            else:
                # Placeholder if image missing
                frames.append(np.zeros((400, 600, 3), dtype=np.uint8))
        else:
            frames.append(np.zeros((400, 600, 3), dtype=np.uint8))

    # 3. Combine frames into 2×2 grid
    if len(frames) >= 4:
        top = np.hstack(frames[:2])
        bottom = np.hstack(frames[2:])
        grid = np.vstack([top, bottom])
    else:
        grid = np.hstack(frames)

    # 4. Display the combined grid
    cv2.imshow("Live Cameras", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Control refresh rate (20 FPS)
    time.sleep(0.05)

cv2.destroyAllWindows()
