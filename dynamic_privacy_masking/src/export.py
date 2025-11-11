import cv2
from retinaface import RetinaFace  # Make sure you have retinaface module installed
import argparse

# ------------------ Arguments ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='/home/jai/work/poc_projects/dynamic_privacy_masking/data/subway.mp4')
parser.add_argument('--output', type=str, required=True, help='output.mp4')
parser.add_argument('--blur', type=int, default=99, help='Kernel size for Gaussian blur')
args = parser.parse_args()

# ------------------ Load video ------------------
cap = cv2.VideoCapture(args.input)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ------------------ Process frames ------------------
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = RetinaFace.detect_faces(frame)
    for key in faces.keys():
        x1, y1, x2, y2 = faces[key]['facial_area']
        face_region = frame[y1:y2, x1:x2]
        # Apply Gaussian blur
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (args.blur, args.blur), 30)

    out.write(frame)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f'Processed {frame_count} frames...')

cap.release()
out.release()
print(f'✅ Done! Masked video saved at: {args.output}')
