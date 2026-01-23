import cv2
cap = cv2.VideoCapture("/home/jai/udit_backend/pose_detection/test_videos/fall_test7.mp4")
print("opened:", cap.isOpened())
ret, frame = cap.read()
print("read:", ret, "shape:" if ret else None, frame.shape if ret else None)
cap.release()
