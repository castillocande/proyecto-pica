import cv2
import argparse
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

model = YOLO("yolov11/best.pt")
video_path = "videos/2023-03-22/video/2023-03-22 13_28_32.mp4"

cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model.track(frame, persist=True, conf=0.2, iou=0.3)[0]

        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            frame = result.plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y))) 
                if len(track) > 30:  
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(200, 20, 100), thickness=2)

        cv2.imshow("YOLO11 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()