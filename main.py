import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture(0)
#Insert video link(as mp4) in the detect file in order to track that

model = YOLO("yolov8m.pt")
#results = model.train(data='coco128.yaml', epochs=5, imgsz=640)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    results = results[0]
    bboxes = np.array(results.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(results.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        print("x", x, "y", y)

    cv2.imshow("IMG", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()