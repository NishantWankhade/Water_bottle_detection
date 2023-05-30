from ultralytics import YOLO
import datetime
import cv2

# Load custom model
model = YOLO("custom_waterBottle_detection_model.pt")

# Video Capture object
video_cap = cv2.VideoCapture(0)

# Present time
start = datetime.datetime.now()
while(video_cap.isOpened()):

    # Present time
    end = datetime.datetime.now()
    total_time = (end - start).total_seconds()

    if(total_time == 15):
        break

    ret, frame = video_cap.read()

    if not ret:
        break

    annotated_frame = model(frame)[0].plot()

    cv2.imshow("Detected Object", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video_cap.release()
cv2.destroyAllWindows()
