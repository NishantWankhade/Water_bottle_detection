from ultralytics import YOLO
import cv2


# defining some constants
CONFIDENCE_THRESHOLD = 0.5
GREEN = (0, 255, 0)


"""Video Capture and writing Object"""
# initialize the video capture object
video_cap = cv2.VideoCapture(0)

if(video_cap.isOpened() == False):
    print("Error reading video file..!")

# Width and the height of the video
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))

size = (frame_width, frame_height)

# video writer object
video_write = cv2.VideoWriter(
    'capture_video.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    10,
    size
)


"""Loading a Custom Model"""
model = YOLO('custom_waterBottle_detection_model.pt')

while(True):

    # Video capture Function returns the current frame
    # and the a boolean value indicating whether the frame
    # is captured or not
    ret, frame = video_cap.read()

    if not ret:
        break

    """Detection per frame"""
    detections = model(frame)[0]

    # Looping over detections
    for data in detections.boxes.data.tolist():

        # Extract the confidence i.e. probability associated with the detection
        confidence = data[4]

        # Filter out the weak detection
        # having less confidence than the confidence_threshold
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[1]), int(
            data[2]), int(data[3]), int(data[4])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)

        cv2.putText(frame, "Bottle", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

        # Writing the frame
        video_write.write(frame)

        # Display the frame
        cv2.imshow("Frame Window", frame)

        if cv2.waitKey(1) == ord('s'):
            break
    
    
    cv2.imshow("Frame Window", frame)
    if cv2.waitKey(1) == ord('s'):
        break

# Releasing the CV objects
video_cap.release()
video_write.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
