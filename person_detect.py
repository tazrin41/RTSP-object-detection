import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (YOLOv8n - lightweight version)
model = YOLO("yolov8n.pt")

# RTSP URL with credentials
rtsp_url = "rtsp://tanvir:iBOS%40321@192.168.2.19:554/"

# Open the RTSP stream using ffmpeg backend for better compatibility
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break

    # Resize the frame to 720x1080 (or as required)
    frame_resized = cv2.resize(frame, (1080, 720))

    # Use YOLOv8 to detect persons
    results = model(frame_resized)

    # Extract bounding boxes, confidences, and labels
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels

        print(f"Frame {frame_count}: Detected {len(boxes)} boxes.")

        for i, box in enumerate(boxes):
            print(f"Box {i}: {box}, Confidence: {confidences[i]}, Label: {labels[i]}")

            # Only process person (class 0) detections with confidence > 0.3
            if labels[i] == 0 and confidences[i] > 0.3:  # Class 0 corresponds to 'person'
                x1, y1, x2, y2 = box
                confidence = confidences[i]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Person {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with detections
    cv2.imshow("RTSP Stream with YOLOv8 Detection", frame_resized)

    # Increment frame count
    frame_count += 1

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
