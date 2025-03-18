import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (YOLOv8n - lightweight version)
model = YOLO("yolov8l.pt")

# RTSP URL with credentials
rtsp_url = "rtsp://tanvir:iBOS%40321@192.168.2.19:554/"

# Open the RTSP stream using ffmpeg backend for better compatibility
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

# COCO Class Labels (YOLOv8 uses COCO dataset for classification)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break

    # Resize the frame to 720x1080 (or as required)
    frame_resized = cv2.resize(frame, (1080, 720))

    # Use YOLOv8 to detect objects
    results = model(frame_resized)

    # Extract bounding boxes, confidences, and labels
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels

        print(f"Frame {frame_count}: Detected {len(boxes)} boxes.")

        for i, box in enumerate(boxes):
            print(f"Box {i}: {box}, Confidence: {confidences[i]}, Label (ID): {labels[i]}")

            # Get the label name from the class ID
            label_id = int(labels[i])  # Convert class ID to int
            if label_id < len(COCO_CLASSES):
                label_name = COCO_CLASSES[label_id]  # Convert class ID to class name
            else:
                label_name = "Unknown"  # In case the label ID is out of bounds

            # Apply higher confidence threshold to reduce incorrect labeling
            if confidences[i] > 0.6:  # Confidence threshold increased to 0.6
                x1, y1, x2, y2 = box
                confidence = confidences[i]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label_name} {confidence:.2f}", (int(x1), int(y1) - 10),
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
