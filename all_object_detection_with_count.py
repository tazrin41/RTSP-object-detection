import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLOv8 model (nano version for speed, use 'yolov8s.pt' for better accuracy)
model = YOLO("yolov8l.pt")

# RTSP URL
rtsp_url = "rtsp://tanvir:iBOS%40321@192.168.2.19:554/"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

# COCO Class Labels
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break

    # Resize the frame for consistent detection
    frame_resized = cv2.resize(frame, (1080, 720))

    # Use YOLOv8 to detect objects
    results = model(frame_resized)

    # Dictionary to store object counts
    object_counts = defaultdict(int)

    # Extract bounding boxes, confidences, and labels
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        confidences = result.boxes.conf  # Confidence scores
        labels = result.boxes.cls  # Class labels

        print(f"Frame {frame_count}: Detected {len(boxes)} objects.")

        for i, box in enumerate(boxes):
            label_id = int(labels[i])  # Convert class ID to integer
            confidence = confidences[i]

            # Ensure the label ID exists in our COCO_CLASSES mapping
            label_name = COCO_CLASSES.get(label_id, "Unknown")

            # Count detected objects
            object_counts[label_name] += 1

            # Only display detections above a confidence threshold
            if confidence > 0.5:  # Increased threshold for better accuracy
                x1, y1, x2, y2 = box

                # Draw bounding box and label on the frame
                cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"{label_name} {confidence:.2f}",
                            (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the count of each detected object class on the frame
    y_offset = 30  # Start displaying object count from this y position
    for obj, count in object_counts.items():
        count_text = f"{obj}: {count}"
        cv2.putText(frame_resized, count_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30  # Move down for next object class count

    # Show the frame with detections and counts
    cv2.imshow("RTSP Stream with YOLOv8 Detection", frame_resized)

    # Increment frame count
    frame_count += 1

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
