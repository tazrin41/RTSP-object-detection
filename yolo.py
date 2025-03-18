from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (YOLOv8n - lightweight version)
model = YOLO("yolov8n.pt")

# Load a local image (replace with your image path)
img = cv2.imread("yolo_test.jpg")

# Use YOLOv8 to detect objects in the image
results = model(img)

# Extract bounding boxes, confidences, and labels
for result in results:
    boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
    confidences = result.boxes.conf  # Confidence scores
    labels = result.boxes.cls  # Class labels

    print(f"Detected {len(boxes)} boxes.")
    for i, box in enumerate(boxes):
        print(f"Box {i}: {box}, Confidence: {confidences[i]}, Label: {labels[i]}")

        # Draw bounding box and label on the frame
        if labels[i] == 0 and confidences[i] > 0.5:  # Class 0 corresponds to 'person'
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"Person {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the image with detections
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
