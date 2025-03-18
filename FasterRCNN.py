import torch
import torchvision
import cv2
import numpy as np

# RTSP stream URL
rtsp_url = "rtsp://tanvir:iBOS%40321@192.168.2.19:554/"

# Load pre-trained Faster R-CNN model (ResNet50 backbone)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device).eval()

# COCO dataset class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

def detect_objects(frame, threshold=0.6):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]

    boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
    labels = pred['labels'][pred['scores'] > threshold].cpu().numpy()
    scores = pred['scores'][pred['scores'] > threshold].cpu().numpy()

    return boxes, labels, scores

# Real-time processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break

    # Resize frame to improve speed (optional)
    frame_resized = cv2.resize(frame, (1080, 720))

    boxes, labels, scores = detect_objects(frame_resized)

    # Draw bounding boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

        # Draw rectangle
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display label and confidence
        cv2.putText(frame_resized, f'{class_name}: {score:.2f}',
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("RTSP - Faster R-CNN Detection", frame_resized)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
