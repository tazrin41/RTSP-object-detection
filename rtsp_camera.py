import cv2

# RTSP URL
rtsp_url = "rtsp://tanvir:iBOS%40321@192.168.2.19:554/"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open RTSP stream")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame")
        break

    # Resize the frame to 1080x720
    frame_resized = cv2.resize(frame, (1200, 800))

    # Show the resized frame
    cv2.imshow("RTSP Stream", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
