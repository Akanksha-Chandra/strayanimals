import cv2
import torch
import pygame

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.mp3')

# Load YOLOv5 model (pre-trained YOLOv5 small model)
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open video capture (0 for webcam, or replace with video file path)
video_source = 0  # Use 0 for webcam or replace with 'test_video.mp4' for video testing
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Cannot access the video source.")
    exit()

print("Press 'q' to exit the application.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from the video source.")
        break

    # Perform inference on the current frame
    results = model(frame)

    # Extract detections and draw bounding boxes for specified animals
    detections = results.pandas().xyxy[0]  # Get bounding box predictions as a Pandas DataFrame
    for _, row in detections.iterrows():
        # Filter by class and confidence
        if row['name'] in ['dog', 'cow', 'cat'] and row['confidence'] > 0.6:  # Confidence > 60%
            print(f"Detected: {row['name']} with confidence: {row['confidence']:.2f}")

            # Draw bounding box
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, row['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Play alert sound
            try:
                alert_sound.play()
            except Exception as e:
                print(f"Error playing sound: {e}")

    # Show the video with detection boxes
    cv2.imshow('Animal Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting application...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
