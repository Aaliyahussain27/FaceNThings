from ultralytics import YOLO
import cv2

print("="*60)
print("FaceNThings - Loading Models...")
print("="*60)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error: Haar Cascade model not found.")
    exit()
print("Face detection loaded successfully.")

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')
print("YOLOv8 model loaded successfully.")

# Objects to detect
target_classes = ['bottle', 'cup', 'cell phone', 'cat', 'book']
print(f"Detecting objects: {target_classes}")

print("="*60)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Webcam initialized.")
print("\nFaceNThings Active:")
print("  • Face detection")
print("  • Object detection")
print("\nPress 'q' to quit")
print("="*60)


# ========== Main Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Face Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Object Detection
    results = yolo_model(frame, verbose=False)
    result = results[0]
    
    object_count = 0
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = yolo_model.names[class_id]
        
        if class_name not in target_classes or confidence < 0.5:
            continue
        
        object_count += 1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Statistics
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display
    cv2.imshow('FaceNThings - Press Q to Quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nShutting down...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n" + "="*60)
print("FaceNThings terminated.")
print("="*60)
