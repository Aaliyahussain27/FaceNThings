from ultralytics import YOLO
import cv2

# YOLOv8 model load
model = YOLO('yolov8n.pt')
print("YOLOv8 model loaded!")

# Target objects
target_classes = ['bottle', 'cup', 'cell phone','cat','book']
print(f"Detecting: {target_classes}")

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera nahi khula!")
    exit()

print("Webcam started!")
print("Object detection active! Press 'q' to quit")

# Main loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Frame nahi aa raha!")
        break
    
    # YOLOv8 detection
    results = model(frame, verbose=False)
    result = results[0]
    
    # Process each detection
    for box in result.boxes:
        # Coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Confidence and class
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        
        # Filter: only target objects
        if class_name not in target_classes:
            continue
        
        # Filter: high confidence only
        if confidence < 0.5:
            continue
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('Object Detection - Press Q to Quit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Done!")