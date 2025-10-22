import cv2
from ultralytics import YOLO

# Load static image
static_img = cv2.imread('privacy.png')
if static_img is None:
    raise FileNotFoundError("Image 'privacy.png' not found.")

# Resize parameters
display_width = 640
display_height = 480
static_img = cv2.resize(static_img, (display_width, display_height))

# YOLOv8 model
yolo_model = YOLO('yolov8s.pt')  # You can use 'yolov8s.pt' for better accuracy

# Webcam
cap = cv2.VideoCapture(0)

# List of violating objects
violating_objects = ['cell phone', 'camera', 'laptop', 'tv', 'monitor']
allow_classes = ['person'] + violating_objects  # Only draw boxes for these

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (display_width, display_height))
    annotated_frame = frame.copy()

    # ---- YOLO Detection ----
    yolo_results = yolo_model(frame, verbose=False)[0]

    person_count = 0
    object_violation = False

    for box in yolo_results.boxes:
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id].lower()
        conf = float(box.conf[0])

        if conf < 0.4:
            continue  # Skip low-confidence detections

        if label not in allow_classes:
            continue  # Skip drawing boxes for unwanted objects (e.g. chair, bottle)

        # Count persons and check for violations
        if label == 'person':
            person_count += 1
        elif label in violating_objects:
            object_violation = True

        # Draw bounding box
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        color = (0, 0, 255) if label in violating_objects else (0, 255, 0)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---- Privacy Logic ----
    privacy_ok = (person_count == 1 and not object_violation)

    if privacy_ok:
        processed_img = static_img.copy()
        cv2.putText(processed_img, "✅ Privacy Maintained", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        processed_img = cv2.GaussianBlur(static_img, (55, 55), 0)
        warning_text = "⚠ Privacy Violation: "
        if person_count != 1:
            warning_text += f"{person_count} People Detected. "
        if object_violation:
            warning_text += "Device Detected!"
        cv2.putText(processed_img, warning_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ---- Labels on Webcam Side ----
    cv2.putText(annotated_frame, "📷 Live Camera", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ---- Combine and Show ----
    combined = cv2.hconcat([annotated_frame, processed_img])
    cv2.imshow("Privacy Cam - Live View + Protected Image", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
