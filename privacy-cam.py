import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import os

AUTHORIZED_FACE_FILE = "authorized_face.jpg"
display_width, display_height = 640, 480

# --- Load static image to protect ---
static_img = cv2.imread('privacy.png')
if static_img is None:
    raise FileNotFoundError("Image 'privacy.png' not found.")
static_img = cv2.resize(static_img, (display_width, display_height))

# --- Load YOLOv8n model (nano) ---
yolo_model = YOLO('yolov8n.pt')  # much faster on CPU

# --- Load authorized face encoding ---
authorized_encoding = None
if os.path.exists(AUTHORIZED_FACE_FILE):
    auth_img = face_recognition.load_image_file(AUTHORIZED_FACE_FILE)
    encs = face_recognition.face_encodings(auth_img)
    if encs:
        authorized_encoding = encs[0]
        print("✅ Authorized face loaded.")
    else:
        print("⚠ No face found in authorized_face.jpg")
else:
    print("⚠ No authorized face registered yet. Press 'r' to register one.")

# --- Webcam ---
cap = cv2.VideoCapture(0)
violating_objects = ['cell phone', 'camera', 'laptop', 'tv', 'monitor']
allow_classes = ['person'] + violating_objects

# --- Frame skipping parameters ---
FRAME_SKIP_YOLO = 3       # run YOLO every 3 frames
FRAME_SKIP_FACE = 5       # run face recognition every 5 frames
frame_counter = 0
yolo_results_cache = None
face_authenticated = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frame = cv2.resize(frame, (display_width, display_height))
    annotated_frame = frame.copy()
    
    # ---- YOLO detection (every FRAME_SKIP_YOLO frames) ----
    if frame_counter % FRAME_SKIP_YOLO == 0:
        yolo_results_cache = yolo_model(frame, verbose=False)[0]
    
    person_count = 0
    object_violation = False
    if yolo_results_cache:
        for box in yolo_results_cache.boxes:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id].lower()
            conf = float(box.conf[0])
            if conf < 0.4 or label not in allow_classes:
                continue
            if label == 'person':
                person_count += 1
            elif label in violating_objects:
                object_violation = True
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = (0, 0, 255) if label in violating_objects else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---- Face Authentication (every FRAME_SKIP_FACE frames) ----
    if frame_counter % FRAME_SKIP_FACE == 0 and authorized_encoding is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_authenticated = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([authorized_encoding], face_encoding, tolerance=0.45)
            if True in matches:
                face_authenticated = True
                break

    # ---- Privacy Logic ----
    privacy_ok = (person_count == 1 and not object_violation and face_authenticated)
    if privacy_ok:
        processed_img = static_img.copy()
        cv2.putText(processed_img, "✅ Authenticated - Privacy Maintained", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        processed_img = cv2.GaussianBlur(static_img, (25, 25), 0)  # smaller blur kernel
        warning_text = "⚠ Privacy Violation: "
        if person_count != 1:
            warning_text += f"{person_count} People. "
        if object_violation:
            warning_text += "Device Detected! "
        if authorized_encoding is None:
            warning_text += "No Registered Face!"
        elif not face_authenticated:
            warning_text += "Unauthorized Face!"
        cv2.putText(processed_img, warning_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(annotated_frame, "📷 Live Camera (press 'r' to register face)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    combined = cv2.hconcat([annotated_frame, processed_img])
    cv2.imshow("Privacy Cam - Live + Protected", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # --- Register current face ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)
        if encs:
            authorized_encoding = encs[0]
            cv2.imwrite(AUTHORIZED_FACE_FILE, frame)
            print("✅ Face registered and saved.")
        else:
            print("⚠ No face detected. Try again.")

cap.release()
cv2.destroyAllWindows()
