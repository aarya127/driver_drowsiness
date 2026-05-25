import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = '/Users/aaryas127/Documents/GitHub/driver_drowsiness/drowsinessCnnModel.h5'
DROWSY_THRESHOLD = 0.5
DROWSY_FRAME_LIMIT = 20  # consecutive frames before alerting

model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

drowsy_frames = 0

print("Drowsiness detection running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    eyes_detected = False
    prediction_label = "No Face"

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)

        # Restrict to upper 55% of face — eyes are never in the lower half
        upper_fh = int(fh * 0.55)
        roi_gray = gray[fy:fy + upper_fh, fx:fx + fw]
        roi_color = frame[fy:fy + upper_fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_roi = cv2.resize(eye_roi, (100, 50))
            eye_roi = eye_roi / 255.0
            eye_roi = eye_roi.reshape(1, 50, 100, 1)

            pred = model.predict(eye_roi, verbose=0)[0][0]

            if pred > DROWSY_THRESHOLD:
                drowsy_frames += 1
                prediction_label = f"DROWSY ({pred:.2f})"
                color = (0, 0, 255)
            else:
                drowsy_frames = max(0, drowsy_frames - 1)
                prediction_label = f"Awake ({pred:.2f})"
                color = (0, 255, 0)

            cv2.putText(roi_color, prediction_label, (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not eyes_detected:
        drowsy_frames = max(0, drowsy_frames - 1)

    if drowsy_frames >= DROWSY_FRAME_LIMIT:
        cv2.putText(frame, "ALERT: DROWSINESS DETECTED!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    status_text = f"Drowsy frames: {drowsy_frames}/{DROWSY_FRAME_LIMIT}"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
