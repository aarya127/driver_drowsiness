import cv2
import numpy as np

# Path to the Haar cascade file for eye detection
cascPath = '/Users/aaryas127/driverDrowsiness/haarcascade_eye.xml'
eyeCascade = cv2.CascadeClassifier(cascPath)

# Start video capture from the default camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# List to store eye ROIs
eye_features = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract region of interest (ROI) for each eye
        eye_roi = frame[y:y+h, x:x+w]

        # Example: Convert ROI to grayscale
        eye_roi_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # Example: Resize ROI to a fixed size (if needed)
        eye_roi_resized = cv2.resize(eye_roi_gray, (100, 50))

        # Append the processed eye_roi to the eye_features list
        eye_features.append(eye_roi_resized)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# Convert the list of eye features to numpy array for easier handling
eye_features = np.array(eye_features)

# Save the eye features to a file (e.g., numpy .npy file)
np.save('/Users/aaryas127/driverDrowsiness/eye_features.npy', eye_features)

print("Eye features saved successfully.")
