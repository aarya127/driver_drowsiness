import cv2


# Path to the Haar cascade file
cascPath = '/Users/aaryas127/driverDrowsiness/haarcascade_frontalcatface.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Check if the cascade was loaded successfully
if faceCascade.empty():
    print("Error: Failed to load cascade classifier")
    exit()

# Start video capture from the default camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw nodes around the faces
        for (x, y, w, h) in faces:
            # Coordinates of the center of the face
            center_x, center_y = x + w // 2, y + h // 2

            # Coordinates of the corners of the face rectangle
            top_left = (x, y)
            top_right = (x + w, y)
            bottom_left = (x, y + h)
            bottom_right = (x + w, y + h)

            # List of nodes to draw
            nodes = [top_left, top_right, bottom_left, bottom_right, (center_x, center_y)]

            # Draw circles at each node
            for node in nodes:
                cv2.circle(frame, node, 5, (0, 255, 0), -1)

            # Print the coordinates
            print("Face detected at:")
            print(f"Top-left: {top_left}")
            print(f"Top-right: {top_right}")
            print(f"Bottom-left: {bottom_left}")
            print(f"Bottom-right: {bottom_right}")
            print(f"Center: ({center_x}, {center_y})")

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything is done, release the capture and destroy all windows
    video_capture.release()
    cv2.destroyAllWindows()
