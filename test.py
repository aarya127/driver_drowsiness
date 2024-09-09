import numpy as np

# Load eye features from file
eye_features = np.load('/Users/aaryas127/driverDrowsiness/eye_features.npy')

# Print shape and data type
print("Shape of eye_features array:", eye_features.shape)
print("Data type of elements:", eye_features.dtype)

# Example of accessing a specific eye ROI (assuming at least one eye was detected)
if eye_features.shape[0] > 0:
    first_eye_roi = eye_features[0]
    print("Shape of the first eye ROI:", first_eye_roi.shape)

# Example of displaying the first eye ROI (assuming it's a grayscale image)
import cv2
cv2.imshow('First Eye ROI', first_eye_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
