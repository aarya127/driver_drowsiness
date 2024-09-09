import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load eye features and labels
eye_features = np.load('/Users/aaryas127/driverDrowsiness/eye_features.npy')
labels = np.load('/Users/aaryas127/driverDrowsiness/labels.npy')  # Assuming you have labels indicating drowsiness state

# Normalize eye features (example, adjust as needed)
eye_features = eye_features / 255.0

# Split data into training and testing sets (example, adjust as needed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(eye_features, labels, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification (awake/drowsy)
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Save the model
model.save('/Users/aaryas127/driverDrowsiness/drowsinessCnnModel.h5')
