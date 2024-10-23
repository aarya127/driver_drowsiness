import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load eye features and labels
eye_features = np.load('/Users/aaryas127/driverDrowsiness/eye_features.npy')
labels = np.load('/Users/aaryas127/driverDrowsiness/labels.npy')  # Assuming you have labels indicating drowsiness state

# Normalize eye features (adjust as needed)
eye_features = eye_features / 255.0
eye_features = eye_features.reshape(-1, 50, 100, 1)  # Ensure shape matches input to the model

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(eye_features, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Define CNN model (with Dropout for regularization)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 100, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Dropout layer to prevent overfitting

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Increase dropout in fully connected layers
    Dense(1, activation='sigmoid')  # Assuming binary classification (awake/drowsy)
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks: EarlyStopping and ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train model with data augmentation and validation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Save the model
model.save('/Users/aaryas127/driverDrowsiness/fineTunedDrowsinessCnnModel.h5')
