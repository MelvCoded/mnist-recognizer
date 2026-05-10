# =============================================================================
# train_model.py
# =============================================================================
# This script:
#   1. Loads the MNIST dataset (built into Keras)
#   2. Preprocesses the images
#   3. Builds a Convolutional Neural Network (CNN)
#   4. Trains the model
#   5. Evaluates accuracy and loss on the test set
#   6. Saves the trained model to the /model folder
#
# Run this once before launching the app to create the model file:
#   python train_model.py
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Fix SSL certificate issue on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------------------------------------------------------
# STEP 1: Load the MNIST Dataset
# -----------------------------------------------------------------------------
# MNIST contains:
#   - 60,000 training images of handwritten digits (0–9)
#   - 10,000 test images
# Each image is 28x28 pixels, grayscale (1 channel)

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")
print(f"  Image shape      : {X_train.shape[1:]} (height x width)")

# -----------------------------------------------------------------------------
# STEP 2: Preprocess the Data
# -----------------------------------------------------------------------------
# CNNs expect input shape: (samples, height, width, channels)
# MNIST is grayscale so channels = 1

# Reshape: add the channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)   # shape: (60000, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)    # shape: (10000, 28, 28, 1)

# Normalize pixel values from [0, 255] -> [0.0, 1.0]
# This helps the model learn faster and more stably
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# One-hot encode labels
# Example: digit 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# This is needed because we have 10 output classes (0–9)
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test,  num_classes=10)

print("\nPreprocessing complete.")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

# -----------------------------------------------------------------------------
# STEP 3: Build the CNN Model
# -----------------------------------------------------------------------------
# Architecture:
#   Conv2D -> MaxPooling -> Conv2D -> MaxPooling -> Flatten -> Dense -> Output
#
# Why this architecture?
#   - Conv layers detect patterns (edges, curves, loops in digits)
#   - Pooling layers reduce spatial size, keeping important features
#   - Dense layers combine features to make a final decision

def build_cnn_model():
    model = models.Sequential([

        # --- Block 1 ---
        # 32 filters, each 3x3. 'relu' activation ignores negative values.
        # 'same' padding keeps output the same size as input.
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=(28, 28, 1), name='conv1'),

        # Reduces each 2x2 region to its max value -> halves spatial dimensions
        layers.MaxPooling2D((2, 2), name='pool1'),

        # --- Block 2 ---
        # 64 filters — more filters to detect more complex features
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        # --- Block 3 ---
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),

        # --- Flatten ---
        # Convert 3D feature maps -> 1D vector before Dense layers
        layers.Flatten(name='flatten'),

        # --- Fully Connected Layers ---
        layers.Dense(128, activation='relu', name='dense1'),

        # Dropout randomly turns off 50% of neurons during training
        # This prevents overfitting (memorizing training data)
        layers.Dropout(0.5, name='dropout'),

        # Output layer: 10 neurons (one per digit 0–9)
        # Softmax converts raw scores -> probabilities that sum to 1
        layers.Dense(10, activation='softmax', name='output')
    ])
    return model

model = build_cnn_model()

# Print a summary of the model layers and parameter counts
model.summary()

# -----------------------------------------------------------------------------
# STEP 4: Compile the Model
# -----------------------------------------------------------------------------
# optimizer: 'adam' adjusts learning rate automatically — good default
# loss: 'categorical_crossentropy' is standard for multi-class classification
# metrics: we track 'accuracy' during training

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------------------------
# STEP 5: Train the Model
# -----------------------------------------------------------------------------
# epochs: how many full passes through the training data
# batch_size: how many images are processed before updating weights
# validation_split: 10% of training data used to monitor overfitting

print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# -----------------------------------------------------------------------------
# STEP 6: Evaluate on Test Set
# -----------------------------------------------------------------------------
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*40}")
print(f"  Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"{'='*40}")

# -----------------------------------------------------------------------------
# STEP 7: Plot Training History (optional visualization)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Save the plot
os.makedirs("model", exist_ok=True)
plt.savefig("model/training_history.png", dpi=150)
print("\nTraining history plot saved to model/training_history.png")

# -----------------------------------------------------------------------------
# STEP 8: Save the Trained Model
# -----------------------------------------------------------------------------
# We save the model so the web app can load it without retraining every time

model_path = "model/mnist_cnn.h5"
model.save(model_path)
print(f"Model saved to {model_path}")