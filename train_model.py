import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
train_data = pd.read_csv("Dataset/fashion-mnist_train.csv")
test_data = pd.read_csv("Dataset/fashion-mnist_test.csv")

# Separate features and labels
X_train = train_data.drop("label", axis=1).values
y_train = train_data["label"].values

X_test = test_data.drop("label", axis=1).values
y_test = test_data["label"].values

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape images
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Validation split
X_train, X_validate, y_train, y_validate = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)

# CNN architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(28, 28, 1)
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_validate, y_validate)
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)

# Classification report
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print(classification_report(y_test, predicted_classes))

# Save model
os.makedirs("model", exist_ok=True)

model.save("model/fashion_cnn.keras")

print("Model saved successfully!")