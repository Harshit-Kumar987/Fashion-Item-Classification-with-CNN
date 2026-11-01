
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fashion_cnn.keras")

model = load_model()

# Fashion MNIST class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

st.title("Fashion Item Classification")
st.write("Upload a clothing image to predict its category.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


def preprocess_image(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Resize while preserving aspect ratio
    image.thumbnail((22, 22))

    # Create 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    # Center the clothing image
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Convert to numpy array
    image_array = np.array(canvas).astype("float32")

    # Normalize pixel values
    image_array = image_array / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    return canvas, image_array


if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(original_image, width=300)

    processed_image, image_array = preprocess_image(original_image)

    st.subheader("Processed Image")
    st.image(processed_image.resize((280, 280)), width=280)

    # Predict
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class] * 100

    st.subheader("Prediction")

    st.success(class_names[predicted_class])

    st.write(f"Confidence: {confidence:.2f}%")

    # Show probabilities
    st.subheader("Class Probabilities")

    for i, name in enumerate(class_names):
        st.write(f"{name}: {predictions[i] * 100:.2f}%")
