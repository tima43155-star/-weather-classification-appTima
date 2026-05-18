import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "weather_classifier_small (1).h5"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Cloudy", "Rain", "Shine", "Sunrise"]

st.title("Weather Image Classification")
st.write("Upload a weather image and AI will predict the weather condition.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")
