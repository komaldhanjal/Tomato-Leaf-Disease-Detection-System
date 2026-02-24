import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

MODEL_PATH = "leaf_disease_cnn_model.h5"
IMG_SIZE = 224

CLASS_NAMES = [
    "Healthy",
    "Early Blight",
    "Late Blight"
]

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------ UI ------------------
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a leaf image and predict the disease")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Preprocess
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100

        st.success(f"🦠 Disease: **{CLASS_NAMES[class_index]}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
