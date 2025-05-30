import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# ========== CONFIG ==========
MODEL_PATH = "my_trained_model.keras"
FILE_ID = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# ========== CUSTOM LOSS / METRIC PLACEHOLDER ==========
# Nếu model dùng custom loss, bạn cần định nghĩa nó để load
# Dưới đây là ví dụ với dice_loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)

custom_objects = {
    "dice_loss": dice_loss
}

# ========== DOWNLOAD MODEL IF NEEDED ==========
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            response = requests.get(GDRIVE_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

model = load_model()

# ========== STREAMLIT UI ==========
st.title("🧠 Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Preprocessing
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    input_array = np.expand_dims(img_array, axis=0)

    # Predict mask
    prediction = model.predict(input_array)
    predicted_mask = np.squeeze(prediction)

    # Convert mask to image (grayscale)
    mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))

    st.subheader("Predicted Mask")
    st.image(mask_image, use_column_width=True, caption="Tumor Segmentation Output")
