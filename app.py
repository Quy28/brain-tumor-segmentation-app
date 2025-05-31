import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
import requests
from pathlib import Path

# Khai báo custom loss
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred, weight=0.7):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return 1 - dice * weight

# Tải model nếu chưa tồn tại
def download_model(model_path, file_id):
    if not Path(model_path).exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        r = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(r.content)

# Load model
@st.cache_resource
def load_model():
    model_path = "my_trained_model.keras"
    file_id = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
    download_model(model_path, file_id)
    return tf.keras.models.load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})

model = load_model()

# Streamlit UI
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload an MRI image. The model will predict the tumor segmentation mask.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    img = np.array(image.resize((256, 256))) / 255.0
    input_tensor = np.expand_dims(img, axis=0)

    with st.spinner("Predicting..."):
        pred_mask = model.predict(input_tensor)[0]

    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    st.image(mask_rgb, caption="Predicted Tumor Mask", use_column_width=True)
