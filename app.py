import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import gdown
import zipfile
import os

# ----- Custom loss function -----
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = K.sum(y_true) + smooth
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - weight * dice

# ----- Hàm tải model từ Google Drive và giải nén -----
@st.cache_resource(show_spinner=True)
def download_and_load_model():
    model_dir = "model_dir"
    model_path = os.path.join(model_dir, "my_trained_model.keras")
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1UXeSEpFDEBbn0bV7OZ8jEtieitA8mjOx"
        zip_path = "model.zip"
        # Tải zip về
        gdown.download(url, zip_path, quiet=False)
        # Giải nén
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(zip_path)
    # Load model với custom loss
    model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    return model

model = download_and_load_model()

st.title("Brain Tumor Segmentation App")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((256, 256))
    img_array = np.array(image_resized) / 255.0
    input_image = np.expand_dims(img_array, axis=0)

    # Dự đoán mask
    pred_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Hiển thị ảnh
    st.subheader("Ảnh gốc")
    st.image(image_resized)

    st.subheader("Mask dự đoán")
    st.image(binary_mask, clamp=True)
