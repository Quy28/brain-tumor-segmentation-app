import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model
import gdown
import os

st.set_option('deprecation.showPyplotGlobalUse', False)

# ----- Định nghĩa custom loss function -----
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = K.sum(y_true) + smooth
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - weight * dice

# ----- Tải model từ Google Drive -----
@st.cache_resource
def load_segmentation_model():
    model_path = "my_trained_model.keras"
    if not os.path.exists(model_path):
        file_id = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    return model

model = load_segmentation_model()

# ----- Giao diện người dùng -----
st.title("Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Tải lên ảnh MRI", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Không thể đọc ảnh.")
    else:
        image_resized = cv2.resize(image, (256, 256))
        input_image = image_resized / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        pred_mask = model.predict(input_image)[0, :, :, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        # ----- Hiển thị -----
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Ảnh gốc")
        plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Mask dự đoán")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        st.pyplot()
