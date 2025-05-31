import os
import gdown
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable

@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = K.sum(y_true) + smooth
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - weight * dice

@st.cache_resource(show_spinner=True)
def load_segmentation_model():
    model_path = "my_trained_model.keras"
    if not os.path.exists(model_path):
        st.info("Đang tải model từ Google Drive, vui lòng chờ...")
        url = "https://drive.google.com/uc?id=1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
        gdown.download(url, model_path, quiet=False)
        st.success("Tải model xong!")
    model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    return model

model = load_segmentation_model()

# Cho phép user upload ảnh để test
uploaded_file = st.file_uploader("Upload ảnh để thử segmentation", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không thể đọc ảnh, vui lòng thử lại với file khác.")
    else:
        image_resized = cv2.resize(image, (256, 256))
        input_image = image_resized / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        pred_mask = model.predict(input_image)[0, :, :, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8)

        st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), caption="Ảnh gốc", use_column_width=True)
        st.image(binary_mask * 255, caption="Mask dự đoán", use_column_width=True)
