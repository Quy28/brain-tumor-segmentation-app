import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import gdown
from PIL import Image

# Tải model từ Google Drive nếu chưa có
model_path = "my_trained_model.keras"
file_id = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    with st.spinner("Đang tải model..."):
        gdown.download(url, model_path, quiet=False)

# Định nghĩa hàm loss
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = K.sum(y_true) + smooth
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - weight * dice

# Load model
model = tf.keras.models.load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})

# Giao diện người dùng
st.title("🧠 Brain Tumor Segmentation App")
st.write("Tải lên ảnh MRI và xem kết quả phân đoạn khối u")

uploaded_file = st.file_uploader("📤 Tải ảnh MRI (.jpg, .png)", type=["jpg", "png"])

if uploaded_file:
    # Đọc ảnh bằng OpenCV từ dữ liệu tải lên
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Ảnh gốc", use_column_width=True)

    # Xử lý ảnh
    image_resized = cv2.resize(image, (256, 256))
    input_image = image_resized / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Dự đoán mask
    pred_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Hiển thị mask
    st.image(binary_mask, caption="🎯 Mask dự đoán", use_column_width=True, clamp=True)

    # Overlay mask lên ảnh gốc
    overlay = cv2.addWeighted(image_resized, 0.6, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.4, 0)
    st.image(overlay, caption="🩻 Ảnh với Mask Overlay", use_column_width=True)
