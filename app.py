import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import cv2
import os
import urllib.request
from PIL import Image
import io

# ----- Định nghĩa custom loss function -----
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = tf.keras.backend.sum(y_true) + smooth
    intersection = tf.keras.backend.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)
    return 1 - weight * dice

# ----- Hàm tải model từ Google Drive nếu chưa có -----
@st.cache_resource
def load_segmentation_model():
    model_path = "my_trained_model.keras"
    file_id = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    # Tải model nếu chưa có
    if not os.path.exists(model_path):
        with st.spinner("Đang tải model từ Google Drive..."):
            urllib.request.urlretrieve(url, model_path)
            st.success("✅ Tải model thành công!")

    model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    return model

# ----- Load model -----
model = load_segmentation_model()

# ----- Giao diện Streamlit -----
st.title("🧠 Brain Tumor Segmentation App")
st.markdown("Tải ảnh MRI lên để phân đoạn khối u.")

uploaded_file = st.file_uploader("📤 Tải ảnh MRI (.jpg hoặc .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh bằng PIL và chuyển sang OpenCV
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Resize ảnh
    image_resized = cv2.resize(image_cv2, (256, 256))
    input_image = image_resized / 255.0
    input_image = np.expand_dims(input_image, axis=0)  # shape: (1, 256, 256, 3)

    # Dự đoán mask
    pred_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # mask nhị phân

    # Hiển thị ảnh gốc và mask
    st.subheader("📷 Ảnh gốc:")
    st.image(image_pil, use_column_width=True)

    st.subheader("🧪 Mask dự đoán:")
    st.image(binary_mask, use_column_width=True, clamp=True, channels="GRAY")

    # Tùy chọn overlay
    st.subheader("🧩 Overlay ảnh + mask:")
    overlay = cv2.addWeighted(image_resized, 0.7, cv2.merge([binary_mask]*3), 0.3, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    st.image(overlay_rgb, use_column_width=True)

# Hiển thị phiên bản Python
import sys
st.caption(f"⚙️ Python version: {sys.version}")
