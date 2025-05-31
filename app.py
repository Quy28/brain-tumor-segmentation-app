import os
import gdown
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

# ----- Định nghĩa custom loss function -----
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred):
    smooth = 1e-6
    weight = K.sum(y_true) + smooth
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1 - weight * dice

@st.cache_resource
def load_segmentation_model():
    model_path = "my_trained_model.keras"
    if not os.path.exists(model_path):
        st.info("Đang tải model từ Google Drive, vui lòng chờ...")
        url = "https://drive.google.com/uc?id=1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
        gdown.download(url, model_path, quiet=False)
        st.success("Tải model xong!")
    model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    return model

# Load model 1 lần khi app khởi chạy
model = load_segmentation_model()

st.title("Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Upload ảnh MRI (JPG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Đọc ảnh từ file upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không thể đọc ảnh, vui lòng thử lại ảnh khác.")
    else:
        # Resize về (256, 256)
        image_resized = cv2.resize(image, (256, 256))
        input_image = image_resized / 255.0
        input_image = np.expand_dims(input_image, axis=0)  # (1, 256, 256, 3)

        # Dự đoán mask
        pred_mask = model.predict(input_image)[0, :, :, 0]
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Hiển thị ảnh và mask bằng Streamlit
        st.subheader("Ảnh gốc")
        st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("Mask dự đoán")
        st.image(binary_mask, clamp=True, channels="L", use_column_width=True)
