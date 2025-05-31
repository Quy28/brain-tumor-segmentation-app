import os
import streamlit as st
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

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
        import gdown
        url = "https://drive.google.com/uc?id=1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
        gdown.download(url, model_path, quiet=False)
        st.success("Tải model xong!")
    else:
        st.write(f"Model đã tồn tại: {model_path}")

    if not os.path.exists(model_path):
        st.error("Model file không tồn tại sau khi tải!")
        return None

    try:
        model = load_model(model_path, custom_objects={"weighted_dice_loss": weighted_dice_loss})
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None
    return model

model = load_segmentation_model()
