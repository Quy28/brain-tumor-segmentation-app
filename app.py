import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image
import requests
import os

# ------------------------
# Custom loss function
# ------------------------
@register_keras_serializable()
def weighted_dice_loss(y_true, y_pred, weight=0.5, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)) * weight

# ------------------------
# Load model (download if needed)
# ------------------------
model_path = "my_trained_model.keras"
file_id = "1FvLjJAjCO85Cwmj2IENVPAPhTF-Fnkn_"
url = f"https://drive.google.com/uc?id={file_id}"

def download_model():
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            response = requests.get(url)
            f.write(response.content)

@st.cache_resource
def load_segmentation_model():
    download_model()
    return load_model(model_path, custom_objects={'weighted_dice_loss': weighted_dice_loss})

# ------------------------
# Image preprocessing
# ------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# ------------------------
# Streamlit UI
# ------------------------
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload an MRI image to predict the tumor segmentation mask.")

uploaded_file = st.file_uploader("Upload MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original MRI Image", use_column_width=True)

    with st.spinner("Loading model and making prediction..."):
        model = load_segmentation_model()
        input_image = preprocess_image(image)
        pred_mask = model.predict(input_image)[0, :, :, 0]
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binarize

        # Convert to 3-channel for visualization
        mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
        mask_overlay = cv2.addWeighted(np.array(image.resize((256, 256))), 0.6, mask_rgb, 0.4, 0)

    st.image(mask_overlay, caption="Predicted Mask Overlay", use_column_width=True)
