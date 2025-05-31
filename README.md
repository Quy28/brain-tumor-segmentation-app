# 🧠 Brain Tumor Segmentation App

Streamlit app for segmenting brain tumors from MRI images using a trained U-Net model.

## ✅ Key Features
- Upload MRI image (PNG/JPG)
- Predict tumor mask with a U-Net model (.keras)
- Automatic download of model from Google Drive if not present

## 🔧 Environment

This app is built and tested on **Python 3.10.12**, the same as used on Kaggle.

Make sure Streamlit Cloud uses the correct Python version.

## 📦 Requirements

See `requirements.txt` for all dependencies:
- TensorFlow 2.17.1
- OpenCV (headless version)
- Streamlit 1.34.0
- Python 3.10.12 ✅
