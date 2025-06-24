import streamlit as st
from PIL import Image
import os
import sys
from io import BytesIO

# ✅ Ensure `src` is in the path (cross-platform)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from src.predict import predict_image  # assumes predict_image returns (label, confidence)

# ✅ Streamlit App Settings
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 LeafScan: Plant Disease Detection Using Deep Learning")

# ✅ File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # ✅ Read image from memory
        img_bytes = uploaded_file.read()
        image = Image.open(BytesIO(img_bytes))
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        # ✅ Save temp file
        temp_path = os.path.join(BASE_DIR, "temp_uploaded_leaf.jpg")
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # ✅ Run prediction
        with st.spinner("🔍 Analyzing..."):
            label, confidence = predict_image(temp_path)
            os.remove(temp_path)  # cleanup

        # ✅ Show result
        st.success(f"✅ Predicted Disease: **{label}**")
        st.write(f"🔬 Confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"❌ Failed to process the image: {e}")
