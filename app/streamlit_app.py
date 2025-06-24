import streamlit as st
from PIL import Image
import os
import sys
from io import BytesIO

# Make sure the `src` folder is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_image  # or predict_image if that's the actual function name

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection from Leaf Image")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Read once and store in memory
    img_bytes = uploaded_file.read()

    try:
        # ✅ Display image from memory buffer
        image = Image.open(BytesIO(img_bytes))
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        # ✅ Save image to disk for model prediction
        with open("temp.jpg", "wb") as f:
            f.write(img_bytes)

        with st.spinner("🔍 Analyzing..."):
            label = predict_image("temp.jpg")  # or predict_image("temp.jpg")
            os.remove("temp.jpg")

        st.success(f"✅ Predicted Disease: **{label}**")

    except Exception as e:
        st.error(f"❌ Failed to process the image: {e}")
