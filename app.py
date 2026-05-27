import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Load trained model
model = joblib.load('fire_smoke_detector.pkl')

# Title
st.title("🔥 Fire and Smoke Detection System")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (64, 64))

    # Flatten image
    img = img.flatten().reshape(1, -1)

    # Prediction
    prediction = model.predict(img)

    # Output
    if prediction[0] == 0:
        st.error("🔥 FIRE DETECTED")
    else:
        st.warning("💨 SMOKE DETECTED")