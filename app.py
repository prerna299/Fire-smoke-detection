import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f5f5;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ff4b4b;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = joblib.load("fire_smoke_detector.pkl")

# ---------------------------
# HEADER
# ---------------------------

st.markdown(
    '<p class="title">🔥 Fire & Smoke Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Detection using Random Forest</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.success("🟢 System Active")

st.sidebar.info("""
Algorithms Used:

✅ KNN

✅ Decision Tree

✅ Random Forest

Final Selected Model:

🏆 Random Forest
""")

# ---------------------------
# IMAGE SOURCE
# ---------------------------

option = st.radio(
    "Select Input Method",
    ["Upload Image", "Capture Image"]
)

image = None

if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "📤 Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

else:

    camera_image = st.camera_input(
        "📷 Capture Image"
    )

    if camera_image is not None:
        image = Image.open(camera_image)

# ---------------------------
# PREDICTION
# ---------------------------

if image is not None:

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Input Image",
            use_container_width=True
        )

    with col2:

        img = np.array(image)

        img = cv2.resize(img, (64, 64))

        img = img.flatten().reshape(1, -1)

        prediction = model.predict(img)

        st.subheader("Detection Result")

        if prediction[0] == 0:

            result = "FIRE"

            st.error("🔥 FIRE DETECTED")

        elif prediction[0] == 1:

            result = "NORMAL"

            st.success("✅ NORMAL ENVIRONMENT")

        else:

            result = "SMOKE"

            st.warning("💨 SMOKE DETECTED")

        st.info("Model Used: Random Forest")

        # -----------------------
        # SAVE HISTORY
        # -----------------------

        record = pd.DataFrame({
            "Time": [datetime.now()],
            "Result": [result]
        })

        if os.path.exists("history.csv"):

            history = pd.read_csv("history.csv")

            history = pd.concat(
                [history, record],
                ignore_index=True
            )

        else:

            history = record

        history.to_csv(
            "history.csv",
            index=False
        )

        # -----------------------
        # DOWNLOAD REPORT
        # -----------------------

        report = f"""
Fire & Smoke Detection Report

Time: {datetime.now()}

Prediction: {result}

Model Used: Random Forest
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="Detection_Report.txt"
        )

# ---------------------------
# HISTORY
# ---------------------------

st.divider()

st.subheader("📜 Recent Detection History")

if os.path.exists("history.csv"):

    history = pd.read_csv("history.csv")

    st.dataframe(
        history.tail(10),
        use_container_width=True
    )

# ---------------------------
# FOOTER
# ---------------------------

st.divider()

st.markdown("""
### About Project

This project detects:

- 🔥 Fire
- 💨 Smoke
- ✅ Normal Environment

The system was trained using multiple Machine Learning algorithms:

- KNN
- Decision Tree
- Random Forest

After comparing Accuracy, Precision and Recall,
Random Forest was selected as the final model.
""")