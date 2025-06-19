import cv2
import os
import time
import streamlit as st
import numpy as np
import re

# Title of the Streamlit app
st.title("📸 Face Data Capture (Local Only)")

# Input for the user's name
raw_name = st.text_input("Enter your full name (letters and spaces only):").strip()

def sanitize_name(name):
    """Sanitize the name to ensure safe folder naming"""
    name = re.sub(r"[^a-zA-Z\s]", "", name)  # Remove special characters
    name = re.sub(r"\s+", "_", name)         # Replace spaces with underscores
    return name

# Processed user name
user_name = sanitize_name(raw_name)

# Image capture configuration
max_images = st.slider("Number of images to capture:", 10, 50, 30)
capture_btn = st.button("Start Capture")

# Placeholder for showing the webcam feed
frame_placeholder = st.empty()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def enhance_image(img):
    """Apply contrast & sharpening to improve low-res webcam quality"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    sharpened = cv2.convertScaleAbs(equalized, alpha=1.5, beta=0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# Handle the image capture logic
if capture_btn and user_name:
    local_path = os.path.join("dataset", user_name)
    os.makedirs(local_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    st.info("🔄 Initializing camera...")
    time.sleep(3)

    count = 0
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠ Webcam not accessible.")
            break

        frame = enhance_image(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

        if len(faces) != 1:
            annotated = frame.copy()
            msg = "❌ No face" if len(faces) == 0 else "❌ Multiple faces"
            cv2.putText(annotated, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            continue

        (x, y, w, h) = faces[0]
        if w < 60 or h < 60:
            cv2.putText(frame, "⬆ Move closer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            continue

        # Save face
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
        filename = f"{count}.jpg"
        cv2.imwrite(os.path.join(local_path, filename), face_resized)

        # Show capture status
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, f"✅ Saved {count+1}/{max_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        count += 1
        time.sleep(3)

    cap.release()
    st.success(f"✅ Capture completed! {count} face images saved to: dataset/{user_name}/")
else:
    st.warning("⚠ Please enter a valid name and click Start Capture.")