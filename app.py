import streamlit as st
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
import pytz
import os

# =========================
# Database Setup
# =========================
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  date TEXT,
                  time TEXT)''')
    conn.commit()
    conn.close()

def mark_attendance(name):
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()

    # Timezone Asia/Kolkata
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    conn.close()

# =========================
# Load Face Embeddings
# =========================
if os.path.exists("embeddings.npy"):
    data = np.load("embeddings.npy", allow_pickle=True).item()
    known_encodings, known_names = data["encodings"], data["names"]
else:
    known_encodings, known_names = [], []

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")
st.title("📸 Automated Attendance System via Face Recognition")

# Initialize DB
init_db()

# Tabs for navigation
tabs = st.tabs(["🎥 Live Recognition", "📊 Attendance Records"])

# =========================
# Tab 1: Live Recognition
# =========================
with tabs[0]:
    st.subheader("Start Camera for Attendance")

    start = st.button("▶ Start Camera")
    stop = st.button("⏹ Stop Camera")

    frame_placeholder = st.empty()
    status = st.empty()

    if "run" not in st.session_state:
        st.session_state.run = False

    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    camera = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = camera.read()
        if not ret:
            status.error("⚠ Could not access camera")
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
                mark_attendance(name)

            # Draw rectangle + name
            top, right, bottom, left = [v * 4 for v in face_location]  # scale back
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        # Show in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()

# =========================
# Tab 2: Attendance Records
# =========================
with tabs[1]:
    st.subheader("Attendance Records (SQLite Database)")

    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT * FROM attendance ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ID", "Name", "Date", "Time"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No attendance records found yet.")
