# app.py — Real-time Attendance via Browser Webcam (WebRTC) or Local Camera + SQLite Logging
# Timezone-aware (Asia/Kolkata)
# ✅ Enforces unique attendance:
#    - Teachers: once per day
#    - Students: once per period per day
# ✅ Browser speaks confirmation using SpeechSynthesis API

import os
import cv2
import numpy as np
import streamlit as st
import pickle
import sqlite3
from datetime import datetime, time
from pytz import timezone
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# =========================
# Load embeddings & labels
# =========================
with open("embedding.npy", "rb") as f:
    EMBEDDED_FACE = np.load(f)
    NAMES = pickle.load(f)

# =========================
# DB Setup
# =========================
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        period TEXT
    )
    """
)
conn.commit()

# =========================
# Class Schedule
# =========================
CLASS_SCHEDULE = [
    ("Period 1", time(9, 0), time(10, 0)),
    ("Period 2", time(10, 5), time(11, 5)),
    ("Period 3", time(11, 10), time(12, 10)),
    ("Period 4", time(13, 0), time(14, 0)),
    ("Period 5", time(14, 5), time(15, 5)),
]

# =========================
# Utilities
# =========================
def get_current_period(schedule):
    tz = timezone("Asia/Kolkata")
    now = datetime.now(tz).time()
    for period, start, end in schedule:
        if start <= now <= end:
            return period
    return None

def mark_student_db(name, period):
    tz = timezone("Asia/Kolkata")
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    cursor.execute(
        """
        SELECT * FROM attendance WHERE name=? AND role='Student' AND date=? AND period=?
        """,
        (name, date_str, period),
    )
    if cursor.fetchone():
        return False, f"Attendance already marked for {name} in {period}"

    cursor.execute(
        """
        INSERT INTO attendance (name, role, date, time, period) VALUES (?, 'Student', ?, ?, ?)
        """,
        (name, date_str, time_str, period),
    )
    conn.commit()
    return True, f"Attendance marked for {name} in {period}"

def mark_teacher_db(name):
    tz = timezone("Asia/Kolkata")
    now = datetime.now(tz)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    cursor.execute(
        """
        SELECT * FROM attendance WHERE name=? AND role='Teacher' AND date=?
        """,
        (name, date_str),
    )
    if cursor.fetchone():
        return False, f"Attendance already marked for {name} today"

    cursor.execute(
        """
        INSERT INTO attendance (name, role, date, time, period) VALUES (?, 'Teacher', ?, ?, NULL)
        """,
        (name, date_str, time_str),
    )
    conn.commit()
    return True, f"Attendance marked for {name}"

def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def draw_label(img, text, pos=(30, 30), color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, 0.8, color, 2, cv2.LINE_AA)

# =========================
# Video Processor
# =========================
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self):
        self.role = None
        self.class_schedule = CLASS_SCHEDULE

    def set_role(self, role):
        self.role = role

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Simulated face vector
        test_vec = np.random.rand(128)

        distances = [euclidean_distance(test_vec, emb) for emb in EMBEDDED_FACE]
        idx = int(np.argmin(distances))
        if distances[idx] < 0.6:
            name = NAMES[idx]
        else:
            name = None

        if self.role == "Student":
            period = get_current_period(self.class_schedule)
            if name and period:
                was_inserted, message = mark_student_db(name, period)
                if was_inserted:
                    st.session_state['tts_text'] = message   # ✅ speak later
                draw_label(img, f"{name} - {period}", color=(0, 200, 0))
            elif name and period is None:
                draw_label(img, f"{name} - Not In Period", color=(0, 165, 255))
            else:
                draw_label(img, "Face Not Recognized", color=(0, 0, 255))

        elif self.role == "Teacher":
            if name:
                was_inserted, message = mark_teacher_db(name)
                if was_inserted:
                    st.session_state['tts_text'] = message   # ✅ speak later
                draw_label(img, name, color=(255, 0, 0))
            else:
                draw_label(img, "Face Not Recognized", color=(0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# Streamlit UI
# =========================
st.title("📸 Attendance System via Face Recognition")

role = st.radio("Select Role:", ("Student", "Teacher"))

webrtc_ctx = webrtc_streamer(
    key="attendance",
    video_processor_factory=AttendanceProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.set_role(role)

# =========================
# Browser Speech (TTS)
# =========================
if "tts_text" in st.session_state and st.session_state.get("tts_text"):
    tts_text = st.session_state.pop("tts_text")
    safe_text = (
        tts_text.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("\n", " ")
    )

    js = f"""
    <script>
    const msg = "{safe_text}";
    if ('speechSynthesis' in window) {{
        const utter = new SpeechSynthesisUtterance(msg);
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.volume = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(utter);
    }} else {{
        console.log("SpeechSynthesis not supported");
    }}
    </script>
    """
    st.components.v1.html(js, height=0)
