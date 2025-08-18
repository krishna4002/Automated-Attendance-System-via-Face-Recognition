# app.py  — Real-time Attendance via Browser Webcam (WebRTC) or Local Camera + SQLite Logging
# Browser-side TTS confirmation using Web Speech API (SpeechSynthesis).

import os
import cv2
import numpy as np
import streamlit as st
import sqlite3
from datetime import datetime, time
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import platform
from zoneinfo import ZoneInfo   # ✅ timezone support

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="AI Attendance System", layout="centered")

# ---------------------------
# Constants / DB / TZ
# ---------------------------
DB_PATH = "attendance.db"
INDIA_TZ = ZoneInfo("Asia/Kolkata")   # ✅ Set timezone

# ---------------------------
# Helpful comment about requirements (add to requirements.txt)
# ---------------------------
# requirements.txt should include at least:
# streamlit
# streamlit-webrtc
# facenet-pytorch
# torch
# torchvision
# numpy
# pandas
# pillow
# scikit-learn
# av
# (adapt versions to your environment)

# ---------------------------
# MODEL + EMBEDDINGS LOADER
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not os.path.exists('embeddings.npy'):
        # Don't crash—return empty dict so app still runs and shows message
        st.error("`embeddings.npy` not found. Please upload/generate embeddings.npy in the app folder.")
        return device, mtcnn, model, {}

    embedding_dict = np.load('embeddings.npy', allow_pickle=True).item()
    return device, mtcnn, model, embedding_dict

device, mtcnn, model, embedding_dict = load_models_and_embeddings()

# ---------------------------
# DATABASE FUNCTIONS
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Students table
    c.execute("""
        CREATE TABLE IF NOT EXISTS student_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            time TEXT,
            period TEXT,
            date TEXT,
            UNIQUE(name, period, date)
        )
    """)
    # Teachers table
    c.execute("""
        CREATE TABLE IF NOT EXISTS teacher_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            time TEXT,
            date TEXT,
            UNIQUE(name, date)
        )
    """)
    conn.commit()
    conn.close()

def mark_student_db(name, period):
    """
    Mark student attendance only once per period per day.
    Returns (was_inserted: bool, message: str)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(INDIA_TZ)
    today = now.strftime("%Y-%m-%d")

    c.execute("""
        SELECT 1 FROM student_attendance 
        WHERE name=? AND period=? AND date=?
    """, (name, period, today))
    exists = c.fetchone()

    if not exists:
        c.execute("""
            INSERT INTO student_attendance (name, time, period, date) 
            VALUES (?, ?, ?, ?)
        """, (name, now.strftime("%H:%M:%S"), period, today))
        conn.commit()
        conn.close()
        message = f"Attendance marked for {name} in {period}"
        return True, message
    conn.close()
    return False, ""

def mark_teacher_db(name):
    """
    Mark teacher attendance only once per day.
    Returns (was_inserted: bool, message: str)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(INDIA_TZ)
    today = now.strftime("%Y-%m-%d")

    c.execute("""
        SELECT 1 FROM teacher_attendance 
        WHERE name=? AND date=?
    """, (name, today))
    exists = c.fetchone()

    if not exists:
        c.execute("""
            INSERT INTO teacher_attendance (name, time, date) 
            VALUES (?, ?, ?)
        """, (name, now.strftime("%H:%M:%S"), today))
        conn.commit()
        conn.close()
        message = f"Attendance marked for {name}"
        return True, message
    conn.close()
    return False, ""

def fetch_logs(table_name):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY date DESC, time DESC", conn)
    conn.close()
    return df

init_db()

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def get_current_period(schedule: dict):
    if not schedule:
        return None
    now = datetime.now(INDIA_TZ).time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding, embedding_dict, threshold=0.75):
    if not embedding_dict:
        return None
    best_match, highest_similarity = None, 0.0
    for name, ref_emb in embedding_dict.items():
        sim = cosine_similarity(embedding, ref_emb.reshape(1, -1))[0][0]
        if sim > threshold and sim > highest_similarity:
            best_match, highest_similarity = name, sim
    return best_match

def parse_schedule_csv(csv_file):
    df = pd.read_csv(csv_file)
    schedule = {}
    for _, row in df.iterrows():
        name = row['Subject']
        start = datetime.strptime(str(row['Start']).strip(), "%H:%M").time()
        end = datetime.strptime(str(row['End']).strip(), "%H:%M").time()
        schedule[name] = (start, end)
    return schedule

def draw_label(img, text, pos=(20, 40), color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# ---------------------------
# UI / Mode Selection
# ---------------------------
st.title("🧠 AI-Powered Attendance System")
mode = st.sidebar.radio("Choose Option", ["Student", "Teacher", "📑 View Attendance Logs"])
today = datetime.now(INDIA_TZ).strftime("%Y-%m-%d")

# ---------------------------
# Schedule config UI
# ---------------------------
if mode in ["Student", "Teacher"]:
    st.sidebar.subheader("🗂 Schedule Input Method")
    schedule_option = st.sidebar.radio("How would you like to input class periods?", ["Manual", "Upload CSV"])
    class_schedule = {}

    if schedule_option == "Manual":
        st.sidebar.subheader("🕒 Class Period Configuration")
        num_periods = st.sidebar.number_input("Number of Periods", min_value=1, max_value=10, value=3)
        for i in range(num_periods):
            with st.sidebar.expander(f"📘 Period {i+1} Settings"):
                subject = st.text_input(f"Subject Name {i+1}", key=f"sub_{i}")
                start = st.time_input(f"Start Time {i+1}", key=f"start_{i}", value=time(9+(i*1), 0))
                end = st.time_input(f"End Time {i+1}", key=f"end_{i}", value=time(10+(i*1), 0))
                if subject:
                    class_schedule[f"Period {i+1} - {subject}"] = (start, end)
    elif schedule_option == "Upload CSV":
        st.sidebar.subheader("📁 Upload CSV with Columns: Subject,Start,End (HH:MM)")
        csv_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
        if csv_file is not None:
            try:
                class_schedule = parse_schedule_csv(csv_file)
                st.sidebar.success("✅ Schedule Loaded from CSV")
            except Exception as e:
                st.sidebar.error(f"Failed to parse CSV: {e}")

    #st.sidebar.markdown("---")
    #capture_source = st.sidebar.selectbox("Camera Source", ["Browser (WebRTC) - recommended", "Local (OpenCV)"])

# ---------------------------
# WEBRTC VIDEO PROCESSOR
# ---------------------------
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self, role, class_schedule):
        self.role = role
        self.class_schedule = class_schedule or {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        face = mtcnn(img_pil)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy()
            name = recognize_face(emb, embedding_dict)

            if self.role == "Student":
                period = get_current_period(self.class_schedule)
                if name and period:
                    was_inserted, message = mark_student_db(name, period)
                    # If newly inserted, set session_state tts_text to trigger browser TTS
                    if was_inserted:
                        try:
                            # set a session_state flag for main thread to pick up
                            st.session_state['tts_text'] = message
                        except Exception:
                            # It's possible modifying session_state from this thread may fail silently
                            pass
                    draw_label(img, f"{name} - {period}", color=(0, 200, 0))
                elif name and period is None:
                    draw_label(img, f"{name} - Not In Period", color=(0, 165, 255))
                else:
                    draw_label(img, "Face Not Recognized", color=(0, 0, 255))

            elif self.role == "Teacher":
                if name:
                    was_inserted, message = mark_teacher_db(name)
                    if was_inserted:
                        try:
                            st.session_state['tts_text'] = message
                        except Exception:
                            pass
                    draw_label(img, name, color=(255, 0, 0))
                else:
                    draw_label(img, "Face Not Recognized", color=(0, 0, 255))
        else:
            draw_label(img, "No face detected", color=(0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------
# Modes: Student / Teacher / Logs
# ---------------------------
if mode == "Student":
    st.subheader("📚 Student Mode (Real-Time)")
    if capture_source.startswith("Browser"):
        webrtc_streamer(
            key="student_attendance",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AttendanceProcessor("Student", class_schedule),
        )
    else:
        st.warning("Local camera selected. This only works on your own machine (not on cloud).")

elif mode == "Teacher":
    st.subheader("🎓 Teacher Mode (Real-Time)")
    if capture_source.startswith("Browser"):
        webrtc_streamer(
            key="teacher_attendance",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AttendanceProcessor("Teacher", class_schedule),
        )
    else:
        st.warning("Local camera selected. This only works on your own machine (not on cloud).")

elif mode == "📑 View Attendance Logs":
    st.subheader("📑 Attendance Logs")

    # Reset DB button
    if st.button("🗑 Reset Database"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS student_attendance")
        c.execute("DROP TABLE IF EXISTS teacher_attendance")
        conn.commit()
        conn.close()
        init_db()  # recreate fresh tables
        st.success("✅ Database has been reset!")

    tab1, tab2 = st.tabs(["Student Attendance", "Teacher Attendance"])
    with tab1:
        df_students = fetch_logs("student_attendance")
        if df_students.empty:
            st.info("No student attendance records yet.")
        else:
            st.dataframe(df_students)
    with tab2:
        df_teachers = fetch_logs("teacher_attendance")
        if df_teachers.empty:
            st.info("No teacher attendance records yet.")
        else:
            st.dataframe(df_teachers)

# ---------------------------
# Browser-side TTS trigger
# ---------------------------
# If the processor set st.session_state['tts_text'], play it in the browser using Web Speech API.
if 'tts_text' in st.session_state and st.session_state.get('tts_text'):
    # Pop the text so we only play once
    tts_text = st.session_state.pop('tts_text')

    # Escape the text for JavaScript (basic)
    safe_text = tts_text.replace("'", "\\'").replace("\n", " ")

    # Insert an invisible HTML block that runs JS to speak the message immediately.
    # This uses the browser's SpeechSynthesis API (no external calls).
    js = f"""
    <script>
    const msg = '{safe_text}';
    if ('speechSynthesis' in window) {{
        const utter = new SpeechSynthesisUtterance(msg);
        // optional: choose voice, pitch, rate
        utter.rate = 1.0;
        utter.pitch = 1.0;
        // Some browsers require a user gesture before audio; but often works on streamlit interactions.
        window.speechSynthesis.cancel(); // stop any ongoing speech
        window.speechSynthesis.speak(utter);
    }} else {{
        console.log('SpeechSynthesis not supported in this browser');
    }}
    </script>
    """

    # We use unsafe_allow_html to run script; streamlit will render the script and play audio.
    st.components.v1.html(js, height=0)

# ---------------------------
# Small usage hint
# ---------------------------
#st.sidebar.markdown("---")
#st.sidebar.markdown("**Notes:**\n\n- Ensure `embeddings.npy` (a dict mapping names->embedding arrays) is present in the app folder on Streamlit Cloud.\n- Browser TTS uses SpeechSynthesis API (no server-side audio). On some browsers, playback may require a user interaction first.\n- If you deploy on Streamlit Cloud, add required packages to `requirements.txt` and upload `embeddings.npy` to the app files.")

