# app.py — Real-time Attendance via Browser Webcam (WebRTC) + SQLite Logging

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
from zoneinfo import ZoneInfo
import time as py_time

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
INDIA_TZ = ZoneInfo("Asia/Kolkata")
DEBOUNCE_SECONDS = 10 # seconds to wait before marking the same person again

# ---------------------------
# MODEL + EMBEDDINGS LOADER
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not os.path.exists('embeddings.npy'):
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
    c.execute("""
        CREATE TABLE IF NOT EXISTS student_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            name TEXT,
            time TEXT,
            period TEXT,
            date TEXT,
            UNIQUE(student_id, period, date)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS teacher_attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            teacher_id TEXT,
            name TEXT,
            time TEXT,
            date TEXT,
            UNIQUE(teacher_id, date)
        )
    """)
    conn.commit()
    conn.close()

def mark_student_db(student_id, name, period):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(INDIA_TZ)
    today = now.strftime("%Y-%m-%d")
    c.execute("INSERT OR IGNORE INTO student_attendance (student_id, name, time, period, date) VALUES (?, ?, ?, ?, ?)",
              (student_id, name, now.strftime("%H:%M:%S"), period, today))
    conn.commit()
    conn.close()

def mark_teacher_db(teacher_id, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(INDIA_TZ)
    today = now.strftime("%Y-%m-%d")
    c.execute("INSERT OR IGNORE INTO teacher_attendance (teacher_id, name, time, date) VALUES (?, ?, ?, ?)",
              (teacher_id, name, now.strftime("%H:%M:%S"), today))
    conn.commit()
    conn.close()

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
    for key, data in embedding_dict.items():
        ref_emb = data["embedding"]
        sim = cosine_similarity(embedding, ref_emb.reshape(1, -1))[0][0]
        if sim > threshold and sim > highest_similarity:
            best_match, highest_similarity = data, sim
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

def draw_label(img, text, y_pos, color=(0, 255, 0), font_scale=0.8, thickness=2):
    cv2.putText(img, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ---------------------------
# UI / Mode Selection
# ---------------------------
st.title("🧠 AI-Powered Attendance System")
mode = st.sidebar.radio("Choose Option", ["Student", "Teacher", "📑 View Attendance Logs"])

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

# ---------------------------
# WEBRTC VIDEO PROCESSOR
# ---------------------------
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self, role, class_schedule):
        self.role = role
        self.class_schedule = class_schedule or {}
        # Initialize debounce state outside of the webrtc stream, in session_state
        if 'last_marked_person' not in st.session_state:
            st.session_state.last_marked_person = None
            st.session_state.last_marked_time = 0
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        face = mtcnn(img_pil)

        y_offset = 40 # Initial y position for the label

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy()
            match = recognize_face(emb, embedding_dict)

            if match:
                person_id = match["id"]
                person_name = match["name"].replace(" ", "_")
                label = f"{person_name} (ID: {person_id})"
                
                # Check current time for debouncing
                current_time = py_time.time()
                is_same_person = (st.session_state.last_marked_person == person_id)
                time_since_last_mark = current_time - st.session_state.last_marked_time
                
                if not is_same_person or time_since_last_mark > DEBOUNCE_SECONDS:
                    # Update session state to prevent immediate re-marking
                    st.session_state.last_marked_person = person_id
                    st.session_state.last_marked_time = current_time

                    if self.role == "Student":
                        period = get_current_period(self.class_schedule)
                        if period:
                            mark_student_db(person_id, person_name, period)
                            draw_label(img, f"{label} - ATTENDANCE MARKED", y_offset, color=(0, 255, 0))
                        else:
                            draw_label(img, f"{label} - Not in Period", y_offset, color=(0, 0, 255))
                    elif self.role == "Teacher":
                        mark_teacher_db(person_id, person_name)
                        draw_label(img, f"{label} - ATTENDANCE MARKED", y_offset, color=(0, 255, 0))
                else:
                    draw_label(img, f"{label} - already Present", y_offset, color=(255, 0, 0))
            else:
                draw_label(img, "Face Not Recognized", y_offset, color=(0, 0, 255))
        else:
            draw_label(img, "No Face Detected",y_offset, color=(0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------
# Modes: Student / Teacher / Logs
# ---------------------------
if mode == "Student":
    st.subheader("📚 Student Mode (Real-Time)")
    st.write("Camera will run continuously to detect faces and mark attendance during active periods.")
    webrtc_streamer(
        key="student_attendance",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: AttendanceProcessor("Student", class_schedule),
    )

elif mode == "Teacher":
    st.subheader("🎓 Teacher Mode (Real-Time)")
    st.write("Camera will run continuously to detect faces and mark attendance.")
    webrtc_streamer(
        key="teacher_attendance",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: AttendanceProcessor("Teacher", class_schedule),
    )

elif mode == "📑 View Attendance Logs":
    st.subheader("📑 Attendance Logs")
    if st.button("🗑 Reset Database"):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS student_attendance")
        c.execute("DROP TABLE IF EXISTS teacher_attendance")
        conn.commit()
        conn.close()
        init_db()
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
