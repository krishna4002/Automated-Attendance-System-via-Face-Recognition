# app.py  — Real-time Attendance via Browser Webcam (WebRTC) or Local Camera

import os
import cv2
import numpy as np
import streamlit as st
import pickle
import csv
from datetime import datetime, time
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import platform
import pyttsx3

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# =============================
# CONFIGURATION / MODELS
# =============================

st.set_page_config(page_title="AI Attendance System", layout="centered")

def play_audio(text):
    """Non-blocking best-effort voice feedback."""
    try:
        if platform.system() == "Windows":
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            print(f"[VOICE] Speaking: {text}")
        else:
            # macOS 'say' (Linux servers typically won't have this; that's okay)
            os.system(f"say '{text}' 2>/dev/null || true")
            print(f"[VOICE - fallback] Speaking: {text}")
    except Exception as e:
        print(f"[VOICE ERROR] {e}")

@st.cache_resource(show_spinner=False)
def load_models_and_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load {name -> 512-d embedding} dict from embeddings.npy
    if not os.path.exists('embeddings.npy'):
        st.error("`embeddings.npy` not found. Please generate embeddings before running the app.")
        return device, mtcnn, model, {}

    embedding_dict = np.load('embeddings.npy', allow_pickle=True).item()
    return device, mtcnn, model, embedding_dict

device, mtcnn, model, embedding_dict = load_models_and_embeddings()
os.makedirs("attendance_logs", exist_ok=True)

# =============================
# UTILITY FUNCTIONS
# =============================

def get_current_period(schedule: dict):
    """Return current period name or None if outside all ranges."""
    if not schedule:
        return None
    now = datetime.now().time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding, embedding_dict, threshold=0.75):
    """Return best-matching name if similarity exceeds threshold, else None."""
    if embedding_dict is None or len(embedding_dict) == 0:
        return None
    best_match = None
    highest_similarity = 0.0
    for name, ref_emb in embedding_dict.items():
        sim = cosine_similarity(embedding, ref_emb.reshape(1, -1))[0][0]
        if sim > threshold and sim > highest_similarity:
            best_match = name
            highest_similarity = sim
    return best_match

def initialize_attendance_file(path, columns):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

def parse_schedule_csv(csv_file):
    df = pd.read_csv(csv_file)
    schedule = {}
    for _, row in df.iterrows():
        name = row['Subject']
        start = datetime.strptime(str(row['Start']).strip(), "%H:%M").time()
        end = datetime.strptime(str(row['End']).strip(), "%H:%M").time()
        schedule[name] = (start, end)
    return schedule

def load_existing_attendance(path):
    """Return a set of existing entries to prevent dupes.
       Student: set of (name, period)
       Teacher: set of name
    """
    existing_entries = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    existing_entries.add((row[0], row[2]))  # (name, period)
                elif len(row) >= 1:
                    existing_entries.add(row[0])  # name only
    return existing_entries

def draw_label(img, text, pos=(20, 40), color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# =============================
# SIDEBAR & SCHEDULE
# =============================

st.title("🧠 AI-Powered Attendance System (Real-Time)")
mode = st.sidebar.radio("Choose Role", ["Student", "Teacher"])
today = datetime.now().strftime("%Y-%m-%d")

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

st.sidebar.markdown("---")
capture_source = st.sidebar.selectbox("Camera Source", ["Browser (WebRTC) - recommended", "Local (OpenCV)"])
st.sidebar.caption("Use Local only on your own machine. On cloud, keep Browser (WebRTC).")

# =============================
# WEBRTC PROCESSOR
# =============================

class AttendanceProcessor(VideoProcessorBase):
    def __init__(self, role, class_schedule, csv_path, existing_entries):
        self.role = role
        self.class_schedule = class_schedule or {}
        self.csv_path = csv_path
        self.existing = existing_entries

    def mark_student(self, name, period):
        uid = (name, period)
        if uid not in self.existing:
            self.existing.add(uid)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime("%H:%M:%S"), period])
            play_audio(f"Attendance marked for {name}")

    def mark_teacher(self, name):
        if name not in self.existing:
            self.existing.add(name)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
            play_audio(f"Attendance marked for {name}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection (single face assumed; extend as needed)
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
                    self.mark_student(name, period)
                    draw_label(img, f"{name} - {period}", color=(0, 200, 0))
                elif name and period is None:
                    draw_label(img, f"{name} - Not In Period", color=(0, 165, 255))
                else:
                    draw_label(img, "Face Not Recognized", color=(0, 0, 255))

            elif self.role == "Teacher":
                if name:
                    self.mark_teacher(name)
                    draw_label(img, name, color=(255, 0, 0))
                else:
                    draw_label(img, "Face Not Recognized", color=(0, 0, 255))
        else:
            draw_label(img, "No face detected", color=(0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# MODES
# =============================

if mode == "Student":
    st.subheader("📚 Student Mode (Real-Time)")
    csv_path = f'attendance_logs/student_attendance_{today}.csv'
    initialize_attendance_file(csv_path, ["Name", "Time", "Class Period"])
    existing_attendance = load_existing_attendance(csv_path)

    if capture_source.startswith("Browser"):
        # WebRTC (works in cloud)
        webrtc_streamer(
            key="student_attendance",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AttendanceProcessor("Student", class_schedule, csv_path, existing_attendance),
        )
        st.info("Using your browser camera via WebRTC. Please allow camera access when prompted.")
    else:
        # Local OpenCV (use only on your machine)
        st.warning("Local camera selected. This only works on your own machine (not on cloud).")
        if st.checkbox("Start Local Webcam"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            if not cap.isOpened():
                st.error("Could not open local camera. Try another index or ensure permissions.")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    face = mtcnn(img_pil)
                    if face is not None:
                        face = face.unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model(face).cpu().numpy()
                        period = get_current_period(class_schedule)
                        name = recognize_face(emb, embedding_dict)
                        if name and period:
                            uid = (name, period)
                            if uid not in existing_attendance:
                                existing_attendance.add(uid)
                                try:
                                    with open(csv_path, 'a', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([name, datetime.now().strftime("%H:%M:%S"), period])
                                    play_audio(f"Attendance marked for {name}")
                                except PermissionError:
                                    st.error("❌ Cannot write to file. Make sure it's not open elsewhere.")
                            label = f"{name} - {period}"
                        elif name and period is None:
                            label = f"{name} - Not In Period"
                        else:
                            label = "Face Not Recognized"
                        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    stframe.image(frame, channels="BGR")
                cap.release()

elif mode == "Teacher":
    st.subheader("🎓 Teacher Mode (Real-Time)")
    csv_path = f'attendance_logs/teacher_attendance_{today}.csv'
    initialize_attendance_file(csv_path, ["Name", "Time"])
    existing_teachers = load_existing_attendance(csv_path)

    if capture_source.startswith("Browser"):
        webrtc_streamer(
            key="teacher_attendance",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AttendanceProcessor("Teacher", class_schedule, csv_path, existing_teachers),
        )
        st.info("Using your browser camera via WebRTC. Please allow camera access when prompted.")
    else:
        st.warning("Local camera selected. This only works on your own machine (not on cloud).")
        if st.checkbox("Start Local Webcam"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            if not cap.isOpened():
                st.error("Could not open local camera. Try another index or ensure permissions.")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    face = mtcnn(img_pil)
                    if face is not None:
                        face = face.unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model(face).cpu().numpy()
                        name = recognize_face(emb, embedding_dict)
                        if name and name not in existing_teachers:
                            existing_teachers.add(name)
                            try:
                                with open(csv_path, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
                                play_audio(f"Attendance marked for {name}")
                            except PermissionError:
                                st.error("❌ Cannot write to file. Make sure it's not open elsewhere.")
                        label = name if name else "Face Not Recognized"
                        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    stframe.image(frame, channels="BGR")
                cap.release()

# Footer hint
st.caption("Tip: On cloud deployments, always choose 'Browser (WebRTC)' to use the user's webcam in real time.")
