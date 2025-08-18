# recognize_and_mark_attendance.py

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

# =============================
# CONFIGURATION
# =============================

def play_audio(text):
    try:
        if platform.system() == "Windows":
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            print(f"[VOICE] Speaking: {text}")
        else:
            os.system(f"say '{text}'")
            print(f"[VOICE - fallback] Speaking: {text}")
    except Exception as e:
        print(f"[VOICE ERROR] {e}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
embedding_dict = np.load('embeddings.npy', allow_pickle=True).item()
os.makedirs("attendance_logs", exist_ok=True)

# =============================
# UTILITY FUNCTIONS
# =============================

def get_current_period(schedule):
    now = datetime.now().time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding):
    best_match = None
    highest_similarity = 0.0
    for name, ref_emb in embedding_dict.items():
        sim = cosine_similarity(embedding, ref_emb.reshape(1, -1))[0][0]
        if sim > 0.75 and sim > highest_similarity:
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
        start = datetime.strptime(row['Start'], "%H:%M").time()
        end = datetime.strptime(row['End'], "%H:%M").time()
        schedule[name] = (start, end)
    return schedule

def load_existing_attendance(path):
    existing_entries = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    existing_entries.add((row[0], row[2]))  # (name, period)
                elif len(row) >= 2:
                    existing_entries.add(row[0])  # for teacher
    return existing_entries

# =============================
# MAIN STREAMLIT APP
# =============================

st.set_page_config(page_title="AI Attendance System", layout="centered")
st.title("🧠 AI-Powered Attendance System")
mode = st.sidebar.radio("Choose Role", ["Student", "Teacher"])
today = datetime.now().strftime("%Y-%m-%d")

# =============================
# Schedule input options
# =============================

st.sidebar.subheader("🗂 Schedule Input Method")
schedule_option = st.sidebar.radio("How would you like to input class periods?", ["Manual", "Upload CSV"])
class_schedule = {}

if schedule_option == "Manual":
    st.sidebar.subheader("🕒 Class Period Configuration")
    num_periods = st.sidebar.number_input("Number of Periods", min_value=1, max_value=10, value=3)

    for i in range(num_periods):
        with st.sidebar.expander(f"📘 Period {i+1} Settings"):
            subject = st.text_input(f"Subject Name {i+1}", key=f"sub_{i}")
            start = st.time_input(f"Start Time {i+1}", key=f"start_{i}")
            end = st.time_input(f"End Time {i+1}", key=f"end_{i}")
            if subject:
                class_schedule[f"Period {i+1} - {subject}"] = (start, end)

elif schedule_option == "Upload CSV":
    st.sidebar.subheader("📁 Upload CSV with Columns: Subject,Start,End")
    csv_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    if csv_file is not None:
        class_schedule = parse_schedule_csv(csv_file)
        st.sidebar.success("✅ Schedule Loaded from CSV")

# =============================
# Student Mode
# =============================

if mode == "Student":
    st.subheader("📚 Student Mode (Dynamic Periods)")
    csv_path = f'attendance_logs/student_attendance_{today}.csv'
    initialize_attendance_file(csv_path, ["Name", "Time", "Class Period"])
    existing_attendance = load_existing_attendance(csv_path)

    if st.checkbox("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

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
                name = recognize_face(emb)

                if name is None:
                    label = "Face Not Recognized"
                elif period is None:
                    label = f"{name} - Not In Period"
                else:
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

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")
        cap.release()

# =============================
# Teacher Mode
# =============================

elif mode == "Teacher":
    st.subheader("🎓 Teacher Mode (One-time Daily Attendance)")
    csv_path = f'attendance_logs/teacher_attendance_{today}.csv'
    initialize_attendance_file(csv_path, ["Name", "Time"])
    existing_teachers = load_existing_attendance(csv_path)

    if st.checkbox("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

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

                name = recognize_face(emb)

                if name:
                    if name not in existing_teachers:
                        existing_teachers.add(name)
                        try:
                            with open(csv_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
                            play_audio(f"Attendance marked for {name}")
                        except PermissionError:
                            st.error("❌ Cannot write to file. Make sure it's not open elsewhere.")
                    label = name  # Clean name only
                else:
                    label = "Face Not Recognized"

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            stframe.image(frame, channels="BGR")
        cap.release()