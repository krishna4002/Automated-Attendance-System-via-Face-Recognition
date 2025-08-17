import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, time
from PIL import Image
import torch
import pytz
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# =============================
# CONFIGURATION
# =============================
st.set_page_config(page_title="AI Attendance System", layout="wide")
tz = pytz.timezone("Asia/Kolkata")

# Database setup
DB_PATH = "attendance.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    name TEXT,
    period TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load embeddings
embedding_dict = np.load("embeddings.npy", allow_pickle=True).item()

# =============================
# FUNCTIONS
# =============================
def get_current_period(schedule):
    now = datetime.now(tz).time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding):
    best_match, highest_similarity = None, 0.0
    for name, ref_emb in embedding_dict.items():
        sim = cosine_similarity(embedding, ref_emb.reshape(1, -1))[0][0]
        if sim > 0.75 and sim > highest_similarity:
            best_match, highest_similarity = name, sim
    return best_match

def mark_attendance(role, name, period=None):
    now = datetime.now(tz)
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")

    if role == "Student":
        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=? AND period=?", (name, date_str, period))
    else:
        cursor.execute("SELECT * FROM attendance WHERE name=? AND date=? AND role=?", (name, date_str, role))

    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO attendance (role, name, period, date, time) VALUES (?, ?, ?, ?, ?)",
                       (role, name, period if period else "", date_str, time_str))
        conn.commit()
        st.success(f"✅ Attendance marked for {name}")

def parse_schedule_csv(csv_file):
    df = pd.read_csv(csv_file)
    schedule = {}
    for _, row in df.iterrows():
        name = row["Subject"]
        start = datetime.strptime(row["Start"], "%H:%M").time()
        end = datetime.strptime(row["End"], "%H:%M").time()
        schedule[name] = (start, end)
    return schedule

def view_attendance():
    df = pd.read_sql("SELECT role, name, period, date, time FROM attendance ORDER BY date DESC, time DESC", conn)
    st.dataframe(df)
    st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv")

# =============================
# STREAMLIT UI
# =============================
st.title("🧠 AI-Powered Attendance System (Cloud Version)")

menu = ["Student Mode", "Teacher Mode", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

# Schedule setup
class_schedule = {}
if choice == "Student Mode":
    st.sidebar.subheader("🗂 Schedule Input")
    schedule_option = st.sidebar.radio("Schedule input method:", ["Manual", "Upload CSV"])

    if schedule_option == "Manual":
        num_periods = st.sidebar.number_input("Number of Periods", min_value=1, max_value=10, value=3)
        for i in range(num_periods):
            with st.sidebar.expander(f"📘 Period {i+1} Settings"):
                subject = st.text_input(f"Subject Name {i+1}", key=f"sub_{i}")
                start = st.time_input(f"Start Time {i+1}", key=f"start_{i}")
                end = st.time_input(f"End Time {i+1}", key=f"end_{i}")
                if subject:
                    class_schedule[f"Period {i+1} - {subject}"] = (start, end)

    elif schedule_option == "Upload CSV":
        st.sidebar.subheader("📁 Upload CSV (Subject,Start,End)")
        csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if csv_file is not None:
            class_schedule = parse_schedule_csv(csv_file)
            st.sidebar.success("✅ Schedule Loaded")

# =============================
# Student Mode
# =============================
if choice == "Student Mode":
    st.subheader("📚 Student Attendance")
    uploaded_img = st.camera_input("Take a photo to mark attendance")
    if uploaded_img:
        img = Image.open(uploaded_img)
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy()
            name = recognize_face(emb)
            period = get_current_period(class_schedule)
            if name and period:
                mark_attendance("Student", name, period)
            elif name:
                st.warning(f"{name} detected, but not in any active class period.")
            else:
                st.error("Face not recognized.")

# =============================
# Teacher Mode
# =============================
elif choice == "Teacher Mode":
    st.subheader("🎓 Teacher Attendance")
    uploaded_img = st.camera_input("Take a photo to mark teacher attendance")
    if uploaded_img:
        img = Image.open(uploaded_img)
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy()
            name = recognize_face(emb)
            if name:
                mark_attendance("Teacher", name)
            else:
                st.error("Face not recognized.")

# =============================
# View Attendance
# =============================
elif choice == "View Attendance":
    st.subheader("📋 Attendance Records")
    view_attendance()
