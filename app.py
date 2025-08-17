# app.py — Real-time Attendance via Browser Webcam (WebRTC) or Local Camera
# Uses embeddings.npy, stores attendance in SQLite (Asia/Kolkata), shows records + CSV download.

import os
import cv2
import numpy as np
import streamlit as st
import csv
from datetime import datetime, time
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import platform
import pyttsx3
import sqlite3
import threading
import pytz

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# -----------------------
# App config / globals
# -----------------------
st.set_page_config(page_title="AI Attendance System", layout="wide")
TZ = pytz.timezone("Asia/Kolkata")
DB_PATH = "attendance.db"
os.makedirs("attendance_logs", exist_ok=True)
_db_lock = threading.Lock()

# -----------------------
# Audio helper (optional)
# -----------------------
def play_audio(text):
    try:
        if platform.system() == "Windows":
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        else:
            os.system(f"say '{text}' 2>/dev/null || true")
    except Exception:
        pass

# -----------------------
# SQLite helpers
# -----------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            period TEXT,
            similarity REAL,
            ts_local TEXT NOT NULL,
            date_local TEXT NOT NULL,
            time_local TEXT NOT NULL
        )
        """)
        # unique constraints to avoid duplicates per-day
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_student ON attendance(name, role, date_local, period)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_teacher ON attendance(name, role, date_local) WHERE period IS NULL")
        conn.commit()

def mark_attendance(name, role, period=None, similarity=None):
    now = datetime.now(TZ)
    ts_local = now.strftime("%Y-%m-%d %H:%M:%S")
    date_local = now.strftime("%Y-%m-%d")
    time_local = now.strftime("%H:%M:%S")
    with _db_lock:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR IGNORE INTO attendance (name, role, period, similarity, ts_local, date_local, time_local)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, role, period, similarity, ts_local, date_local, time_local))
            conn.commit()

def fetch_attendance(date_filter=None, role_filter=None, name_filter=None, sim_min=0.0):
    query = "SELECT id, name, role, COALESCE(period,'') as period, similarity, date_local, time_local FROM attendance WHERE 1=1"
    params = []
    if date_filter:
        query += " AND date_local = ?"
        params.append(date_filter)
    if role_filter and role_filter in ("Student", "Teacher"):
        query += " AND role = ?"
        params.append(role_filter)
    if name_filter:
        query += " AND name LIKE ?"
        params.append(f"%{name_filter}%")
    query += " ORDER BY id DESC"
    with get_conn() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if not df.empty and sim_min > 0:
        df = df[df["similarity"].fillna(0) >= sim_min]
    return df

# -----------------------
# Model + embeddings loader
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models_and_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not os.path.exists('embeddings.npy'):
        st.error("`embeddings.npy` not found. Place it in project folder (expected dict name->embedding).")
        return device, mtcnn, model, {}

    emb_data = np.load('embeddings.npy', allow_pickle=True).item()
    # Normalize embeddings for cosine similarity
    normalized = {}
    for name, emb in emb_data.items():
        arr = np.asarray(emb, dtype=np.float32).reshape(-1)
        if np.linalg.norm(arr) == 0:
            continue
        arr = arr / (np.linalg.norm(arr) + 1e-12)
        normalized[name] = arr
    return device, mtcnn, model, normalized

device, mtcnn, model, embedding_dict = load_models_and_embeddings()
init_db()

# -----------------------
# Utilities
# -----------------------
def get_current_period(schedule):
    if not schedule:
        return None
    now = datetime.now(TZ).time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding, embedding_dict, threshold=0.75):
    if not embedding_dict or embedding is None:
        return None, 0.0
    # normalize probe
    emb = embedding.reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    best_name, best_sim = None, -1.0
    for name, ref in embedding_dict.items():
        sim = float(np.dot(emb, ref))  # since both normalized, dot = cosine
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim

def draw_label(img, text, pos=(20,40), color=(0,255,0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# -----------------------
# UI & Sidebar
# -----------------------
st.title("🧠 AI-Powered Attendance System (Real-Time)")
left_col, right_col = st.columns([3,1])
with right_col:
    st.markdown("**Timezone:** Asia/Kolkata 🇮🇳")
    st.caption(datetime.now(TZ).strftime("Now: %Y-%m-%d %H:%M:%S"))

mode = st.sidebar.radio("Choose Role", ["Student", "Teacher"])
st.sidebar.markdown("---")
st.sidebar.subheader("Schedule input")
schedule_option = st.sidebar.radio("How to provide class periods?", ["Manual", "Upload CSV"])
class_schedule = {}
if schedule_option == "Manual":
    num_periods = st.sidebar.number_input("Number of periods", 1, 10, 3)
    for i in range(num_periods):
        with st.sidebar.expander(f"Period {i+1}"):
            subject = st.text_input(f"Subject {i+1}", key=f"sub_{i}")
            start = st.time_input(f"Start {i+1}", key=f"start_{i}", value=time(9+i,0))
            end = st.time_input(f"End {i+1}", key=f"end_{i}", value=time(10+i,0))
            if subject:
                class_schedule[f"Period {i+1} - {subject}"] = (start, end)
else:
    csv_file = st.sidebar.file_uploader("Upload CSV (Subject,Start,End HH:MM)", type=['csv'])
    if csv_file:
        try:
            df_sched = pd.read_csv(csv_file)
            for _, row in df_sched.iterrows():
                subj = row['Subject']
                s = datetime.strptime(str(row['Start']).strip(), "%H:%M").time()
                e = datetime.strptime(str(row['End']).strip(), "%H:%M").time()
                class_schedule[subj] = (s,e)
            st.sidebar.success("Schedule loaded")
        except Exception as e:
            st.sidebar.error("Failed to parse CSV: " + str(e))

st.sidebar.markdown("---")
capture_source = st.sidebar.selectbox("Camera Source", ["Browser (WebRTC) - recommended", "Local (OpenCV)"])
st.sidebar.caption("Browser for cloud deploy; Local only on your machine.")

# Live feedback box
last_event = st.empty()

# -----------------------
# WebRTC processor
# -----------------------
class AttendanceProcessor(VideoProcessorBase):
    def __init__(self, role, schedule):
        self.role = role
        self.schedule = schedule or {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        face = mtcnn(img_pil)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy().reshape(-1)  # (512,)
            name, sim = recognize_face(emb, embedding_dict, threshold=0.75)
            if self.role == "Student":
                period = get_current_period(self.schedule)
                if name and period:
                    mark_attendance(name, "Student", period, float(sim))
                    draw_label(img, f"{name} ({sim:.2f}) - {period}", color=(0,200,0))
                    last_event.info(f"Marked: {name} • {period} • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                elif name and period is None:
                    draw_label(img, f"{name} ({sim:.2f}) - Not In Period", color=(0,165,255))
                else:
                    draw_label(img, f"Unknown (best {sim:.2f})", color=(0,0,255))
            else:  # Teacher
                if name:
                    mark_attendance(name, "Teacher", None, float(sim))
                    draw_label(img, f"{name} ({sim:.2f})", color=(255,0,0))
                    last_event.info(f"Marked: {name} • Teacher • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                else:
                    draw_label(img, f"Unknown (best {sim:.2f})", color=(0,0,255))
        else:
            draw_label(img, "No face detected", color=(0,0,255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------
# Main area: tabs
# -----------------------
tab_live, tab_records = st.tabs(["🎥 Live Recognition", "📑 Attendance Records"])

with tab_live:
    st.header("Live Recognition")
    st.write("Source:", capture_source)
    # Start/Stop logic
    if capture_source.startswith("Browser"):
        start_btn = st.button("Start (Browser WebRTC)")
        stop_placeholder = st.empty()
        if start_btn:
            webrtc_streamer(
                key=f"{mode.lower()}_webrtc",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=lambda: AttendanceProcessor(mode, class_schedule)
            )
    else:
        start_local = st.button("Start Local Camera")
        frame_slot = st.empty()
        if start_local:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open local camera.")
            else:
                stop_local = st.button("Stop Local Camera")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera read failed.")
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb)
                    face = mtcnn(img_pil)
                    if face is not None:
                        face = face.unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = model(face).cpu().numpy().reshape(-1)
                        name, sim = recognize_face(emb, embedding_dict, threshold=0.75)
                        if mode == "Student":
                            period = get_current_period(class_schedule)
                            if name and period:
                                mark_attendance(name, "Student", period, float(sim))
                                label = f"{name} ({sim:.2f}) - {period}"
                                last_event.info(f"Marked: {name} • {period} • sim={sim:.2f}")
                            elif name:
                                label = f"{name} ({sim:.2f}) - Not In Period"
                            else:
                                label = f"Unknown (best {sim:.2f})"
                        else:
                            if name:
                                mark_attendance(name, "Teacher", None, float(sim))
                                label = f"{name} ({sim:.2f})"
                                last_event.info(f"Marked: {name} • Teacher • sim={sim:.2f}")
                            else:
                                label = f"Unknown (best {sim:.2f})"
                        cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    frame_slot.image(frame, channels="BGR")
                    # break loop if Stop clicked
                    if st.button("Stop Local Camera"):
                        break
                cap.release()

with tab_records:
    st.header("Attendance Records")
    date_sel = st.date_input("Date", value=pd.to_datetime(datetime.now(TZ).strftime("%Y-%m-%d")))
    date_str = pd.to_datetime(date_sel).strftime("%Y-%m-%d")
    role_sel = st.selectbox("Role", ["All", "Student", "Teacher"])
    role_val = None if role_sel == "All" else role_sel
    name_filter = st.text_input("Name contains")
    sim_min = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.01)

    df = fetch_attendance(date_filter=date_str, role_filter=role_val, name_filter=name_filter.strip() or None, sim_min=sim_min)
    if df.empty:
        st.info("No records for selected filters")
    else:
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv_bytes, file_name=f"attendance_{date_str}.csv", mime="text/csv")

st.caption("Tip: On cloud deployments choose Browser (WebRTC). All timestamps saved in Asia/Kolkata.")
