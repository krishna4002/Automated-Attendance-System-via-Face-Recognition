# app.py — Real-time Attendance with WebRTC + SQLite (Asia/Kolkata)

import os
import cv2
import numpy as np
import streamlit as st
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

# WebRTC
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# =============================
# APP CONFIG
# =============================

st.set_page_config(page_title="AI Attendance System", layout="wide")
TZ = pytz.timezone("Asia/Kolkata")
DB_NAME = "attendance.db"
os.makedirs("attendance_logs", exist_ok=True)  # kept if you still want csv exports quickly
_db_lock = threading.Lock()

# =============================
# AUDIO (OPTIONAL)
# =============================

def play_audio(text):
    """Non-blocking best-effort voice feedback."""
    try:
        if platform.system() == "Windows":
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        else:
            os.system(f"say '{text}' 2>/dev/null || true")
    except Exception:
        pass

# =============================
# DB HELPERS (SQLite)
# =============================

def get_conn():
    # check_same_thread=False allows usage in WebRTC worker thread
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('Student','Teacher')),
            period TEXT,                     -- NULL for Teacher
            similarity REAL,                 -- similarity score for proof
            ts_local TEXT NOT NULL,          -- ISO timestamp in Asia/Kolkata
            date_local TEXT NOT NULL,        -- YYYY-MM-DD (Asia/Kolkata)
            time_local TEXT NOT NULL         -- HH:MM:SS (Asia/Kolkata)
        )
        """)
        # Unique constraints to prevent duplicates
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_student ON attendance(name, role, date_local, period)")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_teacher ON attendance(name, role, date_local) WHERE period IS NULL")
        conn.commit()

def mark_attendance(name: str, role: str, period: str | None, similarity: float | None = None):
    now = datetime.now(TZ)
    ts_local = now.strftime("%Y-%m-%d %H:%M:%S")
    date_local = now.strftime("%Y-%m-%d")
    time_local = now.strftime("%H:%M:%S")
    with _db_lock:
        with get_conn() as conn:
            c = conn.cursor()
            # INSERT OR IGNORE due to unique indexes
            c.execute("""
                INSERT OR IGNORE INTO attendance (name, role, period, similarity, ts_local, date_local, time_local)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, role, period, similarity, ts_local, date_local, time_local))
            conn.commit()

def fetch_attendance(date_filter: str | None = None, role_filter: str | None = None, name_filter: str | None = None):
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
    return df

# =============================
# MODELS / EMBEDDINGS
# =============================

@st.cache_resource(show_spinner=False)
def load_models_and_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    if not os.path.exists('embeddings.npy'):
        st.error("`embeddings.npy` not found. Please generate embeddings before running the app.")
        return device, mtcnn, model, {}

    embedding_dict = np.load('embeddings.npy', allow_pickle=True).item()
    # embedding_dict expected as { name: (512,) ndarray } OR name->list
    # normalize stored embeddings once for consistent cosine similarity
    normalized = {}
    for name, emb in embedding_dict.items():
        arr = np.asarray(emb, dtype=np.float32).reshape(1, -1)
        # L2-normalize reference embeddings
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        normalized[name] = arr.squeeze(0)
    return device, mtcnn, model, normalized

device, mtcnn, model, embedding_dict = load_models_and_embeddings()
init_db()

# =============================
# UTILITY: SCHEDULE + RECOGNITION
# =============================

def get_current_period(schedule: dict):
    """Return current period name or None if outside all ranges (Asia/Kolkata)."""
    if not schedule:
        return None
    now = datetime.now(TZ).time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face(embedding: np.ndarray, embedding_dict: dict, threshold: float = 0.75):
    """
    Return (best_name, similarity) if similarity exceeds threshold, else (None, best_similarity).
    embedding: (1,512)
    """
    if not embedding_dict:
        return None, 0.0

    # L2-normalize probe embedding for cosine similarity stability
    emb = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12)

    best_name, best_sim = None, -1.0
    for name, ref_emb in embedding_dict.items():
        # ref_emb is (512,), turn into (1,512)
        sim = cosine_similarity(emb, ref_emb.reshape(1, -1))[0][0]
        if sim > best_sim:
            best_name, best_sim = name, sim

    if best_sim >= threshold:
        return best_name, float(best_sim)
    return None, float(best_sim)

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

# =============================
# UI: HEADER + SIDEBAR
# =============================

st.title("🧠 AI-Powered Attendance System (Real-Time, SQLite)")

left, right = st.columns([3, 2])
with right:
    st.markdown("**Timezone:** Asia/Kolkata 🇮🇳")
    st.caption(datetime.now(TZ).strftime("Now: %Y-%m-%d %H:%M:%S"))

mode = st.sidebar.radio("Choose Role", ["Student", "Teacher"])
today_local = datetime.now(TZ).strftime("%Y-%m-%d")

st.sidebar.subheader("🗂 Schedule Input Method")
schedule_option = st.sidebar.radio("Input class periods?", ["Manual", "Upload CSV"])
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
else:
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

_last_event = st.empty()  # live recognition proof

class AttendanceProcessor(VideoProcessorBase):
    def __init__(self, role, class_schedule):
        self.role = role
        self.class_schedule = class_schedule or {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Face detection (MTCNN expects PIL RGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        face = mtcnn(img_pil)

        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face).cpu().numpy()  # (1,512)

            name, sim = recognize_face(emb, embedding_dict, threshold=0.75)

            if self.role == "Student":
                period = get_current_period(self.class_schedule)
                if name and period:
                    # mark into DB (Asia/Kolkata)
                    mark_attendance(name, "Student", period, sim)
                    draw_label(img, f"{name} ({sim:.2f}) - {period}", color=(0, 200, 0))
                    _last_event.info(f"✅ Marked: **{name}** • **{period}** • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                    # Optional voice
                    # play_audio(f"Attendance marked for {name}")
                elif name and period is None:
                    draw_label(img, f"{name} ({sim:.2f}) - Not In Period", color=(0, 165, 255))
                else:
                    draw_label(img, f"Unknown (best sim {sim:.2f})", color=(0, 0, 255))

            elif self.role == "Teacher":
                if name:
                    mark_attendance(name, "Teacher", None, sim)
                    draw_label(img, f"{name} ({sim:.2f})", color=(255, 0, 0))
                    _last_event.info(f"✅ Marked: **{name}** • **Teacher** • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                    # play_audio(f"Attendance marked for {name}")
                else:
                    draw_label(img, f"Unknown (best sim {sim:.2f})", color=(0, 0, 255))
        else:
            draw_label(img, "No face detected", color=(0, 0, 255))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# MAIN LAYOUT: CAMERA + RECORDS
# =============================

tab_live, tab_records = st.tabs(["🎥 Live Recognition", "📑 Attendance Records"])

with tab_live:
    if mode == "Student":
        st.subheader("📚 Student Mode (Real-Time)")
    else:
        st.subheader("🎓 Teacher Mode (Real-Time)")

    if capture_source.startswith("Browser"):
        webrtc_streamer(
            key=f"{mode.lower()}_attendance",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: AttendanceProcessor(mode, class_schedule),
        )
        st.info("Using your **browser camera** via WebRTC. Please allow camera access when prompted.")
    else:
        st.warning("Local camera selected. This only works on your own machine (not on cloud).")
        run_local = st.checkbox("Start Local Webcam")
        frame_slot = st.empty()
        if run_local:
            cap = cv2.VideoCapture(0)
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
                        name, sim = recognize_face(emb, embedding_dict, threshold=0.75)
                        if mode == "Student":
                            period = get_current_period(class_schedule)
                            if name and period:
                                mark_attendance(name, "Student", period, sim)
                                label = f"{name} ({sim:.2f}) - {period}"
                                _last_event.info(f"✅ Marked: **{name}** • **{period}** • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                            elif name and period is None:
                                label = f"{name} ({sim:.2f}) - Not In Period"
                            else:
                                label = f"Unknown (best sim {sim:.2f})"
                        else:  # Teacher
                            if name:
                                mark_attendance(name, "Teacher", None, sim)
                                label = f"{name} ({sim:.2f})"
                                _last_event.info(f"✅ Marked: **{name}** • **Teacher** • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                            else:
                                label = f"Unknown (best sim {sim:.2f})"
                        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_slot.image(frame, channels="BGR")
                cap.release()

with tab_records:
    st.subheader("📑 Attendance Records (SQLite)")
    colf1, colf2, colf3, colf4 = st.columns([1.2, 1, 1, 1])
    with colf1:
        date_filter = st.date_input("Date", value=pd.to_datetime(today_local)).strftime("%Y-%m-%d")
    with colf2:
        role_filter = st.selectbox("Role", ["All", "Student", "Teacher"])
        role_val = None if role_filter == "All" else role_filter
    with colf3:
        name_filter = st.text_input("Name contains", "")
    with colf4:
        sim_min = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.01)

    df = fetch_attendance(date_filter=date_filter, role_filter=role_val, name_filter=name_filter.strip() or None)
    if sim_min > 0:
        df = df[df["similarity"].fillna(0) >= sim_min]

    st.dataframe(df, use_container_width=True, height=420)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_bytes,
        file_name=f"attendance_{date_filter}.csv",
        mime="text/csv",
    )

st.caption("Tip: On cloud deployments, choose 'Browser (WebRTC)' to use the user's webcam in real time. All timestamps are saved in Asia/Kolkata.")
