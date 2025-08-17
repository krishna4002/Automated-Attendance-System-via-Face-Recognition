# app.py — Real-time Attendance with WebRTC + Robust embeddings.npy loader
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
os.makedirs("attendance_logs", exist_ok=True)
_db_lock = threading.Lock()

# =============================
# AUDIO (OPTIONAL)
# =============================
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

# =============================
# DB HELPERS (SQLite)
# =============================
def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('Student','Teacher')),
            period TEXT,
            similarity REAL,
            ts_local TEXT NOT NULL,
            date_local TEXT NOT NULL,
            time_local TEXT NOT NULL
        )
        """)
        # indexes to reduce duplicates (INSERT OR IGNORE used)
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_student ON attendance(name, role, date_local, period)")
        # teacher uniqueness — period will be NULL
        # Not all SQLite versions allow partial index with WHERE; keep simple uniqueness handled by INSERT OR IGNORE
        conn.commit()

def mark_attendance(name: str, role: str, period: str | None, similarity: float | None = None):
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
# EMBEDDING LOADER + MODELS
# =============================
def _load_embeddings_any(path: str):
    """
    Robust loader for embeddings.npy.
    Supports:
      - dict { name: embedding }  (embedding: (512,) or (k,512))
      - dict { 'names': [...], 'embeddings': [[...], ...] }
      - ndarray (N,512)  -> will assign placeholder names ID_0...
      - list of embeddings (N,512)
    Returns: (names_list, embeddings_array_normalized) where embeddings_array shape (N,512)
    """
    if not os.path.exists(path):
        return [], np.empty((0,512), dtype=np.float32)

    try:
        loaded = np.load(path, allow_pickle=True)
    except Exception as e:
        st.error(f"Failed to load embeddings file: {e}")
        return [], np.empty((0,512), dtype=np.float32)

    # Many .npy files are saved as a pickled object, so .item() may work
    try:
        obj = loaded.item()
    except Exception:
        obj = loaded

    names = []
    embs = []

    if isinstance(obj, dict):
        # case: {'names': [...], 'embeddings': [...]}
        if 'names' in obj and 'embeddings' in obj:
            names = list(obj['names'])
            embs = np.asarray(obj['embeddings'], dtype=np.float32)
        else:
            # assume mapping name -> embedding or name -> list of embeddings
            for k, v in obj.items():
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] == 512:
                    names.append(str(k))
                    embs.append(arr)
                elif arr.ndim == 2 and arr.shape[1] == 512:
                    # multiple stored embeddings for same name -> average them
                    names.append(str(k))
                    embs.append(arr.mean(axis=0))
                else:
                    # unexpected shape; skip
                    continue
            if len(embs) > 0:
                embs = np.vstack(embs)
    elif isinstance(obj, np.ndarray) or isinstance(obj, list) or isinstance(obj, tuple):
        arr = np.asarray(obj, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 512:
            embs = arr
            names = [f"ID_{i}" for i in range(arr.shape[0])]
        elif arr.ndim == 1 and arr.shape[0] == 512:
            embs = arr.reshape(1, -1)
            names = ["ID_0"]
        else:
            st.warning("embeddings.npy format not recognized (unexpected array shape). Please provide either dict{name:embedding} or dict{'names','embeddings'}.")
            return [], np.empty((0,512), dtype=np.float32)
    else:
        st.warning("embeddings.npy format not supported.")
        return [], np.empty((0,512), dtype=np.float32)

    # L2 normalize gallery embeddings (row-wise)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-12)
    return names, embs_norm

@st.cache_resource(show_spinner=False)
def load_models_and_embeddings(path='embeddings.npy'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    names, embs = _load_embeddings_any(path)
    return device, mtcnn, model, names, embs

device, mtcnn, model, names_list, gallery_embs = load_models_and_embeddings('embeddings.npy')
init_db()

# show a small preview in UI
if len(names_list) == 0:
    st.sidebar.error("No embeddings loaded. Please ensure embeddings.npy exists and is in supported format.")
else:
    st.sidebar.success(f"Loaded {len(names_list)} identities")
    # show first few names
    st.sidebar.text("First names:")
    for n in names_list[:8]:
        st.sidebar.text(f" • {n}")

# =============================
# UTILS: schedule + recognition
# =============================
def get_current_period(schedule: dict):
    if not schedule:
        return None
    now = datetime.now(TZ).time()
    for period_name, (start, end) in schedule.items():
        if start <= now <= end:
            return period_name
    return None

def recognize_face_from_embedding(emb: np.ndarray, names, gallery_embs, threshold: float):
    """
    emb: ndarray shape (1,512) or (512,)
    gallery_embs: (N,512) normalized
    returns (name_or_None, best_similarity)
    """
    if gallery_embs.size == 0 or len(names) == 0:
        return None, 0.0
    probe = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    probe = probe / (np.linalg.norm(probe, axis=1, keepdims=True) + 1e-12)
    sims = np.dot(probe, gallery_embs.T)[0]  # cosine similarities
    idx = int(np.argmax(sims))
    best_sim = float(sims[idx])
    if best_sim >= threshold:
        return names[idx], best_sim
    return None, best_sim

def draw_label(img, text, pos=(20, 40), color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def parse_schedule_csv(csv_file):
    df = pd.read_csv(csv_file)
    schedule = {}
    for _, row in df.iterrows():
        name = row['Subject']
        start = datetime.strptime(str(row['Start']).strip(), "%H:%M").time()
        end = datetime.strptime(str(row['End']).strip(), "%H:%M").time()
        schedule[name] = (start, end)
    return schedule

# =============================
# UI: header + sidebar
# =============================
st.title("🧠 AI-Powered Attendance System (Real-Time, Robust Embeddings)")

left, right = st.columns([3, 2])
with right:
    st.markdown("**Timezone:** Asia/Kolkata 🇮🇳")
    st.caption(datetime.now(TZ).strftime("Now: %Y-%m-%d %H:%M:%S"))

mode = st.sidebar.radio("Choose Role", ["Student", "Teacher"])
today_local = datetime.now(TZ).strftime("%Y-%m-%d")

st.sidebar.subheader("🗂 Schedule Input")
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
            st.sidebar.success("✅ Schedule Loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")

st.sidebar.markdown("---")
capture_source = st.sidebar.selectbox("Camera Source", ["Browser (WebRTC) - recommended", "Local (OpenCV)"])
threshold = st.sidebar.slider("Similarity threshold", 0.50, 0.95, 0.75, 0.01)
st.sidebar.caption("Lower threshold = more matches (but more false positives).")

_last_event = st.empty()  # for live recognition proof in UI

# =============================
# WEBRTC processor
# =============================
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
                emb = model(face).cpu().numpy()  # (1,512)
            name, sim = recognize_face_from_embedding(emb, names_list, gallery_embs, threshold)
            if self.role == "Student":
                period = get_current_period(self.class_schedule)
                if name and period:
                    mark_attendance(name, "Student", period, sim)
                    draw_label(img, f"{name} ({sim:.2f}) - {period}", color=(0,200,0))
                    _last_event.info(f"✅ Marked: {name} • {period} • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                elif name and period is None:
                    draw_label(img, f"{name} ({sim:.2f}) - Not In Period", color=(0,165,255))
                else:
                    draw_label(img, f"Unknown (best sim {sim:.2f})", color=(0,0,255))
            else:  # Teacher
                if name:
                    mark_attendance(name, "Teacher", None, sim)
                    draw_label(img, f"{name} ({sim:.2f})", color=(255,0,0))
                    _last_event.info(f"✅ Marked: {name} • Teacher • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                else:
                    draw_label(img, f"Unknown (best sim {sim:.2f})", color=(0,0,255))
        else:
            draw_label(img, "No face detected", color=(0,0,255))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =============================
# MAIN layout
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
        st.info("Using your browser camera via WebRTC. Please allow camera access when prompted.")
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
                        name, sim = recognize_face_from_embedding(emb, names_list, gallery_embs, threshold)
                        if mode == "Student":
                            period = get_current_period(class_schedule)
                            if name and period:
                                mark_attendance(name, "Student", period, sim)
                                label = f"{name} ({sim:.2f}) - {period}"
                                _last_event.info(f"✅ Marked: {name} • {period} • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                            elif name and period is None:
                                label = f"{name} ({sim:.2f}) - Not In Period"
                            else:
                                label = f"Unknown (best sim {sim:.2f})"
                        else:
                            if name:
                                mark_attendance(name, "Teacher", None, sim)
                                label = f"{name} ({sim:.2f})"
                                _last_event.info(f"✅ Marked: {name} • Teacher • sim={sim:.2f} • {datetime.now(TZ).strftime('%H:%M:%S')}")
                            else:
                                label = f"Unknown (best sim {sim:.2f})"
                        cv2.putText(frame, label, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
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

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv_bytes,
        file_name=f"attendance_{date_filter}.csv",
        mime="text/csv",
    )

st.caption("Tip: On cloud deployments choose 'Browser (WebRTC)' to use the user's webcam in real time. Timestamps saved in Asia/Kolkata.")
