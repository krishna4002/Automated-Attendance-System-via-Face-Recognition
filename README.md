# 🧠 AI-Powered Attendance System using Face Recognition

A real-time AI-powered attendance system that uses face recognition through a webcam to **automate attendance** for both **students (per class period)** and **teachers (once per day)**. The system includes **voice confirmation** (Windows only), **CSV logging**, and **schedule customization** — all in a simple Streamlit web app.

This project demonstrates a practical use of machine learning, computer vision, and user interface design to solve a real-world problem in education and corporate environments.

---

## 📖 About This Project

This system is designed to eliminate the manual and error-prone process of recording attendance. It works by recognizing the user's face from a live webcam feed and automatically logging their name, time, and subject in a structured CSV file.

- **Students**: Attendance is recorded for each subject period.
- **Teachers**: Attendance is recorded only once per day.
- The system prevents **duplicate entries**, supports **schedule upload via CSV**, and offers **voice-based confirmation** after successful recognition.

---

## ✨ Features

- 📸 Real-time face recognition via webcam
- 🧑‍🎓 Student attendance by class period
- 👨‍🏫 Teacher attendance once per day
- ❗ Prevents duplicate entries
- 🔊 Voice confirmation when marked (Windows only)
- 🗃 CSV logs auto-saved daily
- 📅 Custom class schedules (manual or CSV upload)
- 🧠 Face recognition using `facenet-pytorch`
- ⚙️ Fully local: No cloud API or internet dependency
- 💻 Simple, interactive web interface using Streamlit

---

## 📂 Project Structure

```
ai-attendance-system/
├── app.py                   # Main Streamlit app for marking attendance
├── face_captured_app.py     # Streamlit UI to collect face data from webcam
├── generate_embeddings.py   # Script to create/update face embeddings
├── dataset/                 # Folder for registered users' face images
│   └── Krishnagopal Jay/            # Example: contains 1.jpg, 2.jpg, ..., 30.jpg
├── embeddings.npy           # NumPy array of all face embeddings
├── attendance_logs/         # Automatically saved daily attendance CSVs
│   ├── student_attendance_YYYY-MM-DD.csv
│   └── teacher_attendance_YYYY-MM-DD.csv
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation (this file)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-attendance-system.git
cd ai-attendance-system
```

### 2. (Optional) Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Collecting Face Data via UI

Instead of manually placing images in `dataset/`, you can run the face capture UI:

```bash
streamlit run face_captured_app.py
```

This interface allows you to:
- Capture images directly from the webcam
- Save them under a named folder inside `dataset/`
- Automatically collect ~30 images per person for training

After capturing faces, update the embeddings by running:

```bash
python generate_embeddings.py
```

---

## 👤 Add Students or Teachers (Manual Method)

1. Inside the `dataset/` folder, create a folder with the person’s **full name**  
   Example: `dataset/Krishnagopal Jay/`

2. Add **30+ clear face images** of that person (1 face per image)

3. Run:

```bash
python generate_embeddings.py
```

This script generates `embeddings.npy` for face matching.

---

## ▶️ Run the Attendance App

```bash
streamlit run app.py
```

From the web interface, choose:
- `Student` → Period-wise attendance
- `Teacher` → Daily attendance (once)

---

## 🗓 Schedule Setup

### ✅ Option 1: Manual Entry
Add subjects and timings directly inside the app.

### 📁 Option 2: Upload CSV Schedule

Example `schedule.csv`:

```csv
Subject,Start,End
Math,09:00,09:45
Physics,10:00,10:45
Break,10:45,11:00
```

Upload via the Streamlit interface when prompted.

---

## 🔊 Voice Confirmation (Windows Only)

When attendance is successfully marked, the system speaks:

> “Attendance marked for Krishnagopal Jay”

Make sure:
- Your device has working speakers
- `pyttsx3` is installed (already included)
- You are running the system on Windows OS

---

## 🗃 Attendance Logs

All logs are stored in the `attendance_logs/` folder:

```
attendance_logs/
├── student_attendance_YYYY-MM-DD.csv
├── teacher_attendance_YYYY-MM-DD.csv
```

Each row contains: **Name, Role, Time, Subject (if applicable)**.

---

## 📌 Behavior & Constraints

- 🧍 One face per image (no group photos)
- ✅ Only one attendance per person per period/day
- ⏰ Student attendance allowed **only during scheduled time slots**
- 🔁 Embeddings must be updated when adding new users
- 🎯 Recognition accuracy improves with clear, consistent images

---

## 🧰 Troubleshooting

| Issue | Possible Solution |
|-------|-------------------|
| Voice not working | Ensure you're on Windows and speakers are on |
| Face not recognized | Add better images (front-facing, well-lit) |
| Can't write CSV | Close the CSV file in Excel and try again |
| App not launching | Run `streamlit run app.py` in an active environment |

---

## 📦 Requirements

Your `requirements.txt` includes:

```
streamlit
opencv-python
numpy
pandas
torch
facenet-pytorch
scikit-learn
pyttsx3
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 🧪 Technologies Used

| Component | Library |
|----------|---------|
| Face Recognition | `facenet-pytorch` |
| Webcam Feed | `opencv-python` |
| Data Handling | `numpy`, `pandas` |
| UI | `streamlit` |
| Voice Feedback | `pyttsx3` |
| Face Matching | `scikit-learn` cosine similarity |

---

## 🧩 Possible Extensions

- Admin dashboard for report generation
- Attendance heatmaps or visual analytics
- Google Sheets or cloud backup integration
- OTP/email-based authentication
- Mobile-friendly version or Android app

---
