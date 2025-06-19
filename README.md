You're absolutely right — if the preview on GitHub isn't rendering properly, it's likely due to incorrect code block formatting or extra syntax that's misinterpreted by Markdown. Let's **fix that now** to ensure it looks perfect in GitHub's README preview.

Below is the **clean, fully readable `README.md`** in proper GitHub markdown format — with fixed code blocks and preview-safe formatting:

---

### ✅ FINAL `README.md` (GitHub-Preview Friendly)

```markdown
# 🧠 AI-Powered Attendance System using Face Recognition

A real-time facial recognition system to automate attendance in schools, colleges, or offices. Supports both **students (per class period)** and **teachers (once per day)**. Includes **voice confirmation (Windows)** and **CSV attendance logs**.

> Built using Python, Streamlit, and FaceNet. Ensures non-duplicate, secure, and organized attendance.

---

## ✨ Features

- 📸 Real-time face recognition via webcam
- 🧑‍🎓 Student attendance by class period
- 👨‍🏫 Teacher attendance once per day
- ❗ Prevents duplicate entries
- 🔊 Voice confirmation (Windows only)
- 🗃 Attendance logs auto-saved as CSV
- 📅 Manual or CSV-based schedule input
- 🧠 Face embeddings using `facenet-pytorch`

---

## 📂 Project Structure

```

ai-attendance-system/
├── app.py                  # Main Streamlit app
├── generate\_embeddings.py  # Generate face encodings
├── dataset/                # Registered face images (1 folder per person)
│   └── John Doe/           # e.g., contains 1.jpg to 30.jpg
├── embeddings.npy          # Saved face embeddings
├── attendance\_logs/        # Daily CSV attendance logs
├── requirements.txt        # Dependencies list
└── README.md               # This file

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-attendance-system.git
cd ai-attendance-system
````

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

## 👤 Add Students or Teachers

1. Create a folder inside `dataset/` with the person's **full name**
   Example: `dataset/Krishnagopal Jay/`
2. Add **30+ clear face images** (1 face per image)
3. Run the embedding generator:

```bash
python generate_embeddings.py
```

This will create/update `embeddings.npy`.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then choose:

* `Student` → For subject-wise period attendance
* `Teacher` → For once-a-day attendance

---

## 🗓 Class Period Schedule

### Option 1: Manual Entry in App

Use the Streamlit UI to enter subject names, start and end times.

### Option 2: Upload CSV

**CSV format example:**

```csv
Subject,Start,End
Math,09:00,09:45
Physics,10:00,10:45
Break,10:45,11:00
```

Upload this file inside the app when prompted.

---

## 🔊 Voice Confirmation (Windows Only)

When a face is recognized, the app says:

> "Attendance marked for Krishnagopal Jay"

Make sure:

* Your speakers are ON
* `pyttsx3` is installed (already in `requirements.txt`)
* You're on **Windows**

---

## 🗃 Attendance Logs

CSV logs are saved automatically:

```
attendance_logs/
├── student_attendance_YYYY-MM-DD.csv
├── teacher_attendance_YYYY-MM-DD.csv
```

Each file logs name, time, subject, and role.

---

## 📌 System Behavior Rules

* ✅ One face per image (no group photos)
* 🔁 No duplicate entries per period/day
* ⏰ Student attendance is time-bound to class periods
* 🔊 Voice feedback works only on Windows
* 💡 Consistent face images = better recognition

---

## 🧰 Troubleshooting

| Issue                   | Solution                                           |
| ----------------------- | -------------------------------------------------- |
| Voice not working       | Ensure `pyttsx3` is installed and speakers are on  |
| Face not recognized     | Improve image clarity, lighting, and angles        |
| CSV not writing         | Close the CSV file if open in Excel                |
| Streamlit not launching | Activate venv and try `streamlit run app.py` again |

---

## 📦 Requirements

File: `requirements.txt`

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

Install via:

```bash
pip install -r requirements.txt
```

---

## 📜 License

Licensed under the **MIT License**.
Use ethically and with user consent when collecting facial data.

---

