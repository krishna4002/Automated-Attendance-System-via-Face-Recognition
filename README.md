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
├── app.py                  # Main Streamlit app (UI and logic)
├── generate_embeddings.py  # Script to generate face embeddings from images
├── dataset/                # Folders of registered users (1 per person)
│   └── John Doe/           # Example: contains 1.jpg, 2.jpg, ..., 30.jpg
├── embeddings.npy          # Face embeddings generated from dataset
├── attendance_logs/        # Auto-generated daily CSV attendance logs
│   ├── student_attendance_YYYY-MM-DD.csv
│   └── teacher_attendance_YYYY-MM-DD.csv
├── requirements.txt        # Python dependencies
└── README.md               # This file
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

## 👤 Add Students or Teachers

1. Inside `dataset/`, create a folder named after the person  
   Example: `dataset/Krishnagopal Jay/`

2. Add at least **30 clear images** of the person’s face (1 face per image)

3. Run:

```bash
python generate_embeddings.py
```

This will generate `embeddings.npy` for face recognition.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Select:
- `Student` → Period-wise attendance
- `Teacher` → Once-daily attendance

---

## 🗓 Class Period Setup

### ✅ Option 1: Manual Input
Enter periods and timings via the UI.

### 📁 Option 2: Upload CSV

Example format:

```csv
Subject,Start,End
Math,09:00,09:45
Physics,10:00,10:45
Break,10:45,11:00
```

Upload this file inside the app.

---

## 🔊 Voice Confirmation (Windows Only)

When a face is recognized, the app will say:

> "Attendance marked for Krishnagopal Jay"

Make sure:
- Your speakers are ON
- You're on **Windows**
- `pyttsx3` is installed (already in `requirements.txt`)

---

## 🗃 Attendance Logs

CSV logs are automatically saved in `attendance_logs/`:

```
attendance_logs/
├── student_attendance_YYYY-MM-DD.csv
├── teacher_attendance_YYYY-MM-DD.csv
```

Each record includes name, time, subject, and role.

---

## 📌 System Behavior

- ✅ One face per image
- 🔁 No duplicate entries per period/day
- ⏰ Student attendance only within valid time range
- 🧠 Clear images = higher recognition accuracy
- 🔊 Voice confirmation available only on Windows

---

## 🧰 Troubleshooting

| Issue                    | Solution |
|--------------------------|----------|
| Voice not working        | Ensure `pyttsx3` is installed and speakers are active |
| Face not recognized      | Use better image quality and lighting |
| CSV write error          | Close the CSV file in Excel or other programs |
| Streamlit not launching  | Ensure virtual env is activated and run `streamlit run app.py` |

---

## 📦 Requirements

These are included in `requirements.txt`:

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

Install with:

```bash
pip install -r requirements.txt
```

---

## 📜 License

Licensed under the **MIT License**.  
Use responsibly and with user consent when collecting facial data.

---

## 🙋‍♂️ Need Help?

- Open an [Issue](https://github.com/your-username/ai-attendance-system/issues)
- Contact: your.email@example.com

---

## 🙌 Credits

- 👤 Facial Recognition: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- 🧠 Voice Engine: [pyttsx3](https://pypi.org/project/pyttsx3/)
- 📱 UI: [Streamlit](https://streamlit.io)

---

## 📹 Demo GIF (Optional)

If available, add a preview like this:

```markdown
![Demo Preview](demo.gif)
```
