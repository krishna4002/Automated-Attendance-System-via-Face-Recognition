import cv2
import numpy as np
import os
from datetime import datetime
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer

# Load FaceNet model
embedder = FaceNet()
l2_normalizer = Normalizer('l2')

# Load precomputed embeddings
embeddings = np.load('embeddings/embeddings.npy')
names = np.load('embeddings/names.npy')

def get_embedding(face_img):
    # FaceNet handles preprocessing internally
    embedding = embedder.embeddings([face_img])[0]
    return l2_normalizer.transform([embedding])[0]

def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    file_path = f"attendance/{date_str}.csv"
    os.makedirs("attendance", exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("Name,Time,Date\n")

    with open(file_path, 'r+') as f:
        lines = f.readlines()
        recorded_names = [line.split(',')[0] for line in lines[1:]]
        if name not in recorded_names:
            time_now = datetime.now().strftime("%H:%M:%S")
            f.write(f"{name},{time_now},{date_str}\n")
            print(f"✅ Attendance marked for: {name}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detected_faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32') / 255.0

        embedding = get_embedding(face_img)

        distances = [cosine_similarity(embedding, emb) for emb in embeddings]
        idx = np.argmax(distances)
        max_sim = distances[idx]

        if max_sim > 0.75:
            name = names[idx]
            mark_attendance(name)
            label = f"{name} ({round(max_sim, 2)})"
        else:
            label = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
