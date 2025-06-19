import os
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load face detector and FaceNet model
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=40, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Directory where student images are stored
dataset_path = 'dataset'  # Each subfolder is a student name
embedding_dict = {}

print("🔍 Generating face embeddings from dataset...")

# Loop through each student
for student_name in os.listdir(dataset_path):
    student_folder = os.path.join(dataset_path, student_name)
    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)

        # Load BGR image using OpenCV
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Detect face
        face_tensor = mtcnn(img_pil)
        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy()
                embeddings.append(embedding)

    # Average embeddings
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        embedding_dict[student_name] = avg_embedding
        print(f"✅ Processed: {student_name} ({len(embeddings)} images)")

# Save to .npy file
np.save('embeddings.npy', embedding_dict)
print("✅ All embeddings saved to 'embeddings.npy'.")