import cv2
import numpy as np
import os
from os import listdir
import sqlite3
from datetime import datetime

# Database setup
def init_database():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, name TEXT UNIQUE, created_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY, user_id INTEGER, timestamp TEXT, confidence REAL,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS training_stats
                 (id INTEGER PRIMARY KEY, algorithm TEXT, total_images INTEGER,
                  total_users INTEGER, training_date TEXT, accuracy REAL)''')
    conn.commit()
    return conn

# Check if dataset folder exists
if not os.path.exists('dataset'):
    print("ERROR: 'dataset' folder not found!")
    print("Please run dataset.py first to create training data.")
    exit(1)

# Check if dataset is empty
dataset_folders = [d for d in listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
if not dataset_folders:
    print("ERROR: 'dataset' folder is empty!")
    print("Please run dataset.py first to create training data.")
    exit(1)

# Initialize database
conn = init_database()
c = conn.cursor()

# Insert users into database
for person in dataset_folders:
    c.execute("INSERT OR IGNORE INTO users (name, created_date) VALUES (?, ?)",
              (person, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
conn.commit()

# Choose recognition algorithm (EigenFaces for better accuracy)
print("Available algorithms:")
print("1. LBPH (Local Binary Patterns Histograms) - Fast, good for small datasets")
print("2. EigenFaces - Better accuracy, slower")
print("3. FisherFaces - Best accuracy, needs multiple images per person")

choice = input("Choose algorithm (1-3) [default: 2]: ").strip()
if choice == '1':
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    algorithm_name = 'LBPH'
elif choice == '3':
    recognizer = cv2.face.FisherFaceRecognizer_create()
    algorithm_name = 'FisherFaces'
else:
    recognizer = cv2.face.EigenFaceRecognizer_create()
    algorithm_name = 'EigenFaces'

face_cascade = cv2.CascadeClassifier(os.path.join('Haarcascade .xml files', 'haarcascade_frontalface_default.xml'))

faces, labels = [], []

label_dict = {}  # ID to name mapping
person_id = 0
total_images = 0

for person in dataset_folders:
    label_dict[person_id] = person
    person_dir = os.path.join('dataset', person)
    person_files = [f for f in listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not person_files:
        print(f"WARNING: No images found for {person}, skipping...")
        continue

    person_images = 0
    for f in person_files:
        img_path = os.path.join(person_dir, f)
        img = cv2.imread(img_path, 0)
        if img is None:
            print(f"WARNING: Could not read {img_path}, skipping...")
            continue
        faces.append(img)
        labels.append(person_id)
        person_images += 1
        total_images += 1

    print(f"Loaded {person_images} images for {person}")
    person_id += 1

# Check if we have training data
if not faces:
    print("ERROR: No valid images found in dataset!")
    conn.close()
    exit(1)

print(f"\nTraining {algorithm_name} model with {total_images} images from {len(label_dict)} people...")

# Train the model
recognizer.train(faces, np.array(labels))

# Save model and labels
recognizer.save('trainer.yml')
np.save('labels.npy', label_dict)

# Save training statistics
c.execute("INSERT INTO training_stats (algorithm, total_images, total_users, training_date) VALUES (?, ?, ?, ?)",
          (algorithm_name, total_images, len(label_dict), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
conn.commit()

print(f"✅ Training complete! Trained on {total_images} images from {len(label_dict)} people using {algorithm_name}")
print("Model saved as 'trainer.yml'")

conn.close()