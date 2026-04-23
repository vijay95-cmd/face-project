import cv2
import numpy as np
import os
import glob
import sqlite3
from datetime import datetime
from config import config
from logger import logger

def get_latest_model_files():
    model_pattern = config['paths']['model_file'].replace('.yml', '') + '_*.yml'
    labels_pattern = config['paths']['labels_file'].replace('.npy', '') + '_*.npy'

    model_files = glob.glob(model_pattern)
    labels_files = glob.glob(labels_pattern)

    if not model_files or not labels_files:
        # Fallback to default
        return config['paths']['model_file'], config['paths']['labels_file']

    latest_model = max(model_files, key=os.path.getctime)
    latest_labels = max(labels_files, key=os.path.getctime)

    return latest_model, latest_labels

def mark_attendance(name, confidence):
    """Mark attendance for a recognized person."""
    try:
        db_path = config['paths']['database_file']
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Get user id
        c.execute("SELECT id FROM users WHERE name = ?", (name,))
        user = c.fetchone()
        if user:
            user_id = user[0]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute("INSERT INTO attendance (user_id, timestamp, confidence) VALUES (?, ?, ?)",
                     (user_id, timestamp, confidence))
            conn.commit()
            logger.info(f"Attendance marked for {name} at {timestamp}")
        else:
            logger.warning(f"User {name} not found in database")
        
        conn.close()
    except Exception as e:
        logger.error(f"Error marking attendance for {name}: {e}")

def mark_attendance(name, confidence):
    """Mark attendance in database."""
    try:
        conn = sqlite3.connect(config['paths']['database_file'])
        c = conn.cursor()
        
        # Get user id
        c.execute("SELECT id FROM users WHERE name = ?", (name,))
        user = c.fetchone()
        if user:
            user_id = user[0]
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            c.execute("INSERT INTO attendance (user_id, timestamp, confidence) VALUES (?, ?, ?)",
                     (user_id, timestamp, confidence))
            conn.commit()
            logger.info(f"Attendance marked for {name}")
        
        conn.close()
    except Exception as e:
        logger.error(f"Failed to mark attendance for {name}: {e}")

# Check if trainer.yml exists
model_file, labels_file = get_latest_model_files()
if not os.path.exists(model_file):
    logger.error(f"{model_file} not found!")
    print(f"ERROR: {model_file} not found!")
    print("Please run train.py first to train the model.")
    exit(1)

# Check if labels.npy exists
if not os.path.exists(labels_file):
    logger.error(f"{labels_file} not found!")
    print(f"ERROR: {labels_file} not found!")
    print("Please run train.py first to generate labels.")
    exit(1)

# recognizer load
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_file)

# labels load
try:
    labels_dict = np.load(labels_file, allow_pickle=True).item()
except Exception as e:
    logger.error(f"Failed to load {labels_file}: {e}")
    print(f"ERROR: Failed to load {labels_file}: {e}")
    exit(1)

# face cascade
face_cascade = cv2.CascadeClassifier(os.path.normpath(config['paths']['haarcascade_frontalface']))

# Check if cascade file loaded correctly
if face_cascade.empty():
    logger.error("Could not load face cascade classifier!")
    print("ERROR: Could not load face cascade classifier!")
    exit(1)

# camera start
camera_index = config['gui']['camera_index']
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    logger.error(f"Cannot open camera at index {camera_index}!")
    print(f"ERROR: Cannot open camera at index {camera_index}!")
    exit(1)

logger.info("Face Recognition started. Press 'q' to exit.")
print("Face Recognition started. Press 'q' to exit.")

while True:
    ret, img = cap.read()
    
    if not ret:
        print("ERROR: Failed to grab frame from camera!")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, config['recognition']['scale_factor'], config['recognition']['min_neighbors'])

    recognized_names = set()  # To avoid duplicate attendance in same frame

    for (x, y, w, h) in faces:
        try:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            confidence_threshold = config['recognition']['confidence_threshold']
            if confidence < confidence_threshold:
                name = labels_dict.get(id_, "Unknown")
                if name != "Unknown" and name not in recognized_names:
                    mark_attendance(name, confidence)
                    recognized_names.add(name)
                    color = (0, 255, 0)  # Green for recognized
                else:
                    color = (0, 255, 255)  # Yellow for already recognized
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # rectangle draw
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

            # name show
            cv2.putText(img, f"{name} ({confidence:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            logger.error(f"ERROR during prediction: {e}")
            print(f"ERROR during prediction: {e}")
            continue

    # screen show
    cv2.imshow("Face Recognition", img)

    # exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
logger.info("Face Recognition closed.")
print("Face Recognition closed.")
