import cv2
import numpy as np
import os

# Check if trainer.yml exists
if not os.path.exists('trainer.yml'):
    print("ERROR: trainer.yml not found!")
    print("Please run train.py first to train the model.")
    exit(1)

# Check if labels.npy exists
if not os.path.exists('labels.npy'):
    print("ERROR: labels.npy not found!")
    print("Please run train.py first to generate labels.")
    exit(1)

# recognizer load
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# labels load
try:
    labels_dict = np.load('labels.npy', allow_pickle=True).item()
except Exception as e:
    print(f"ERROR: Failed to load labels.npy: {e}")
    exit(1)

# face cascade
face_cascade = cv2.CascadeClassifier(os.path.join('Haarcascade .xml files', 'haarcascade_frontalface_default.xml'))

# Check if cascade file loaded correctly
if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier!")
    exit(1)

# camera start
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

print("Face Recognition started. Press 'q' to exit.")

while True:
    ret, img = cap.read()
    
    if not ret:
        print("ERROR: Failed to grab frame from camera!")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        try:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 100:
                name = labels_dict.get(id_, "Unknown")
            else:
                name = "Unknown"

            # rectangle draw
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            # name show
            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        except Exception as e:
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
print("Face Recognition closed.")