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

face_cascade = cv2.CascadeClassifier(os.path.join('Haarcascade .xml files', 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join('Haarcascade .xml files', 'haarcascade_eye.xml'))

# Check if cascade files loaded correctly
if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier!")
    exit(1)

if eye_cascade.empty():
    print("WARNING: Could not load eye cascade classifier! Eye detection will be disabled.")
    eye_cascade = None

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels = np.load('labels.npy', allow_pickle=True).item()

cap = cv2.VideoCapture(0)  # Webcam
#open webcame
if not cap.isOpened():
    print("cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("failed to grab frame")
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)  # Blue box
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Recognize face
        try:
            id_, conf = recognizer.predict(roi_gray)
            print(f"Prediction: id={id_}, confidence={conf:.2f}")  # Debug output
            
            if conf <= 70:  # Increased threshold for better recognition
                name = labels.get(id_, "Unknown")
                cv2.putText(img, f"{name} ({conf:.0f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.putText(img, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,127,255), 2)  # Red eyes
    
    cv2.imshow('Face Detection and Recognition', img)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC dabaao stop
        break

cap.release()
cv2.destroyAllWindows()
