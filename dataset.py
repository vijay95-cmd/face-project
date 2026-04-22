import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

face_cascade = cv2.CascadeClassifier(os.path.join('Haarcascade .xml files', 'haarcascade_frontalface_default.xml'))

# Check if cascade file loaded correctly
if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier!")
    exit(1)

name = input("Apna naam enter karo: ")  # e.g. "Rahul"

# Validate name
if not name or name.strip() == "":
    print("ERROR: Name cannot be empty!")
    exit(1)

os.makedirs(os.path.join("dataset", name), exist_ok=True)
count = 0

print(f"Starting dataset collection for {name}. Show your face to camera.")
print("Press 'q' to stop early.")

while count < 30:  # 30 photos lo
    ret, img = cam.read()
    
    if not ret:
        print("ERROR: Failed to grab frame from camera!")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/{name}/face_{count}.jpg", face)
        count += 1
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    
    # Show progress
    cv2.putText(img, f"Collected: {count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Dataset', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Collection stopped by user.")
        break

cam.release()
cv2.destroyAllWindows()

if count > 0:
    print(f"Dataset ready! Collected {count} images for {name}")
else:
    print("ERROR: No images were collected!")