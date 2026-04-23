import cv2
import numpy as np
import os
from config import config
from logger import logger

def is_blurry(image, threshold=100):
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(laplacian_var < threshold), laplacian_var

def main():
    cam = cv2.VideoCapture(config['gui']['camera_index'])

    # Check if camera opened successfully
    if not cam.isOpened():
        logger.error(f"Cannot open camera at index {config['gui']['camera_index']}!")
        print(f"ERROR: Cannot open camera at index {config['gui']['camera_index']}!")
        return

    face_cascade = cv2.CascadeClassifier(os.path.normpath(config['paths']['haarcascade_frontalface']))

    # Check if cascade file loaded correctly
    if face_cascade.empty():
        logger.error("Could not load face cascade classifier!")
        print("ERROR: Could not load face cascade classifier!")
        return

    name = input("Apna naam enter karo: ")  # e.g. "Rahul"

    # Validate name
    if not name or name.strip() == "":
        logger.error("Name cannot be empty!")
        print("ERROR: Name cannot be empty!")
        return

    dataset_dir = config['paths']['dataset_dir']
    person_dir = os.path.join(dataset_dir, name.strip())
    os.makedirs(person_dir, exist_ok=True)
    count = 0
    images_per_person = config['training']['images_per_person']

    logger.info(f"Starting dataset collection for {name}")
    print(f"Starting dataset collection for {name}. Show your face to camera.")
    print("Press 'q' to stop early.")

    while count < images_per_person:
        ret, img = cam.read()
        
        if not ret:
            logger.error("Failed to grab frame from camera!")
            print("ERROR: Failed to grab frame from camera!")
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, config['recognition']['scale_factor'], config['recognition']['min_neighbors'])
        
        for (x,y,w,h) in faces:
            face_img = img[y:y+h, x:x+w]  # Use color image for blur check
            blurry, sharpness = is_blurry(face_img)
            
            if blurry:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)  # Red for blurry
                cv2.putText(img, f"Blurry ({sharpness:.1f})", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                continue
            
            face = gray[y:y+h, x:x+w]
            # Resize to standard size
            face = cv2.resize(face, (100, 100))
            img_path = os.path.join(person_dir, f"face_{count}.jpg")
            cv2.imwrite(img_path, face)
            count += 1
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  # Green for good
            logger.info(f"Saved image {count}/{images_per_person} for {name}")
        
        # Show progress
        cv2.putText(img, f"Collected: {count}/{images_per_person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Dataset', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Collection stopped by user.")
            print("Collection stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        logger.info(f"Dataset ready! Collected {count} images for {name}")
        print(f"Dataset ready! Collected {count} images for {name}")
    else:
        logger.error("No images were collected!")
        print("ERROR: No images were collected!")

if __name__ == "__main__":
    main()
