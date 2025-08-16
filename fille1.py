import cv2
from deepface import DeepFace
import numpy as np

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (face detection works better on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    # The scaleFactor and minNeighbors can be tuned for better performance
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        try:
            # Crop the face from the original color frame
            face_roi = frame[y:y + h, x:x + w]

            # Analyze the face for emotion using DeepFace
            # The 'actions' parameter specifies what to analyze. We only need 'emotion'.
            # 'enforce_detection=False' tells DeepFace to not run its own detector since we've already done it.
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # The result is a list of dictionaries, get the first one
            dominant_emotion = result[0]['dominant_emotion']

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the detected emotion text above the rectangle
            text = f"Emotion: {dominant_emotion.capitalize()}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            # DeepFace might not find an emotion in a small or blurry face crop
            print(f"Could not analyze face: {e}")
            
    # Display the final frame with annotations
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Application closed successfully. 👋")