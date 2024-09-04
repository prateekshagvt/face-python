# face-python
import cv2 
import numpy as np 
import dlib 
import time 

# Connects to your computer's default camera 
cap = cv2.VideoCapture(0) 

# Detect the coordinates 
detector = dlib.get_frontal_face_detector() 

# Capture frames continuously 
while True: 
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    if not ret:
        break

    # Flip the frame horizontally (mirror image)
    frame = cv2.flip(frame, 1) 

    # RGB to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect faces
    faces = detector(gray) 

    # Iterator to count faces 
    i = 0
    for face in faces: 
        # Get the coordinates of faces 
        x, y = face.left(), face.top() 
        x1, y1 = face.right(), face.bottom() 

        # Draw rectangle around faces
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2) 

        # Increment face count
        i = i + 1

        # Display the box and faces count
        cv2.putText(frame, 'face num: ' + str(i), (x - 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

        print(f"Face coordinates: {face}, Face count: {i}") 

    # Introduce a delay (optional)
    # time.sleep(0.2) # Delay for 0.2 seconds

    # Display the resulting frame 
    cv2.imshow('frame', frame) 

    # This command lets us quit with the "q" button on a keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Release the capture and destroy the windows 
cap.release() 
cv2.destroyAllWindows()
