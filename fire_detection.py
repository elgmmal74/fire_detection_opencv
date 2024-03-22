import cv2
import pygame

# Load the fire detection classifier file
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

# Set the alarm sound file
alarm_sound_file = 'Alarm_Sound.mp3'

# Initialize the pygame library for sound playback
pygame.mixer.init()

# Use the camera (number 1) as the video source
cap = cv2.VideoCapture(1)

# Variable to track the state of the alarm
is_alarm_on = False

while True:
    # Read the frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better detection performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect fires in the grayscale frame
    fires = fire_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in fires:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Fire', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if not is_alarm_on:
            pygame.mixer.music.load(alarm_sound_file)
            pygame.mixer.music.play()
            is_alarm_on = True

    # Display the processed frame
    cv2.imshow('Fire Detection', frame)

    # Wait for the 'q' key press to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
