import cv2

# Load the Haar Cascade Classifier for human detection


# Open the video file for reading (change 'video_file_path' to your video file)
video_file_path = r'C:/Users/user/Downloads/vid-2.mp4'
cap = cv2.VideoCapture(video_file_path)
cap.set(cv2.CAP_PROP_FPS, 120)
human_cascade = cv2.CascadeClassifier(r'C:\Users\user\OneDrive\Desktop\arc systemversion 2\haarcascade_fullbody.xml')


ret=True
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans in the frame
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Human Detection', frame)

    # Press 'p' to exit
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

