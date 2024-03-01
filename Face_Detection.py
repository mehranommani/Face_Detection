import cv2 as cv

# img = cv.imread('/Users/mehranommani/Downloads/read pic/cat_large.jpg')
# cv.imshow('Cat',img)

# Rescale frames to a certain size, to reduce computational load
def rescaleFrame(frame, scale=0.75):
    # Calculate the new width and height
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    # Create a tuple with the new dimensions
    dimentions = (width, height)

    # Resize the frame to the new dimensions
    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)


# Initialize video capture from the webcam
capture = cv.VideoCapture(1)
# Load the Haar cascade file for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')
while True:
    # Read the frame
    isTrue, frame = capture.read()
    
    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces_rec = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
    print(f'Number of faces found = {len(faces_rec)}')
    
    # Draw rectangles around the faces
    for (x, y, w, h) in faces_rec:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    # Display the frame
    cv.imshow('Video', frame)

    # Break the loop if 'd' is pressed
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release the video capture and destroy all OpenCV windows
capture.release()
cv.destroyAllWindows()