import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# read image
# img = cv2.imread('wingTeam.jpg')
webcam = cv2.VideoCapture(0)


while True:
    successful_frame_read, frame = webcam.read()

    # convert to grey scale
    greyScaleImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faceCoordinates = trained_face_data.detectMultiScale(greyScaleImage)

    # create tuple for face coordinates
    for (x, h, y, w) in faceCoordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (randrange(256), randrange(256), randrange(256)), 6)

    # show image
    cv2.imshow("Face Detection", frame)

    # prevents window from closing instantly
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
