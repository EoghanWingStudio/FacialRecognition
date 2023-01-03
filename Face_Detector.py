import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# read image
img = cv2.imread('wingTeam.jpg')

# convert to grey scale
greyScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
faceCoordinates = trained_face_data.detectMultiScale(greyScaleImage)

# create tuple for face coordinates
for (x, h, y, w) in faceCoordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h),
                  (randrange(256), randrange(256), randrange(256)), 6)

# show image
cv2.imshow("Face Detection", img)

# prevents window from closing instantly
cv2.waitKey()

print("Code Complete")
