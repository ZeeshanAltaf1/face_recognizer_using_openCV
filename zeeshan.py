import cv2
import os

person_name = "zeeshan"   
path = "dataset/" + person_name

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{path}/{count}.jpg", face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Collecting Images", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= 100:
        break

cap.release()
cv2.destroyAllWindows()
