import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

dataset_path = "/Downloads/face_recognizer_using_openCV/dataset" #your data path
faces = []
labels = []
names = {}
label_id = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    names[label_id] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.write("trained.yml")

print("Training Completed!")
print("Saved as trained.yml")
