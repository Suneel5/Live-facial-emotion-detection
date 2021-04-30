import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

saved_model=load_model('fer_model.h5')
emotions = {0: 'Angry',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Neutral'}

def predict_emotion(img_array):
    #Predicts the emotion from the given image
    gray_img=cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_img, (48, 48))
    img = img / 225
    predicted_label = np.argmax(saved_model.predict(img.reshape(1, 48, 48, 1)), axis=-1)
    predicted_emotion = emotions[predicted_label[0]]
    return predicted_emotion

def emotion_detction(img):
    #Detcting face in the image so we can feed it to the  model for emotion prediction
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)
    # if faces != ():
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
        # sending only face part of image  for accurate prediction
        predicted_emotion = predict_emotion(img[y:y + h, x:x + w])
        cv2.putText(img, predicted_emotion, (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
    return img

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    #gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img=emotion_detction(frame)

    key = cv2.waitKey(1)
    cv2.imshow('frame', img)
    if key == ord('q'):
        break




