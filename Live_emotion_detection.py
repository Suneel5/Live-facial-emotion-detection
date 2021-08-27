import cv2
import numpy as np
import PIL
from tensorflow.keras.models import load_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #to use cpu only

saved_model=load_model('fer_model.h5')
emotions = {0: 'Angry',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Neutral'}

def predict_emotion(img_array):
    #Predicts the emotion class from the given image array

    gray_img=cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_img, (48, 48))
    img = img / 225
    predicted_label = np.argmax(saved_model.predict(img.reshape(1, 48, 48, 1)), axis=-1)
    predicted_emotion = emotions[predicted_label[0]]
    return predicted_emotion

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
        # feeding only face part of image to the CNN model for accurate prediction
        predicted_emotion = predict_emotion(img[y:y + h, x:x + w])
        cv2.putText(img, predicted_emotion, (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
        emoji = PIL.Image.open(f'emoji/{predicted_emotion}.png')
        emoji=emoji.resize((100,100))  
        img=PIL.Image.fromarray(img)
        img.paste(emoji,(x,y-100,x+100,y))
        img=np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    key = cv2.waitKey(1)
    cv2.imshow('Facial Emotion recognition- Press q to exit!', img)

    #press q to exit
    if key == ord('q'):
        break



