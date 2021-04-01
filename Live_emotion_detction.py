import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

saved_model=load_model('fer_model.h5')
emotions = {0: 'Angry',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Neutral'}

def plot_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def predict_emotion(img_array):
    #Predicts the emotion from the given image

    img = cv2.resize(img_array, (48, 48))
    img = img / 225
    predicted_label = np.argmax(saved_model.predict(img.reshape(1, 48, 48, 1)), axis=-1)
    predicted_emotion = emotions[predicted_label[0]]
    predicted_emotion = f'{predicted_emotion} Face'
    return predicted_emotion

def emotion_detction(img):
    #Detcting face in the image so we can feed it to the  model for emotion prediction
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img)

    if faces != ():
        for x, y, w, h in faces:
            img_rec = cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            # sending only face part of image  for accurate prediction
            predicted_emotion = predict_emotion(img_rec[y:y + h, x:x + w])
            cv2.putText(img_rec, predicted_emotion, (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)
            plot_img(img_rec)
    else:
        plot_img(img)


cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotion_detction(gray_img)
    key=cv2.waitKey(1)

    if key == ord('q'):
        break
