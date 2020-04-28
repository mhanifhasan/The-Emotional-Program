import os
import cv2
import numpy as np
import threading
import winsound
import pygame
#from getkey import getkey, keys
from keras.models import model_from_json
from keras.preprocessing import image


winsound.PlaySound("filename", winsound.SND_ASYNC | winsound.SND_ALIAS )


print("\n\t\t Welcome to THE EMOTIONAL PROGRAM [TEP]\n")
predicted_emotion = []

def Key_Grabber():
  pygame.mixer.init()
  while(True):
      key = input("Press Enter to play song according to your expression \nOnce songs are playing Press S or . to stop the songs\n")
      if((key =='s' or key == 'S') or (key =='.')):
          key='s'
      else:
          key='k'
      
      
      if(key == 'K' or key == 'k'):
          
          if(predicted_emotion == 'happy'):
              print("Happy Song")
              pygame.mixer.music.load("Bruno Mars - 24K Magic (Happy Music).mp3")
              pygame.mixer.music.play()
          if(predicted_emotion == 'sad'):
              print("Sad Song")
              pygame.mixer.music.load("Serhat Durmus - Yalan (Sad Music).mp3")
              pygame.mixer.music.play()
          if(predicted_emotion == 'angry'):
              print("Angry Song")
              pygame.mixer.music.load("DOOM (2016) - BFG Division (Angry Music).mp3")
              pygame.mixer.music.play()
          if(predicted_emotion == 'fear'):
              print("Fear Music")
              pygame.mixer.music.load("Scary horror music (Fear Music) .mp3")
              pygame.mixer.music.play()
          if(predicted_emotion == 'neutral'):
              print("Neutral Song")
              pygame.mixer.music.load("My Love Is Winter (Natural Music).mp3")
              pygame.mixer.music.play()
          
          
      if(key == 's' or key == 'S'):
          print("Music Stopped!")
          pygame.mixer.music.stop()  
  

thread1 = threading.Thread(target = Key_Grabber, args = ())
thread1.start()  

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('training with 3rd layer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', '', 'fear', 'happy', 'sad', '', 'neutral')
        predicted_emotion = emotions[max_index]
        
        #print(predicted_emotion)
        
        
        
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows