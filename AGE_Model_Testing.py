# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:54:16 2023

@author: User
"""

import numpy as np # linear algebra
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw 
from keras.models import load_model
import cv2

model = load_model('C:/Users/User/AGEmodel.h5')
IMG_WIDTH, IMG_HEIGHT = 96, 96
race_mapper = {0:'white', 1:'black',2: 'asian',3: 'indian', 4:'other'}
gender_mapper = {0:'male',1:'female'}

#title_font = ImageFont.load_default()

img = cv2.imread('../AGE/test6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('../AGE/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
n = len(faces)
print('Number of detected faces:', n)

if len(faces) > 0:
   for i, (x, y, w, h) in enumerate(faces):
 
      # To draw a rectangle in a face
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
      face = img[y:y + h, x:x + w]
      cv2.imwrite(f'face{i}.jpg', face)
      print(f"face{i}.jpg is saved")

cv2.imshow("image", img)

while 1:
    if cv2.waitKey(1)==13: 
        print("This is closing. Bbye!")
        break
cv2.destroyAllWindows()

for i in range(0,n):
    plt.figure(figsize=(10,10))
    face_img = np.array(Image.open(f"C:\\Users\\User\\OneDrive\\Documents\\Sem 6\\AGE\\face{i}.jpg").resize((IMG_WIDTH,IMG_HEIGHT)) ) / 255.
    predict = model.predict(np.array([face_img]))
    age = int(predict[0] * 100)
    race = race_mapper[np.argmax(predict[1])]
    gender = gender_mapper[np.argmax(predict[2])]
    plt.axis('off')
    plt.title('Predict %i, %s, %s' % (age, race, gender))
    plt.imshow(face_img)