# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:20:24 2022

@author: karan
"""

import os
import cv2


recognizer = cv2.face.LBPHFaceRecognizer_create()  
recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")

path='TestImage/'
correct,incorrect=0,0
correct_list,inc_list=[],[]

for image_name in os.listdir('TestImage'):
    id=int(image_name.split('.')[1])
    img=cv2.imread(path+image_name)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred=recognizer.predict(img)[0]
    
    if pred==id:
        correct+=1
        correct_list.append(image_name)
    else:
        incorrect+=1
        inc_list.append(image_name)
        

print('Number of correct instances=', correct)
print('Number of incorrect instances=', incorrect)
print('Accuracy=', correct/(correct+incorrect))
        