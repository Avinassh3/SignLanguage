from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
import math
import os

from matplotlib import pyplot as plt

model = load_model('FinalModel.h5')
class_labels={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26, 'space': 27}
labels=list(class_labels.keys())


# tesing with Test images provided from  competiton 

dirtest="D://Avinassh//major project//asl_alphabet_test"

for filee in os.listdir(dirtest):
    img=cv2.imread(dirtest+"//"+filee,0)
    img_gray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    img_gray=np.reshape(img_gray,[1,200,200,1])
    ar=model.predict(img_gray)
    
    ind=np.argmax(ar[0])

    print("for this :{}  image it as predicted this class: {}   label with  {}% ".format(filee[:3],labels[ind],ar[0][ind]*100))






# testing with other image 
img=cv2.imread('test.jpg',0)


img_gray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

img_gray=cv2.resize(img_gray,(200,200))

cv2.imshow('image',img_gray)
cv2.waitKey(0)
img_gray=np.reshape(img_gray,[1,200,200,1])

ar=model.predict(img_gray)

ind=np.argmax(ar[0])

print("for this image it as predicted this class: {}   label with  {}% ".format(labels[ind],ar[0][ind]*100))

print("actual label is B")
