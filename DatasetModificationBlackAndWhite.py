
# coding: utf-8

# In[1]:


import os
from os import path
import shutil
import sys
import cv2

import numpy as np
# In[5]:


#input of this function is String:"j1230" output of this function is "1230"
def GivenStringReturnNumberInit(strr):
    ind=0
    for i in range(0,len(strr)):
        if strr[i].isdigit():
            ind=i
            break
    return strr[ind:]
        

#Current folder : where Dataset is present
dirc="D://Avinassh//major project//data//asl_alphabet_train"

#New Folder in which we store grayscale  train images 
dirtest="D://Avinassh//major project//Dataset//asl_alphabet_test"


#New Folder in which we store grayscale  test images 
dirtrain="D://Avinassh//major project//Dataset//asl_alphabet_train"

#Variables used to store Total count of images
totalCount=0
splitCount=0

for foldername in os.listdir(dirc):
    count=0
    movetra=dirtrain+"//"+foldername
    movedir=dirtest+"//"+foldername
    print("Started to convert and split of "+foldername+"  sign")
    if path.exists(movedir)==False:
        os.mkdir(movedir)
        print("Directory created  "+foldername)
    if path.exists(movetra)==False:
        os.mkdir(movetra)
        print("Directory created  "+movetra)
    for filename in os.listdir(dirc+"//"+foldername):
        totalCount+=1
        val=GivenStringReturnNumberInit(filename[:-4])
        try:
            img=cv2.imread(dirc+"//"+foldername+"//"+filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray= np.reshape(img_gray,[200,200,3])
            val=int(val)
            if val%3==0:
                count+=1
                cv2.imwrite(os.path.join(movedir,filename),img_gray)
            else:
                cv2.imwrite(os.path.join(movetra,filename),img_gray)
        except Exception as err :
            print(err)
    splitCount+=count
    print("Completed convertion and spliting of "+foldername+"  sign")
print(splitCount/totalCount)            

