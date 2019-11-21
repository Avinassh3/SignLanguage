
# coding: utf-8

# In[1]:


import os
from os import path
import shutil
import sys
import cv2
import time
from PIL import Image
import numpy as np
import pickle
# In[5]:


#input of this function is String:"j1230" output of this function is "1230"

class_labels={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26, 'space': 27}
def GivenStringReturnNumberInit(strr):
    ind=0
    for i in range(0,len(strr)):
        if strr[i].isdigit():
            ind=i
            break
    return strr[ind:]
        
def GivenStringReturnLabelInit(strr):
    ind=0
    for i in range(0,len(strr)):
        if strr[i].isdigit():
            ind=i
            break
    return strr[:ind]        

#Current folder : where Dataset is present
dirc="D://Avinassh//major project//data//asl_alphabet_train"

#New Folder in which we store grayscale  train images 
dirtest="D://Avinassh//major project//Dataset//asl_alphabet_test"


#New Folder in which we store grayscale  test images 
dirtrain="D://Avinassh//major project//Dataset//asl_alphabet_train"

#Variables used to store Total count of images
totalCount=0
splitCount=0

X_TrainImages=[]
X_TrainLabels=[]
Y_TestImages=[]
Y_TestLabels=[]
totaltime=time.clock()
for foldername in os.listdir("D://Avinassh//major project//data//asl_alphabet_train"):
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
    folcount=time.clock()
    for filename in os.listdir(dirc+"//"+foldername):
        totalCount+=1
        val=GivenStringReturnNumberInit(filename[:-4])
        try:
            img=cv2.imread(dirc+"//"+foldername+"//"+filename,0)
            img_gray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            
            val=int(val)
            if val%3==0:
                count+=1
                X_TrainImages.append(img_gray)
                X_TrainLabels.append(class_labels[GivenStringReturnLabelInit(filename[:-4])])
            else:
                Y_TestImages.append(img_gray)
                Y_TestLabels.append(class_labels[GivenStringReturnLabelInit(filename[:-4])])
        except Exception as err :
            print(err)
    splitCount+=count
    print("Completed convertion and spliting of "+foldername+"  sign"+"\t in : "+str(time.clock()-folcount)+"secs")
print(splitCount/totalCount)            

print("completed coversion in {} mins".format(str((time.clock()-totaltime)/60)))



# coding: utf-8
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# In[48]:


from keras.preprocessing.image import ImageDataGenerator


# In[49]:


image_gen=ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True
                        )


# In[50]:



import warnings
warnings.filterwarnings('ignore')




# In[61]:


from keras.models import Sequential
from keras.layers import Activation,Dropout,Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization


# In[62]:


from keras.layers import Convolution2D,BatchNormalization,AveragePooling2D
input_shape=(200,200,1)
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.50))

model.add(Dense(28))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[64]:


model.summary()


# In[65]:


steps_for=56000//50
validate=28000//50
x=np.array(X_TrainImages)
y=np.array(X_TrainLabels)

x=x.reshape(-1,200,200,1)

result=model.fit(x,y,epochs=5)


model.save("FinalModel.h5")


