from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
import math


model = load_model('FinalModel.h5')
class_labels={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26, 'space': 27}
labels=list(class_labels.keys())
img = cv2.imread('test.jpg')



img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3])
cv2.imwrite("modified.jpeg",img)


ar=model.predict(img)
print(" prediciton percentages ")
for i in range(0,len(ar[0])):
    print(labels[i]+':'+ str(math.floor(ar[0][i]*100)),end="%\n")




