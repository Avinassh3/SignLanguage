
# coding: utf-8
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
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


image_gen.flow_from_directory("D://Avinassh//major project//Dataset")


# In[51]:


import warnings
warnings.filterwarnings('ignore')


# In[60]:


train_image_gen=image_gen.flow_from_directory('D://Avinassh//major project//Dataset//asl_alphabet_test',batch_size=50,target_size=(200,200),class_mode="binary")
test_image_gen=image_gen.flow_from_directory('D://Avinassh//major project//Dataset//asl_alphabet_train',batch_size=50,target_size=(200,200),class_mode="binary")



# In[61]:


from keras.models import Sequential
from keras.layers import Activation,Dropout,Dense,Flatten,Conv2D,MaxPooling2D,BatchNormalization


# In[62]:


from keras.layers import Convolution2D,BatchNormalization,AveragePooling2D
input_shape=(200,200,3)
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


result=model.fit_generator(train_image_gen,epochs=25,steps_per_epoch=steps_for,validation_data=test_image_gen,verbose=1,validation_steps=validate)

model.save('FinalModel.h5') 

classindex=train_image_gen.class_indices
print(classindex)