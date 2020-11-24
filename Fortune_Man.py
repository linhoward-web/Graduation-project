
#pip install opencv-python
import cv2
import glob
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
import pandas as pd # data processing, 
import os
from keras.utils import np_utils
#from numpy import random as np_random
import numpy as np
#numpy補充
#np.random.bit_generator = np.random._bit_generator
files = glob.glob("../Ai_fortune_teller/Male/*/*.jpg")
files = [i.replace("\\","/") for i in files]
labels = [i.split('/')[4] for i in files]
for label in files:
    print(label)
files, labels = sklearn.utils.shuffle(files, labels)


labels_all = os.listdir("../Ai_fortune_teller/Male/")
N_CLASSES = len(labels_all)
label_encoder = LabelEncoder()
label_encoder.fit(labels_all)

x_Trains=[]

y_Labels=[]
y_Trains=[]
print("======================")
count=0
for label in labels_all:
    y_Labels.append(label)    
    path='../Ai_fortune_teller/Male/'+label+'/*.jpg'
    for file in glob.glob(path): 
        
        img = cv2.imread(file)
        img=cv2.resize(img,(300,250), interpolation=cv2.INTER_CUBIC)
        #print(img.shape)
        x_Trains.append(image.img_to_array(img))
        y_Trains.append(count)
    count+=1
    
print("y_Labels:",y_Labels)
y_Trains=np.array(y_Trains)
x_Trains=np.array(x_Trains)
print(x_Trains.shape)
print(y_Trains.shape)
x_Trains4D=x_Trains.reshape(x_Trains.shape[0],300,250,3).astype('float32')
x_Train4D_normalize=x_Trains4D/255
 #將類別做Onehot encoding

y_TrainsOneHot=np_utils.to_categorical(y_Trains)
print(y_Trains)
print(y_TrainsOneHot)
   
print("size",len(y_Trains))
#將類別做Onehot encoding
print("#將類別做Onehot encoding")
y_TrainsOneHot=np_utils.to_categorical(y_Trains)
print(y_Trains)
print(y_TrainsOneHot)


#Create CNN Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model=Sequential()

#Filgers=16, Kernel size=(5,5), Padding='same'
model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(300,250,3),
                 activation='relu'))
#MaxPooling size=(2,2)
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(300,250,3),
                 activation='relu'))
#MaxPooling size=(2,2)
model.add(MaxPooling2D(pool_size=(3,3)))
#Drop some neurons to avoid overfitting
#model.add(Dropout(0.25))
#Flatten
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(22,activation='softmax')) #Dense(前面數字為class數目)

#Training Model
#Training Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainsOneHot,                        
                        shuffle=True,
                        validation_split=0.01,
                        epochs=500,
                        batch_size=150,
                        verbose=2)
# CNN model to JSON
model_json = model.to_json()
with open("Fortune_Model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Fortune_Model.h5")
print("Saved model to disk")
