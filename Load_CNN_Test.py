# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:05:28 2019

@author: JAVEN
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:52:13 2019

@author: Javen
"""

# later...
from keras.models import model_from_json
import h5py.h5
# load json and create model
json_file = open('Fortune_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Fortune_Model.h5")

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:38:24 2019

@author: Javen
"""

import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [1, 1, 1])

import matplotlib.image as mpimg

pic = cv2.imread('../Ai_fortune_teller/test_9_2.jpg')  
pic = cv2.resize(pic,(300,250), interpolation=cv2.INTER_CUBIC)
print(pic.shape)
test_image = image.img_to_array(pic)
test_image=test_image
print(test_image.shape)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image.shape)
x_test_image4D=test_image.reshape(test_image.shape[0],300,250,3).astype('float32')
x_test_image4D=x_test_image4D/255
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
prediction = loaded_model.predict_classes(x_test_image4D)
print(prediction)



     
#keras.datasets.mnist.load_data()
#(x_Train,y_Train),(x_Test,y_Test)=keras.datasets.mnist.load_data()

#first_train_img = np.reshape(test_image, (28, 28))
#plt.matshow(first_train_img, cmap = plt.get_cmap('gray'))
#plt.show()



#x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
#x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')


#x_Train4D_normalize=x_Train4D/255
#x_Test4D_normalize=x_Test4D/255

#將類別做Onehot encoding
#y_TrainOneHot=np_utils.to_categorical(y_Train)
#y_TestOneHot=np_utils.to_categorical(y_Test)




# evaluate loaded model on test data
#loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#prediction=loaded_model.predict_classes(x_Test4D_normalize[:1])

#print(prediction[:1])
#print(y_Test[:1])
#irst_test_img = np.reshape(test_image, (28, 28))
#plt.matshow(first_test_img, cmap = plt.get_cmap('gray'))
#plt.show()
#score = loaded_model.evaluate(x_Test4D_normalize, y_TestOneHot, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
