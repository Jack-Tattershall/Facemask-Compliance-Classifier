import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle #save data
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


""" NOTES:
    Wont need if internal images used
    Also, might want to use colour Vs. grayscale as this is a key factor is classification task
    if colour, will affect reshape on feature array

"""
DATA_DIR = ""
CATEGORIES = ["Mask", "No Mask"]
IMG_SIZE = 50

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category) #Path to cats or dogs directory
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category) #Path to cats or dogs directory
       
        class_num = CATEGORIES.index(category) #numerical value of class based on index in CATGORIES

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = [] #feature set
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

#features has to be an np.array due to how Keras works
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #end 1 because greyscale 1D; 3 if colour, -1 any structure;

#Save features
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

#Save labels
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#To load data back in, not have to recreate training data
# X = pickle.load(open("X.pickle", "rb"))
# y = pickle.load(open("y.pickle", "rb"))

X = X/255.0 #normalise

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)
