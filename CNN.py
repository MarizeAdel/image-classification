# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:18:23 2022

@author: Marize
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Read Dataset

path=os.path.join(r"E:\study\CNN\DATA\mnist_png\training")
labels=os.listdir(path)
imgs=[]
lbls=[]
all_imgs=[]
for lbl in labels:
    path_with_lbl=os.path.join(path,lbl)
    imgs=os.listdir(path_with_lbl)
    for img in imgs:
        all_imgs.append(os.path.join(path_with_lbl,img))
        lbls.append(int(lbl))

# Preproccesing Datset

imgs=[]
dim=(28,28)
for img in all_imgs:
    image=cv2.imread(img,cv2.IMREAD_COLOR)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(im_rgb, dim, interpolation = cv2.INTER_AREA)
    imgs.append(image)

imgs=np.array(imgs)/255
lbls=np.array(lbls)


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(imgs,lbls,test_size=0.4,stratify=lbls)
val_x,test_x,val_y,test_y=train_test_split(test_x,test_y,test_size=0.5,stratify=test_y)


model = Sequential()

#add model layers
model.add(Conv2D(32,(3,3),input_shape=(28,28,3), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#declare our fully connected layers ( here we specify 5120 nodes, each activated by a ReLU function). 
model.add(Dense(512, activation='relu'))
#The second is our soft-max classification, or output layer, which is the size of the number of our classes (10 classes)
model.add(Dense(10, activation='softmax'))
model.summary()
#compile model using accuracy to measure model performance
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=['accuracy'])

history = model.fit(train_x,train_y,validation_data=(val_x,val_y),epochs=10)

test_loss_digit, test_acc_digit = model.evaluate(test_x,test_y)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Predicting the labels-DIGIT
y_predict = model.predict(test_x)
y_predict=np.argmax(y_predict, axis=1) # Here we get the index of maximum value in the encoded vector
y_test_digit_eval=np.argmax(test_x, axis=1)
#Confusion matrix for Digit MNIST

con_mat=confusion_matrix(test_y,y_predict)
plt.style.use('seaborn-deep')
plt.figure(figsize=(10,10))
sns.heatmap(con_mat,annot=True,annot_kws={'size': 15},linewidths=0.5,fmt="d")
plt.title('True or False predicted digit MNIST\n',fontweight='bold',fontsize=15)
plt.show()


#---
