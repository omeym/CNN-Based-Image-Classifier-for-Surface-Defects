import tensorflow as tf

from keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import cv2
import os
import glob
from keras.callbacks.callbacks import ModelCheckpoint
#from tensorflow.keras.callbacks import ModelCheckpoint

#VGG19
#def create_model():
#    model = sequential([
#    conv2d(64, 3, padding='same', activation='relu', input_shape=(200, 200 ,3)),
#    conv2d(64, 3, padding='same', activation='relu'),
#    maxpooling2d(),
#    conv2d(128, 3, padding='same', activation='relu'),
#    conv2d(128, 3, padding='same', activation='relu'),
#    maxpooling2d(),
#    conv2d(256, 3, padding='same', activation='relu'),
#    conv2d(256, 3, padding='same', activation='relu'),
#    conv2d(256, 3, padding='same', activation='relu'),
#    conv2d(256, 3, padding='same', activation='relu'),
#    maxpooling2d(),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    maxpooling2d(),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    conv2d(512, 3, padding='same', activation='relu'),
#    maxpooling2d(),
#    flatten(),
#    dense(4096, activation='relu'),
#    dense(4096, activation='relu'),
#    dense(1000, activation='relu'),
#    dense(6, activation= 'softmax')])

#    model.compile(optimizer='adam',
#              loss= 'categorical_crossentropy',
#              metrics=['categorical_accuracy'])
#    return model

#AlexNet
def create_model():

    model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(6, activation= 'softmax')])

    model.compile(optimizer='adam',
                  loss= 'categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


class_labels = np.array(['Cr', 'In', 'Pa', 'Ps', 'Rs', 'Sc'])


#Change it to appropriate path or just use \Training Data\ in your local directory
data_dir = 'D:\Work\Academics\AME 505-Engineering Information Modelling\Project\CNN Implementation_Val Split\Dataset'

#give validation split here
val_split = 0.2

dataset_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip= True, vertical_flip= True, validation_split= val_split)


data_Cr_dir = os.path.join(data_dir, 'Cr')  # directory with our Cr defect pictures
data_In_dir = os.path.join(data_dir, 'In')  # directory with our In defect pictures
data_Pa_dir = os.path.join(data_dir, 'Pa')  # directory with our Pa defect pictures
data_Ps_dir = os.path.join(data_dir, 'Ps')  # directory with our Ps defect pictures
data_Rs_dir = os.path.join(data_dir, 'Rs')  # directory with our Rs pictures
data_Sc_dir = os.path.join(data_dir, 'Sc')  # directory with our Sc defect pictures


batch_size_train = 200
batch_size_test = 100
epochs = 1
IMG_HEIGHT = 200
IMG_WIDTH = 200



train_data_gen = dataset_image_generator.flow_from_directory(batch_size=batch_size_train, directory= data_dir, subset = "training" ,shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode= 'categorical')
val_data_gen = dataset_image_generator.flow_from_directory(batch_size = batch_size_test, directory=data_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='categorical', subset = "validation")


model = create_model()

#Save Model
filepath = "D:/Work/Academics/AME 505-Engineering Information Modelling/Project/CNN.h5"
model.save(filepath, overwrite=True, include_optimizer=True)

#if(Load_Model_Flag)			#Set this flag to be true when you load this file
#	new_model = tf.keras.models.load_model('D:/Work/Academics/AME 505-Engineering Information Modelling/Project/CNN.h5')
#	model = new_model
#else
#	model = create_model()		#Comment the earlier model = create_model() at line 109


history = model.fit(
    train_data_gen,
    steps_per_epoch=batch_size_train,
    epochs=epochs,
    validation_data=val_data_gen, validation_steps= batch_size_test)


#These are the model training and validation accuracies
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

#These are the model training and validation losses
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)


#Graphs for showing training and validation accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



#Classification report and prediction
Y_pred = new_model.predict(val_data_gen);               #Use this command to make the model predict
y_pred = np.argmax(Y_pred, axis=1)
print(val_data_gen.classes)
print(y_pred)
print('Confusion Matrix')
confusion_mat = tf.math.confusion_matrix(val_data_gen.classes, y_pred)
print(confusion_mat)
print('Classification Report')
target_names = ['Cr', 'In', 'Pa', 'Ps', 'Rs', 'Sc']
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))


