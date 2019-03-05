"""
March 4, 2019
	authors: 
		Caio E. C. Oliveira (github.com/caduunb), Mechatronics Engineering Student at University of Brasilia.
		Sangeeta Biswas, Post-Doc Researcher at Brno University of Technology, Czech Republic
	usage:
		python3 classification_model.py --logdir="<PATH/TO/LOGS/>"
	purpose: 
		Training a convolutional neural network on 5 types of Diabetic Retinopathy classification.
	version: 
		Python 3.6.7, tensorflow 1.12
	note:
		The model performs at 63.5% validation accuracy.
"""

# 	Modules
# ---
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPool2D
from tensorflow.keras.models     import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np
import random
import datetime
import os

# 	Defining constants
# ---
IMG_W = 64
IMG_H = 64
GRAYSCALE = False
if GRAYSCALE:
	CH_NO = 1
else:
	CH_NO = 3
DATASET_SIZE = 0 #it is defined by the length of the label List
TRAIN_SIZE = int(0.7*DATASET_SIZE)
SCALE_IMAGE = False
NUM_EPOCHS = 50
BATCH_SIZE = 32
CLASS_NO = 5

now = datetime.datetime.now()
NAME_TENSORBOARD = "logs/{}".format(now.strftime("%Y-%m-%d-%H:%M"))

#   Control the usage of the memory. 1.0 for full usage.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#   Import dataset from a .npz file
# ---
DATA_DIR = "/home/user/example/project"
DATASET_FILE  = "dataset_all_DR.npz"
DATASET_PATH = DATA_DIR + "/" + DATASET_FILE
dataset = np.load(DATASET_PATH)

# Loading shuffled data.
all_labelsList = dataset['labelList']
DATASET_SIZE = len(all_labelsList)
TRAIN_SIZE = int(0.7*DATASET_SIZE)
index = np.arange(0, DATASET_SIZE)
np.random.shuffle(index)
index1 = index[:TRAIN_SIZE]
index2 = index[TRAIN_SIZE:]
train_imageList  = dataset['imageList'][index1]
train_labelsList = dataset['labelList'][index1]
test_imageList   = dataset['imageList'][index2]
test_labelsList  = dataset['labelList'][index2]

# Encoding lists into numpy arrays
train_labelsSet = np.array(train_labelsList)
test_labelsSet  = np.array(test_labelsList)
LEN_TRAIN = len(train_imageList)
LEN_TEST = len(test_imageList)
train_imageSet = np.zeros((LEN_TRAIN, IMG_H, IMG_W, CH_NO), dtype = 'uint8')
test_imageSet = np.zeros((LEN_TEST, IMG_H, IMG_W, CH_NO), dtype = 'uint8')
train_labelsSet = tf.keras.utils.to_categorical(train_labelsSet, CLASS_NO)  # One-dot enconding
test_labelsSet = tf.keras.utils.to_categorical(test_labelsSet, CLASS_NO)    # One-dot enconding

# Scaling images into [0,1]
if SCALE_IMAGE:
	train_imageSet = train_imageSet/255.
	test_imageSet  = test_imageSet/255.

# 	Model Architecture
# ---
# Input layer
input_layer = Input((IMG_W, IMG_H, CH_NO))
hidden_layers = input_layer

# Hidden Convolutional layers
hidden_layers = Conv2D(kernel_size = 4, filters = 4, strides = 2, padding = 'valid', activation = 'relu')(hidden_layers)
hidden_layers = Conv2D(kernel_size = 4, filters = 8, strides = 2, padding = 'valid', activation = 'relu')(hidden_layers)
hidden_layers = Conv2D(kernel_size = 4, filters = 16, strides = 1, padding = 'valid', activation = 'relu')(hidden_layers)
hidden_layers = Conv2D(kernel_size = 4, filters = 64, strides = 1, padding = 'valid', activation = 'relu')(hidden_layers)

# Hidden Dense Layers
for i in range(2):
    hidden_layers = Dense(50)(hidden_layers)
    hidden_layers = Dropout(0.3)(hidden_layers)
    hidden_layers = Activation('relu')(hidden_layers)

# Output layer
hidden_layers = Flatten()(hidden_layers)
output_layer = Dense(CLASS_NO, activation = 'softmax')(hidden_layers)
model = Model(input_layer,output_layer)

model.summary()

"""
print ("Training image Set shape:")
print (train_imageSet.shape)
print ("Testing image Set shape:")
print (test_imageSet.shape)
print ("Training labels Set shape:")
print (train_labelsSet[0:10])
print ("Testing labels Set shape:")
print (test_labelsSet[0:10])
"""

#   Setting TensorBoard
# ---
tensorboard = TensorBoard(log_dir=NAME_TENSORBOARD)

# 	Model training
# ---
model.compile(
	optimizer= Adam(lr = 0.001),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])
model.fit(train_imageSet, train_labelsSet, batch_size = BATCH_SIZE, epochs= NUM_EPOCHS, validation_split = 0.2,
			use_multiprocessing = True, callbacks = [tensorboard])

#   Predictions
# ---
#print (model.predict(test_imageList[:10]))
test_loss, test_acc = model.evaluate(test_imageSet, test_labelsSet)
