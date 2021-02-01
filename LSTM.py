import os
import cv2
import numpy
import dlib
import imageio
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
import tensorflow as tf
from tensorflow.keras import models, layers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(
			gpus[0],
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)])
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print(e)

image_rows, image_columns, image_depth = 192, 192, 12
filepath="output/lstm_trained_weights.hdf5"

segmentorder = [];
uniquesegments = []
training_list = []
traininglabels = []

lbptopval = 1

training_list = numpy.load('./lstm_microexpstcnn_images.npy')
traininglabels =numpy.load('./lstm_microexpstcnn_labels.npy')

timesteps = image_depth
width = image_rows
height = image_columns
channels = 1
action_num = 2

trainX = training_list
trainy = traininglabels
model = Sequential()
batch_size = len(uniquesegments)
# Embedding layer
model = models.Sequential(
	[
		layers.Input(
			shape=(timesteps, width, height, channels)
		),
		layers.ConvLSTM2D(
			filters=32, kernel_size=(3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1
		),
		layers.MaxPool3D(
			pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same"
		),
		layers.BatchNormalization(),
		layers.ConvLSTM2D(
			filters=16, kernel_size=(3, 3), padding="same", return_sequences=True, dropout=0.1, recurrent_dropout=0.1
		),
		layers.MaxPool3D(
			pool_size=(1, 2, 2), strides=(1, 2, 2), padding="same"
		),
		layers.BatchNormalization(),
		layers.ConvLSTM2D(
			filters=8, kernel_size=(3, 3), padding="same", return_sequences=False, dropout=0.1, recurrent_dropout=0.1
		),
		layers.MaxPool2D(
			pool_size=(2, 2), strides=(2, 2), padding="same"
		),
		layers.BatchNormalization(),
		layers.Flatten(),
		layers.Dense(192, activation='relu'),
		layers.Dense(action_num, activation='softmax')
	]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#model.fit(trainX, trainy, epochs=100, batch_size=batch_size, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

train_images, validation_images, train_labels, validation_labels =  train_test_split(training_list, traininglabels, test_size=0.3, random_state=4)
history = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 30, epochs = 300, shuffle=True)
