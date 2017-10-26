'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pickle
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
import pandas as pd
import numpy as np
import tqdm

batch_size = 10
num_classes = 1
epochs = 12

# input image dimensions
img_rows, img_cols = 75, 75

# Load Data from JSON
def get_data():
  print ("Loading Data....")
  train = pd.read_json('data/train.json')
  x_train=[]
  for index, row in train.iterrows():
    x_train.append([np.array(row['band_1']).reshape(75,75), np.array(row['band_2']).reshape(75,75)])
  x_train=np.array(x_train)

  test = pd.read_json('data/test.json')
  x_test=[]
  for index, row in test.iterrows():
    x_test.append([np.array(row['band_1']).reshape(75,75), np.array(row['band_2']).reshape(75,75)])
  x_test=np.array(x_test)

  y_train=np.array(train['is_iceberg'])
  test_ids=np.array(test['id'])

  print ("Loading Complete.")

  print('x_train shape:', x_train.shape)
  print('y_train shape:', y_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  return x_train, y_train, x_test, test_ids


# the data, shuffled and split between train and test sets
# # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# # (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train, x_test, test_ids=get_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 2, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 2, img_rows, img_cols)
    input_shape = (2, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 2)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 2)
    input_shape = (img_rows, img_cols, 2)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# print(y_train.shape[0], 'train labels')
# # print(y_test.shape[0], 'test labels')

def train():
  model = Sequential()
  model.add(BatchNormalization(input_shape=input_shape))
  model.add(Conv2D(16, kernel_size=(3, 3),
                   activation='relu'))#,
                   # input_shape=input_shape))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(48, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='sigmoid'))

  model.compile(loss=keras.losses.binary_crossentropy,
                optimizer='rmsprop',
                metrics=['accuracy'])
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
            # ,validation_data=(x_test, y_test)
            )
  # score = model.evaluate(x_test, y_test, verbose=0)
  # print('Test loss:', score[0])
  # print('Test accuracy:', score[1])
  return model

def save_model(model, path='model.json'):
  # serialize model to JSON
  model_json = model.to_json()
  with open(path, "w") as json_file:
      json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")

# load json and create model
def load_model(path='model.json'):
  json_file = open(path, 'r')
  model_json = json_file.read()
  json_file.close()
  model = model_from_json(model_json)
  # load weights into new model
  model.load_weights("model.h5")
  print("Loaded model from disk")
  return model

def results():
  print ("Generating Submission File")
  y_test=model.predict(x_test)
  submission = pd.DataFrame({'id': test_ids, 'is_iceberg': y_test.reshape((y_test.shape[0]))})
  submission.to_csv("subv1.csv", index=False)

model=train()
save_model(model)
model=load_model('model.json')
results()




# df_sub=pd.DataFrame(y_test)
# df_sub.to_csv('sub.csv')