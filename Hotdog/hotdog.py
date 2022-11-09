from tensorflow.keras.preprocessing.image import load_img, smart_resize
import os
from os import listdir
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

train_hot = []
for images in os.listdir('train/hot_dog'):
    name = ''.join(['train/hot_dog/', images])
    img = load_img(name)
    img = smart_resize(img, (100,100))
    img = np.asarray(img)
    dim1, dim2 = img.shape[0], img.shape[1]
    img = img.reshape((dim1 * dim2, 3))
    img = img.astype('float32') / 255
    train_hot.append(img)

train_nothot = []
for images in os.listdir('train/not_hot_dog'):
    name = ''.join(['train/not_hot_dog/', images])
    img = load_img(name)
    img = smart_resize(img, (100, 100))
    img = np.asarray(img)
    dim1, dim2 = img.shape[0], img.shape[1]
    img = img.reshape((dim1 * dim2, 3))
    img = img.astype('float32') / 255
    train_nothot.append(img)

train = np.array(train_hot + train_nothot)
train = train.reshape((train.shape[0], train.shape[1] * train.shape[2]))

train_labels = np.zeros(train.shape[0])
# train_labels = np.array([1])
for i in range(len(train_hot) - 1):
    # train_labels = np.vstack((train_labels, np.ones(1)))
    train_labels[i] = 1

# for i in range(len(train_nothot)):
#     train_labels = np.vstack((train_labels, np.zeros(1)))

inputs = keras.Input(shape=(train.shape[1],), name='inputs')
feature = layers.Dense(512, activation='relu')(inputs)
outputs = layers.Dense(1, activation='sigmoid')(feature)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, train_labels, epochs=10, batch_size=128)

test_hot = []
for images in os.listdir('test/hot_dog'):
    name = ''.join(['test/hot_dog/', images])
    img = load_img(name)
    img = smart_resize(img, (100, 100))
    img = np.asarray(img)
    dim1, dim2 = img.shape[0], img.shape[1]
    img = img.reshape((dim1 * dim2, 3))
    img = img.astype('float32') / 255
    test_hot.append(img)

test_nothot = []
for images in os.listdir('test/not_hot_dog'):
    name = ''.join(['test/not_hot_dog/', images])
    img = load_img(name)
    img = smart_resize(img, (100, 100))
    img = np.asarray(img)
    dim1, dim2 = img.shape[0], img.shape[1]
    img = img.reshape((dim1 * dim2, 3))
    img = img.astype('float32') / 255
    test_nothot.append(img)

test = np.array(test_hot + test_nothot)
test = test.reshape((test.shape[0], test.shape[1] * test.shape[2]))

test_labels = np.zeros(test.shape[0])
# train_labels = np.array([1])
for i in range(len(test_hot) - 1):
    # train_labels = np.vstack((train_labels, np.ones(1)))
    test_labels[i] = 1

print(model.evaluate(test, test_labels))
