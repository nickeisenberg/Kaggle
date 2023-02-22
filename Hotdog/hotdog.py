from tensorflow.keras.preprocessing.image import load_img
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

#|%%--%%| <qbobvZ9rEG|jn1whacKxL>
r"""°°°
# Load and reshape the jpg images for training.
°°°"""
#|%%--%%| <jn1whacKxL|JjygWtE4uf>

path_train_hd = '/Users/nickeisenberg/GitRepos/Kaggle/Hotdog/train/hot_dog/'
hd_train_imgs = []
for img_name in os.listdir(path_train_hd):
    img = load_img(path_train_hd + img_name).resize((180, 180))
    hd_train_imgs.append(np.array(img, dtype='uint8') / 255.)
hd_train_imgs = np.array(hd_train_imgs)

path_train_nhd = '/Users/nickeisenberg/GitRepos/Kaggle/Hotdog/train/not_hot_dog/'
nhd_train_imgs = []
for img_name in os.listdir(path_train_nhd):
    img = load_img(path_train_nhd + img_name).resize((180, 180))
    nhd_train_imgs.append(np.array(img, dtype='uint8') / 255.)
nhd_train_imgs = np.array(nhd_train_imgs)

#|%%--%%| <JjygWtE4uf|yJUIH5E2kK>

hd_train_imgs.shape
nhd_train_imgs.shape

#|%%--%%| <yJUIH5E2kK|sZG31shZjK>
r"""°°°
# We need to create labels
°°°"""
#|%%--%%| <sZG31shZjK|5Qe1299RO7>

labels = np.zeros(hd_train_imgs.shape[0] + nhd_train_imgs.shape[0])
labels[: hd_train_imgs.shape[0]] = 1

train_imgs = np.vstack((hd_train_imgs, nhd_train_imgs))

#|%%--%%| <5Qe1299RO7|XwRLJTg0XR>
r"""°°°
# We can shuffle the data and the labels
°°°"""
#|%%--%%| <XwRLJTg0XR|JnIvE31t2l>

np.random.seed(1)

inds = np.linspace(0, train_imgs.shape[0] - 1, train_imgs.shape[0]).astype(int)
np.random.shuffle(inds)

labels = labels[inds]
train_imgs = train_imgs[inds]

#|%%--%%| <JnIvE31t2l|hwhsZ4shUZ>
r"""°°°
# View some of the images
°°°"""
#|%%--%%| <hwhsZ4shUZ|BbkSwiaUOr>

fig, ax = plt.subplots(1, 2)
ax[0].imshow(train_imgs[0])
ax[0].set_title(f'label: {labels[0]}')
ax[1].imshow(train_imgs[3])
ax[1].set_title(f'label: {labels[3]}')
plt.show()

#|%%--%%| <BbkSwiaUOr|1qV7W1YXwA>
r"""°°°
# Before defining the model, we can create a data augmentation layer to help
# the model since there is very little training data.
°°°"""
#|%%--%%| <1qV7W1YXwA|XUH53PLDD0>

data_augmentation = keras.Sequential(
        [
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(.1),
            layers.RandomZoom(.2)
        ]
)

#|%%--%%| <XUH53PLDD0|lHYbPYSwnn>
r"""°°°
# Lets define the model
°°°"""
#|%%--%%| <lHYbPYSwnn|dhKTgMDxYM>

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

#|%%--%%| <dhKTgMDxYM|lLlnqfTk73>

model.summary()

#|%%--%%| <lLlnqfTk73|XZeW9U3tHI>

model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

#|%%--%%| <XZeW9U3tHI|mkQwBXEkF0>
r"""°°°
# Lets create a callback to save the best model only.
°°°"""
#|%%--%%| <mkQwBXEkF0|SyCt6H1PXv>

model_path = '/Users/nickeisenberg/GitRepos/Kaggle/Hotdog/Models/'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path + 'data_aug_base_model.keras',
            save_best_only=True,
            monitor='val_accuracy')
        ]

#|%%--%%| <SyCt6H1PXv|sGk3pX49uU>
r"""°°°
# Now lets fit the model to the training data
°°°"""
#|%%--%%| <sGk3pX49uU|KEI6P48kfu>

history = model.fit(
        train_imgs,
        labels,
        epochs=30,
        shuffle=False,
        callbacks=callbacks,
        validation_split=.1)

#|%%--%%| <KEI6P48kfu|vxnRPIE1uT>

model = keras.models.load_model(model_path + 'data_aug_base_model.keras')

#|%%--%%| <vxnRPIE1uT|NBNG45vIrF>
r"""°°°
# We now need to load the test data and evaluate the model
°°°"""
#|%%--%%| <NBNG45vIrF|abIB4ao9td>

path_test_hd = '/Users/nickeisenberg/GitRepos/Kaggle/Hotdog/test/hot_dog/'
hd_test_imgs = []
for img_name in os.listdir(path_test_hd):
    img = load_img(path_test_hd + img_name).resize((180, 180))
    hd_test_imgs.append(np.array(img, dtype='uint8') / 255.)
hd_test_imgs = np.array(hd_test_imgs)

path_test_nhd = '/Users/nickeisenberg/GitRepos/Kaggle/Hotdog/test/not_hot_dog/'
nhd_test_imgs = []
for img_name in os.listdir(path_test_nhd):
    img = load_img(path_test_nhd + img_name).resize((180, 180))
    nhd_test_imgs.append(np.array(img, dtype='uint8') / 255.)
nhd_test_imgs = np.array(nhd_test_imgs)

test_imgs = np.vstack((hd_test_imgs, nhd_test_imgs))

test_labels = np.zeros(test_imgs.shape[0])
test_labels[: hd_test_imgs.shape[0]] = 1

#|%%--%%| <abIB4ao9td|v8dNMD5dGy>

model.evaluate(test_imgs, test_labels)
