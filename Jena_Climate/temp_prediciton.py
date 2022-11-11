'''
Data set from:
https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016?resource=download&select=jena_climate_2009_2016.csv
'''

from sys import exit
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import timeseries_dataset_from_array
from copy import deepcopy

path_to_data = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016_.csv'

climate_df = pd.read_csv(path_to_data)
date = climate_df['Date Time'].values
climate_df.drop(columns=['Date Time'], inplace=True)

temperature = climate_df['T (degC)'].values
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(temperature)
# ax[1].plot(temperature[:1440])
# plt.show()

raw_data = deepcopy(climate_df.values)

num_train_samples = int(.5 * len(raw_data))
num_val_samples = int(.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

# print(f'The number of training samples : {num_train_samples}')
# print(f'The number of validation samples : {num_val_samples}')
# print(f'The number of test samples : {num_test_samples}')

train_mean = np.mean(raw_data[: num_train_samples], axis=0)
train_std = np.std(raw_data[: num_train_samples], axis=0)
raw_data -= train_mean
raw_data /= train_std

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

# for inps, tars in train_dataset:
#     for i, t in zip(inps[0:2,:, 1], tars[:2]):
#         print('input:')
#         print(np.array(i))
#         print('output:')
#         print(np.array(t))
#     break

val_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

# for inps, tars in train_dataset:
#     print(inps[0][-1,1] * train_std[1] + train_mean[1])
#     print(tars[0])
#     break


# create a simple baseline to beat.
# the baseline will be that Temp(t) = Tempt(t + 24h)
def baseline(dataset):
    total_abs_error = 0
    samples_seen = 0
    for inps, tars in dataset:
        preds = inps[:, -1, 1] * train_std[1] + train_mean[1]
        total_abs_error += np.sum(np.abs(preds - tars))
        samples_seen += inps.shape[0]
    return total_abs_error / samples_seen

print(f'Baseline Validation MAE: {baseline(val_dataset)}')
print(f'Baseline Test MAE: {baseline(test_dataset)}')

from tensorflow import keras
from tensorflow.keras import layers

# basic model
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint('jena_dense.keras',
                                    save_best_only=True)
]
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model('jena_dense.keras')

print(f'Dense model Test MAE: {model.evaluate(test_dataset)[1] : .2f}')

loss = history.history['mae']
val_loss = history.history['val_mae']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='training MAE')
plt.plot(epochs, val_loss, 'b', label='validation MAE')
plt.title('Dense layer model')
plt.legend()
plt.show()

# lstm model no dropout
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint('jena_lstm_nodropout.keras',
                                    save_best_only=True)
]

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model('jena_lstm_nodropout.keras')
print(f'LSTM model (no dropout) Test MAE: {model.evaluate(test_dataset)[1] : .2f}')

loss = history.history['mae']
val_loss = history.history['val_mae']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training MAE')
plt.plot(epochs, val_loss, 'b', label='Validation MAE')
plt.title('LSTM model (no dropout)')
plt.legend()
plt.show()

# lstm model dropout
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=.25, return_sequences=True)(inputs)
x = layers.LSTM(32, recurrent_dropout=.25)(x)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint('jena_lstm_dropout.keras',
                                    save_best_only=True)
]

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model('jena_lstm_dropout.keras')
print(f'LSTM model (with dropout) Test MAE: {model.evaluate(test_dataset)[1] : .2f}')

loss = history.history['mae']
val_loss = history.history['val_mae']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training MAE')
plt.plot(epochs, val_loss, 'b', label='Validation MAE')
plt.title('LSTM model (dropout)')
plt.legend()
plt.show()
