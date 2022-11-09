'''
Data set from:
https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016?resource=download&select=jena_climate_2009_2016.csv
'''

from sys import exit
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import timeseries_dataset_from_array

path_to_data = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016.csv'
climate_df = pd.read_csv(path_to_data)
climate_df.drop(columns=['Date Time'], inplace=True)

temperature = climate_df['T (degC)'].values
# fig, ax = plt.subplots(1, 2)
# ax[0].plot(temperature)
# ax[1].plot(temperature[:1440])
# plt.show()

raw_data = climate_df.drop(columns=['T (degC)']).values

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
delay = 6 * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = timeseries_dataset_from_array(
    data=raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=False,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

