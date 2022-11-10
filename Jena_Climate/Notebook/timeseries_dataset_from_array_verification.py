from sys import exit
import pandas as pd
import numpy as np
from tensorflow.keras.utils import timeseries_dataset_from_array

path = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016_.csv'

df = pd.read_csv(path)

vals = df.values
dates = vals[:, 0]
vals = np.array(vals[:, 1:], dtype='float32')

temps = np.array(vals[:, 1], dtype='float32')

num_train = int(len(temps) * .5)
num_val = int(len(temps) * .25)
num_test = int(len(temps) - num_train - num_val)

delay = (6 * 120) + (6 * 24) - 6
dates_delay = dates[delay:]
temps_delay = temps[delay:]

dataset_dates = timeseries_dataset_from_array(
    data=dates,
    sampling_rate=6,
    targets=None,
    sequence_length=120)

dataset_temps = timeseries_dataset_from_array(
    data=temps[: -delay],
    targets=temps[delay:],
    sampling_rate=6,
    sequence_length=120)

dataset = timeseries_dataset_from_array(
    data=vals[: -delay],
    targets=temps[delay:],
    sampling_rate=6,
    sequence_length=120,
    batch_size=256,
    start_index=num_train,
    end_index=num_train + num_val)

err = 0
samples = 0
for inps, tars in dataset:
    preds = inps[:, -1, 1]
    err += np.sum(np.abs(preds - tars))
    samples += inps.shape[0]
print(err / samples)

exit()
for inps, tars in dataset_temps:
    print(inps[0])
    print(tars[0])
    print(df.loc[df['Date Time'] == '06.01.2009 23:10:00'])
    print('')
    print(inps[1])
    print(tars[1])
    print(df.loc[df['Date Time'] == '06.01.2009 23:20:00'])
    break

exit()
for dates_ in dataset_dates:
    print(np.array(dates_[0]))
    print(dates_delay[0])
    print('')
    print(np.array(dates_[1]))
    print(dates_delay[1])
    break

