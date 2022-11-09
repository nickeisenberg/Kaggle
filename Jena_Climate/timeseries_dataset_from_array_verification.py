from sys import exit
import pandas as pd
import numpy as np
from tensorflow.keras.utils import timeseries_dataset_from_array

path = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016.csv'
path_ = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016_.csv'

df = pd.read_csv(path)
df_ = pd.read_csv(path_)

vals = df.values
vals_ = df_.values

dates = vals_[:, 0]
temps = np.array(vals_[:, 2], dtype='float32')

dataset_dates = timeseries_dataset_from_array(
    data=dates,
    sampling_rate=6,
    targets=None,
    sequence_length=120)

dataset_temps = timeseries_dataset_from_array(
    data=temps,
    sampling_rate=6,
    targets=None,
    sequence_length=120)

delay = (6 * 120) + (6 * 24) - 6
dates_delay = dates[delay:]
temps_delay = temps[delay:]

for temps in dataset_temps:
    print(np.array(temps[0]))
    print(temps_delay[0])
    print(df_.loc[df_['Date Time'] == '06.01.2009 23:10:00'])
    print('')
    print(np.array(temps[1]))
    print(temps_delay[1])
    print(df_.loc[df_['Date Time'] == '06.01.2009 23:20:00'])
    break


# for dates_ in dataset_dates:
#     print(np.array(dates_[0]))
#     print(dates_delay[0])
#     print('')
#     print(np.array(dates_[1]))
#     print(dates_delay[1])
#     break



