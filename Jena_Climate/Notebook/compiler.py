import numpy as np
import pandas as pd
from sys import exit
import matplotlib.pyplot as plt

exit()

path = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016_.csv'
df = pd.read_csv(path)

temps = df['T (degC)'].values

num_train = int(len(temps) * .5)
num_val = int(len(temps) * .25)
num_test = int(len(temps) - num_train - num_val)

print(num_val)
exit()
delay = (6 * 24 * 5) + (6 * 24) - 6

df = df[['Date Time', 'T (degC)']]
df_train = df.iloc[: num_train]
df_val = df.iloc[num_train : num_train + num_val]
df_test = df.iloc[num_train + num_val : ]

df24 = df.iloc[[*range(0,len(df),144)]]

temps24 = df24['T (degC)'].values

num = 0
err = 0
for i in range(1, len(temps24)):
    error = np.abs(temps24[i] - temps24[i-1])
    num += 1
    err += error

print(err / num)


