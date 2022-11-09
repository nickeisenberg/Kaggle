from sys import exit
import numpy as np
import pandas as pd
from tensorflow.keras.utils import timeseries_dataset_from_array
import tensorflow as tf
print('--------------------------------------------------')

x = np.ones((1000, 6))
y = np.ones((1000,))

for i in range(1000):
    x[i, :] *= i
    y[i] *= i
    y[i] += 1

delay = 2 * (6 + 5 - 1)

dataset = timeseries_dataset_from_array(data=x[:-delay],
                                        targets=y[delay:],
                                        sequence_length=6,
                                        sampling_rate=2,
                                        batch_size=4,
                                        shuffle=False,
                                        start_index=0,
                                        end_index=20)

for inp, tar in dataset:
    # print(f'for inp:\n{inp}')
    # print(f'for tar:\n{tar}')
    for i in range(inp.shape[0]):
        X = []
        for x in inp[i]:
            print(x.shape)
            print(type(x))
            X.append(x)
        X = tf.convert_to_tensor(X)
        print(X)

exit()
print('--------------------------------------------------')

dataset = timeseries_dataset_from_array(data=x[:-delay],
                                        targets=y[delay:],
                                        sequence_length=6,
                                        sampling_rate=2,
                                        batch_size=4,
                                        shuffle=True,
                                        start_index=0,
                                        end_index=20)

for inp, tar in dataset:
    print(f'for inp:\n{inp}')
    print(f'for tar:\n{tar}')
    # for i in range(inp.shape[0]):
    #     print([int(x) for x in inp[i]], int(tar[i]))
