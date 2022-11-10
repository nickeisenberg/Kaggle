from sys import exit
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from tensorflow.keras.utils import timeseries_dataset_from_array
import tensorflow as tf
print('--------------------------------------------------')

# dummy data, each row is data for a certiain time.
# 6 pieces of data for 1000 different times
x = np.ones((1000, 8))

# possible dummy targets
y = np.ones((1000,))
for i in range(1000):
    x[i, :] *= i
    y[i] *= i

#y = np.ones((1000,2))
#for i in range(1000):
#    x[i, :] *= i
#    y[i, :] *= i

# y = np.ones((1000,2,2))
# for i in range(1000):
#     x[i, :] *= i
#     y[i, :, :] *= i

# a delay for when to start the targets
delay = 2 * (6 + 5 - 1)

dataset = timeseries_dataset_from_array(data=x[:-delay],
                                        targets=y[delay:],
                                        sequence_length=6,
                                        sampling_rate=2,
                                        batch_size=5,
                                        shuffle=False,
                                        start_index=0,
                                        end_index=20)

for inps, tars in dataset:
    # print(f'for inp:\n{inp}')
    # print(f'for tar:\n{tar}')
    for i in range(inps.shape[0]):
        print(inps[i])
        print(tars[i])

exit()
for inps, tars in dataset:
    # print(f'for inp:\n{inp}')
    # print(f'for tar:\n{tar}')
    for i in range(inps.shape[0]):
        X = []
        for x in inps[i]:
            X.append(x)
        X = tf.convert_to_tensor(X)
        print('---')
        print(X)
        print(tars[i])
        print('---')

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
