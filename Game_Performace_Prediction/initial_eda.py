import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#|%%--%%| <ZEPxUYhlgk|Tu7nsZ8F0s>

path = '/Users/nickeisenberg/GitRepos/OFFLINE_DIRECTORY/Datasets/'
path += 'predict-student-performance-from-game-play/'

train_set = pd.read_csv(path + 'train.csv')
train_labels = pd.read_csv(path + 'train_labels.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

#|%%--%%| <Tu7nsZ8F0s|EJMOWRlTPF>



