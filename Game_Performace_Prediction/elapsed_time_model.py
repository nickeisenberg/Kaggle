import pandas as pd
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

#|%%--%%| <pfWACn3Lhp|Tu7nsZ8F0s>

path = '/Users/nickeisenberg/GitRepos/OFFLINE_DIRECTORY/Datasets/'
path += 'predict-student-performance-from-game-play/'

train_set = pd.read_csv(path + 'train.csv')
train_labels = pd.read_csv(path + 'train_labels.csv')
sample_sub = pd.read_csv(path + 'sample_submission.csv')

test_set = pd.read_csv(path + 'test.csv')

#|%%--%%| <Tu7nsZ8F0s|vzJ7XC8Hxh>
r"""°°°
The training set is has three level groups and the sample submission shows
which of the 18 questions belongs to each group. We may eventually want to
separate the training set into three different sets based on the level group.
°°°"""
#|%%--%%| <vzJ7XC8Hxh|EJMOWRlTPF>

train_set.head()
sample_sub.head()

#|%%--%%| <EJMOWRlTPF|P7JwsboB3V>
r"""°°°
Also, it will be easier for subsetting to reformat the the sample sub and the
training set as follows.
°°°"""
#|%%--%%| <P7JwsboB3V|ESf1c9HrUg>

train_labels_sp = pd.DataFrame()
cols = ['session_id', 'question']
train_labels_sp[cols] = train_labels['session_id'].str.split('_', expand=True)
train_labels_sp['correct'] = train_labels['correct']
train_labels_sp['question'] = train_labels_sp['question'].str[1:]
train_labels_sp = train_labels_sp.astype(int)

sample_sub_sp = pd.DataFrame()
cols = ['session_id', 'question']
sample_sub_sp[cols] = sample_sub['session_id'].str.split('_', expand=True)
sample_sub_sp['question'] = sample_sub_sp['question'].str[1:]
sample_sub_sp['correct'] = sample_sub['correct']
sample_sub_sp = sample_sub_sp.astype(int)
sample_sub_sp['level_group'] = sample_sub['session_level'].str.split(
        '_', expand=True)[1]

#|%%--%%| <ESf1c9HrUg|7s6trNHRsr>
r"""°°°
The newly formated sets
°°°"""
#|%%--%%| <7s6trNHRsr|pVh5jFMpRB>

train_labels_sp.head()
sample_sub_sp.head()

train_labels_sp.tail()
sample_sub_sp.tail()

#|%%--%%| <pVh5jFMpRB|ayHj7rQZAQ>
r"""°°°
Lets go through the training labels and get some statistics about each question
and how each session did overal.
°°°"""
#|%%--%%| <ayHj7rQZAQ|P0jv5JL55X>

s_ids = train_labels_sp['session_id'].value_counts().index.values

id_avg = {}
for i, id in enumerate(s_ids):
    df_id = train_labels_sp.loc[train_labels_sp['session_id'] == id]
    id_avg[id] = df_id['correct'].mean()

qu_avg = {}
for qu in range(1, 19):
    df_qu = train_labels_sp.loc[train_labels_sp['question'] == qu]
    qu_avg[qu] = df_qu['correct'].mean()

fig, ax = plt.subplots(1, 2)
_ = ax[0].hist(x=list(id_avg.values()), bins=25)
_ = ax[0].set_title('Session scores')
_ = ax[1].bar(x=range(1, 19), height=list(qu_avg.values()))
_ = ax[1].set_xticks(range(1, 19))
_ = ax[1].set_title('Question scores')
plt.show()

#|%%--%%| <P0jv5JL55X|UFPJGkiNrf>
r"""°°°
Lets see if we can find a correlation between session or question performacne
and elapsed time.
°°°"""
#|%%--%%| <UFPJGkiNrf|y0oKRfYmPl>

id_avg_df = pd.DataFrame(id_avg.items(), columns=['session_id', 'score'])
sorted_inds = np.argsort(id_avg_df.values[:, 1])[::-1]
id_avg_df = id_avg_df.iloc[sorted_inds]

#|%%--%%| <y0oKRfYmPl|zqOi3ru0Es>

id_avg_df.head()

#|%%--%%| <zqOi3ru0Es|GA2k3ZhRm0>



