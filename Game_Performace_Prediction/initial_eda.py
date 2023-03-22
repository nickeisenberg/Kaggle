import pandas as pd
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
and then screen and room coordinate clicks.
°°°"""
#|%%--%%| <UFPJGkiNrf|y0oKRfYmPl>

id_avg_df = pd.DataFrame(id_avg.items(), columns=['session_id', 'score'])
sorted_inds = np.argsort(id_avg_df.values[:, 1])[::-1]
id_avg_df = id_avg_df.iloc[sorted_inds]

#|%%--%%| <y0oKRfYmPl|zqOi3ru0Es>

id_avg_df.head()

#|%%--%%| <zqOi3ru0Es|0ZttEmk8Xb>
r"""°°°
Lets compare the top n sessions with the bottom n sessions in terms of the the
screen and room coord clicks.
°°°"""
#|%%--%%| <0ZttEmk8Xb|FngvcNH9k4>

n = 250

def coord_getter(ids, cols, verbose=False):
    coors = []
    for i, id in enumerate(ids):
        coors.append(
                train_set.loc[train_set['session_id'] == id][cols]
                )
        if verbose:
            if i / 100 == int(i / 100):
                print(i / ids.size)
    return pd.concat(coors).dropna()

cols = ['session_id',
        'screen_coor_x', 'screen_coor_y',
        'room_coor_x', 'room_coor_y']
top_n_coor_df = coord_getter(id_avg_df['session_id'].values[: n], cols,
                             verbose=True)
bot_n_coor_df = coord_getter(id_avg_df['session_id'].values[-n:], cols)

fig, ax = plt.subplots(1, 2)
_ = ax[0].hist2d(top_n_coor_df['screen_coor_x'].values,
                 top_n_coor_df['screen_coor_y'].values,
                 bins=40)
_ = ax[0].set_title(f'Screen coords for top {n} scores')
_ = ax[1].hist2d(bot_n_coor_df['screen_coor_x'].values,
                 bot_n_coor_df['screen_coor_y'].values,
                 bins=40)
_ = ax[1].set_title(f'Screen coords for bottom {n} scores')
plt.show()

fig, ax = plt.subplots(1, 2)
_ = ax[0].hist2d(top_n_coor_df['room_coor_x'].values,
                 top_n_coor_df['room_coor_y'].values,
                 bins=40)
_ = ax[0].set_title(f'Room coords for top {n} scores')
_ = ax[1].hist2d(bot_n_coor_df['room_coor_x'].values,
                 bot_n_coor_df['room_coor_y'].values,
                 bins=40)
_ = ax[1].set_title(f'Room coords for bottom {n} scores')
plt.show()

#|%%--%%| <FngvcNH9k4|jLrPtrXrB3>
r"""°°°
Lets see how the top session look in reference to the top n sessions.
°°°"""
#|%%--%%| <jLrPtrXrB3|yeGpqMxDRh>

top_1_coor_df = coord_getter(id_avg_df['session_id'].values[0:1], cols)
fig, ax = plt.subplots(1, 2)
_ = ax[0].hist2d(top_n_coor_df['room_coor_x'].values,
                 top_n_coor_df['room_coor_y'].values,
                 bins=40)
_ = ax[1].hist2d(top_1_coor_df['room_coor_x'].values,
                 top_1_coor_df['room_coor_y'].values,
                 bins=40)
plt.show()

#|%%--%%| <yeGpqMxDRh|A5Sd71sU90>
r"""°°°
It appears that there is a differece between the coordinate clicks for the best
and worst sessions. Lets try to train a CNN classifer for one of the questions
based on the coordinates. Lets use the room coordinates rather than the screen
coordinates since this will not vary depending on where the session has the
app window placed on their screen.
°°°"""
#|%%--%%| <A5Sd71sU90|o6oELHalcl>

im = np.histogram2d(top_1_coor_df['room_coor_x'].values,
                    top_1_coor_df['room_coor_y'].values,
                    bins=50)

plt.imshow(im[0])
plt.show()

#|%%--%%| <o6oELHalcl|tjTqxuzD93>
r"""°°°
Lets create the training set. We will have to create a training set for each
question and so each question will have its own model.
°°°"""
#|%%--%%| <tjTqxuzD93|ELa3yxdYRy>

def im_getter(df, ids, cols, verbose=False):
    ims = []
    for i, id in enumerate(ids):
        id_df = df.loc[df['session_id'] == id][cols].dropna()
        im = np.histogram2d(id_df[cols[0]].values, id_df[cols[1]].values, bins=50)
        ims.append(im[0])
        if verbose:
            if i / 100 == int(i / 100):
                print(i / ids.size)
    return ids, np.array(ims)

bol_1_c = train_labels_sp['question'] == 1
bol_1_c *= train_labels_sp['correct'] == 1
bol_1_ic = train_labels_sp['question'] == 1
bol_1_ic *= train_labels_sp['correct'] == 0
qu1_c = train_labels_sp.loc[bol_1_c]
qu1_ic = train_labels_sp.loc[bol_1_ic]

cols = ['room_coor_x', 'room_coor_y']
ids_c, ims_c = im_getter(train_set, qu1_c['session_id'].values[:10],
                         cols,
                         verbose=True)
ids_ic, ims_ic = im_getter(train_set, qu1_ic['session_id'].values[:10],
                           cols,
                           verbose=True)

train_ims_qu1 = np.vstack((ims_c, ims_ic))
train_labels_qu1 = np.hstack((np.ones(ids_c.size), np.zeros(ids_ic.size)))

train_ims_qu1.shape

#|%%--%%| <ELa3yxdYRy|sE2vnBz8Z0>
r"""°°°
Lets reshape the images into 3d tensors 
°°°"""
#|%%--%%| <sE2vnBz8Z0|5PLhyIDrsC>

stacked_ims = ims_c.reshape(np.hstack((ims_c.shape, 1)))

stacked_ims.shape

fig, ax = plt.subplots(1, 2)
ax[0].imshow(stacked_ims[0])
ax[1].imshow(ims_c[0])
plt.show()


#|%%--%%| <5PLhyIDrsC|DSFfc2SC0m>
r"""°°°
Now lets define the model.
°°°"""
#|%%--%%| <DSFfc2SC0m|RGqk0Pxw40>

inputs = keras.Input(shape=(50, 50, 1))
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
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

model.summary()

model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

model_path = '/Users/nickeisenberg/GitRepos/Kaggle/'
model_path += 'Game_Performace_Prediction/Models/'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path + 'cnn_model.keras',
            save_best_only=True,
            monitor='val_accuracy')
        ]

history = model.fit(
        stacked_ims,
        train_labels_qu1,
        epochs=1,
        shuffle=False,
        callbacks=callbacks,
        validation_split=.1)

model = keras.models.load_model(model_path + 'cnn_model.keras')

#|%%--%%| <RGqk0Pxw40|4ugzqFldN4>
r"""°°°
Lets create the test set
°°°"""
#|%%--%%| <4ugzqFldN4|oCiFt3obYO>


test_ids = test_set['session_id'].value_counts().index.values
cols = ['room_coor_x', 'room_coor_y']
test_ids, test_ims = im_getter(test_set, test_ids, cols, verbose=True)

test_ims = test_ims.reshape(np.hstack((test_ims.shape, 1)))

model.predict(test_ims)
