# unfinished...

import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
answers_data = pd.read_csv('sample_submission.csv')

# Remove all categorical columns
train_data = train_data.select_dtypes(exclude=['object'])
train_data.drop(['Id'], axis=1, inplace=True)
train_data.fillna(0, inplace=True)
inp_train = train_data.drop(['SalePrice'], axis = 1)
out_train = train_data.SalePrice

test_data = test_data.select_dtypes(exclude=['object'])
ID = test_data.Id
test_data.fillna(0, inplace=True)
test_data.drop(['Id'], axis=1, inplace=True)

# Normalize the values
sc = StandardScaler()
inp_train = sc.fit_transform(inp_train)
inp_test = sc.fit_transform(test_data)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(200, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

model.fit(inp_train, out_train, batch_size=15, epochs=100)

predictions = model.predict(test_data).tolist()
predictions = pd.Series(predictions)

pred = []
for x in predictions:
    pred.append(x[0])
pred = pd.Series(pred)

answers_data['Guess'] = pred
answers_data.to_csv('final.csv')


