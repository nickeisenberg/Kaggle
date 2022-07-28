import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
answers_data = pd.read_csv('gender_submission.csv')

train_data = train_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

train_data['Sex'] = train_data['Sex'].replace(['male'], 1)
train_data['Sex'] = train_data['Sex'].replace(['female'], 0)
test_data['Sex'] = test_data['Sex'].replace(['male'], 1)
test_data['Sex'] = test_data['Sex'].replace(['female'], 0)

train_data = train_data[['PassengerId','Sex','SibSp','Parch','Pclass','Survived']]
test_data = test_data[['PassengerId','Sex','SibSp','Parch','Pclass']]

inp_train = train_data.iloc[:, 0:5]
out_train = train_data.iloc[:, 5]

sc = StandardScaler()
inp_train = sc.fit_transform(inp_train)
inp_test = sc.fit_transform(test_data)

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(3, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(inp_train, out_train, batch_size=10, epochs=100)

predictions = model.predict(inp_test).tolist()
predictions = pd.Series(predictions)

pred = []
for x in predictions:
    pred.append(x[0])
pred = pd.Series(pred)

answers_data['Check'] = pred

rounding = []
for x in answers_data['Check']:
    if x <= .5:
        rounding.append(0)
    else:
        rounding.append(1)

answers_data['final'] = pd.Series(rounding)
answers_data.to_csv('final.csv')


answers_data.to_csv('predictions.csv')

match = 0
nomatch = 0
count = 0
for val in answers_data.values:
    if val[1] == val[3]:
        match += 1
    else:
        nomatch += 1
    count += 1

print(f'The proportion of correct guesses is {match / count}')
