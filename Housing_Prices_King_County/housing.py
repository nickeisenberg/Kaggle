# Solution from https://www.kaggle.com/code/tomasmantero/predicting-house-prices-keras-ann

import pandas as pd
import numpy as np
import random as rnd

# scaling and test/train split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# evaluation on test data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score

# Model creation
from tensorflow import keras
from tensorflow.keras import layers

df = pd.read_csv('kc_house_data.csv')

# # Pearson Correlation Matrix
# sns.set(style="whitegrid", font_scale=1)
# plt.figure(figsize=(13,13))
# plt.title('Pearson Correlation Matrix',fontsize=25)
# sns.heatmap(df.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
#             annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})
# 
# plt.show()

# price_corr = df.corr()['price'].sort_values(ascending=False)
# print(price_corr)

df = df.drop('id', axis=1)
df = df.drop('zipcode',axis=1)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df = df.drop('date',axis=1)

# Features
X = df.drop('price',axis=1)

# Label
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)

scaler = MinMaxScaler()

# fit and transfrom
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the model 
model = keras.Sequential([
    layers.Dense(19, activation='relu'),
    layers.Dense(19, activation='relu'),
    layers.Dense(19, activation='relu'),
    layers.Dense(19, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')

model.fit(x=X_train, y=y_train.values,
          validation_data=(X_test, y_test.values),
          batch_size=128,epochs=400)

# predictions on the test set
predictions = model.predict(X_test)

print('MAE: ',mean_absolute_error(y_test,predictions))
print('MSE: ',mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(mean_squared_error(y_test,predictions)))
print('Variance Regression Score: ',explained_variance_score(y_test,predictions))
print('\n\nDescriptive Statistics:\n',df['price'].describe())

results = pd.DataFrame()
results['Actual'] = pd.Series(y_test)
results['Prediticion'] = pd.Series(predictions.tolist())
results.to_csv('FinalResults.csv')
