# Solution given at https://towardsai.net/p/deep-learning/house-price-predictions-using-keras

import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.ensemble import IsolationForest # Outliers 

from keras.models import Sequential # Sequential Neural Network
from keras.layers import Dense
from keras.callbacks import EarlyStopping # Early Stopping Callback
from keras.optimizers import Adam # Optimizer

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = train['SalePrice'].values
data = pd.concat([train,test], axis=0, sort=False)
data.drop(['SalePrice'], axis=1, inplace=True)

missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
NAN_col = missing_values.index.to_list()
missing_values_data = pd.DataFrame(missing_values)
missing_values_data.reset_index(level=0, inplace=True)
missing_values_data.columns = ['Features', 'Number of Missing Values']
missing_values_data['Percentage of Missing Values'] = 100 * missing_values_data['Number of Missing Values'] / len(data)

# Pearson Correlation Matrix
# sns.set(style="whitegrid", font_scale=1)
# plt.figure(figsize=(13,13))
# plt.title('Pearson Correlation Matrix',fontsize=25)
# sns.heatmap(data.corr(),linewidths=0.25,vmax=0.7,square=True,cmap="GnBu",linecolor='w',
#             annot=True, annot_kws={"size":7}, cbar_kws={"shrink": .7})
# plt.show()

# Fill basment sqft missing values with 0 as this means there is no basement
data['BsmtFinSF1'].fillna(0, inplace=True)
data['BsmtFinSF2'].fillna(0, inplace=True)
data['TotalBsmtSF'].fillna(0, inplace=True)
data['BsmtUnfSF'].fillna(0, inplace=True)

# For electrical and kitchen we replace by just manual inspecting for common values
data['Electrical'].fillna('FuseA',inplace = True)
data['KitchenQual'].fillna('TA',inplace=True)

# For lot frontage we replace with the mean from all house with same 1st floor sqft as there is a high correlation
LotFrontage_corr = data.corr()['LotFrontage'].sort_values(ascending=False)
data['LotFrontage'].fillna(data.groupby('1stFlrSF')['LotFrontage'].transform('mean'),inplace=True)
data['LotFrontage'].interpolate(method='linear',inplace=True)

# For MasVnrArea we do the same correlation trick# 
MasVnrArea_corr = data.corr()['MasVnrArea'].sort_values(ascending=False)
data['MasVnrArea'].fillna(data.groupby('MasVnrType')['MasVnrArea'].transform('mean'),inplace=True)
data['MasVnrArea'].interpolate(method='linear',inplace=True)

# For all others, we use the mean for numerical and NA for categorical 
for col in NAN_col:
    data_type = data[col].dtype
    if data_type == 'object':
        data[col].fillna('NA',inplace=True)
    else:
        data[col].fillna(data[col].mean(),inplace=True)

# Adding some new features by combining others
data['Total_Square_Feet'] = (
    data['BsmtFinSF1'] + data['BsmtFinSF2']
    + data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF'])

data['Total_Bath'] = (
    data['FullBath'] + (0.5 * data['HalfBath'])
    + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))

data['Total_Porch_Area'] = (
    data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch']
    + data['ScreenPorch'] + data['WoodDeckSF'])

data['SqFtPerRoom'] = (
    data['GrLivArea'] /
    (data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['KitchenAbvGr']))

# # Numeric versus categorical columns 
# column_data_type = []
# for col in data.columns:
#     data_type = data[col].dtype
#     if data[col].dtype in ['int64','float64']:
#         column_data_type.append('numeric')
#     else:
#         column_data_type.append('categorical')
# plt.figure(figsize=(15,5))
# sns.countplot(x=column_data_type)
# plt.show()

# One-hot encoding for categorical data
data = pd.get_dummies(data)

# Separate the test and train data 
train = data[:1460].copy()
test = data[1460:].copy()
train['SalePrice'] = y

# Find the features of the training set that have high price correlation 
# top_features = train.corr()[['SalePrice']].sort_values(by=['SalePrice'], ascending=False).head(30)
# plt.figure(figsize=(5,10))
# sns.heatmap(top_features, cmap='rainbow', annot=True, annot_kws={"size": 16}, vmin=-1)
# plt.show()

# Function to plot and manually inspect for outliers
def plot_data(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' Analysis')

# Drop the OverallQual outliers

# plot_data('OverallQual',True) 
train = train.drop(train[(train['OverallQual'] == 10) & (train['SalePrice'] < 200000)].index)

# plot_data('Total_Bath')
train = train.drop(train[(train['Total_Bath'] > 4) & (train['SalePrice'] < 200000)].index)

# plot_data('TotalBsmtSF')
train = train.drop(train[(train['TotalBsmtSF'] > 3000) & (train['SalePrice'] < 400000)].index)

# reset the index 
train.reset_index()

# We now want remove the outliers in the other categories that have a lower correlaton to price
# We use isolation forest for this

clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
train.reset_index(drop = True, inplace = True)
# print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
# print("Number of rows without outliers:", train.shape[0])

# Now we scale the data
sc = StandardScaler()

X = train.copy()
X.drop(['SalePrice'],axis=1,inplace=True) # Dropped the y feature
Y = train['SalePrice'].values

X = sc.fit_transform(X)

# Create the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(320, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(384, activation='relu'))
    model.add(Dense(352, activation='relu'))
    model.add(Dense(448, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'mse')
    return model
model = create_model()
# model.summary()

# # Find overfitting 

# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
# history = model.fit(x=X,y=Y,
#           validation_split=0.1,
#           batch_size=128,epochs=1000, callbacks=[early_stop])
# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

# Reset the model
model = create_model()
history = model.fit(x=X,y=Y,
          batch_size=128,epochs=170)
# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

X_test = sc.transform(test) # Scaling the testing data.
result = model.predict(X_test) # Prediction using model
result = pd.DataFrame(result,columns=['Estimate']) # Dataframe
result['Id'] = test['Id'] # Adding ID to our result dataframe.
result = result[['Id','Estimate']]

actual = pd.read_csv('sample_submission.csv')
result['Actual'] = actual['SalePrice']
result['Percent Error'] = 100 * abs(result['Estimate'] - result['Actual']) / result['Actual']
result.to_csv('FinalPredicitons.csv')


