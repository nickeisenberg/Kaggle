# unfinished...

import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

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
