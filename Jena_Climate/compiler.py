import pandas as pd
import numpy as np


path_to_data = '/Users/nickeisenberg/GitRepos/Kaggle/Jena_Climate/DataSet/jena_climate_2009_2016.csv'
climate_df = pd.read_csv(path_to_data)

climate_df.drop(columns=[climate_df.columns[0]], inplace=True)
print(climate_df.head())
