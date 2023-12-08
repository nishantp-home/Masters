import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame([[1, np.nan, 2], [2,3,5], [np.nan, 4, 6]])

# Important methods to process data
print(df.isna()) # check if data frame contains NA
print(df.dropna()) # drop all rows containing NAs
print(df.dropna(axis=1)) # drop all columns containing NAs
print(df.fillna(0))   # fill NA values with zeros
print(df.fillna(method="ffill"))  # specify a forward-fill to propagate the previous value forward
print(df.fillna(method="ffill", axis=1))  #fill forward column-wise
print(df.fillna(method="bfill"))  #fill backward
