# Ranking and Sorting data

import pandas as pd
import numpy as np

s = pd.Series(range(4), index=['d','a', 'b', 'c'])
print(s.sort_index())

# Read data from CSV file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section3'
filePath += '\\GlobalFirePower.csv'
data = pd.read_csv(filePath)


data.sort_index(axis=1) # Index according to columns. Columns are arranged alphabetically
# Default: ascending
data.sort_index(axis=1, ascending=False) # Now Descending

x1 = data.sort_values(by='ISO3') # countries alphabetically A-Z (ascending order)
x2 = data.sort_values(by='Rank', ascending=False)  # By rank (descending order)


# Ranking data
x = data.rank(axis=1)

# Storing a portion of data
x = data.iloc[0:8, 0:8]

x['mRank'] = x['Fit-for-Service'].rank(ascending=False)