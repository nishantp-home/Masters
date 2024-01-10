# Ranking and Sorting data

import pandas as pd
import numpy as np

s = pd.Series(range(4), index=['d','a', 'b', 'c'])
print(s.sort_index())

# Read data from CSV file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section3'
file = filePath + '\\GlobalFirePower.csv'
data = pd.read_csv(file)

# Pivoting
p = pd.pivot_table(data=data, index='ISO3') # Single column
p = pd.pivot_table(data=data, index=['Country', 'ISO3']) # Mulitple columns
p = pd.pivot_table(data=data, index=['Country', 'ISO3'], values=['Attack Aircraft', 'Active Personnel']) # Mulitple columns including corresponding values


# Read data from CSV file
file2 = filePath + '\\endangeredLang.csv'
data2 = pd.read_csv(file2)
data2 = data2.rename(columns={data2.columns[3]: 'CountryCode', data2.columns[4]: 'Endangered'})

# Sum endangered Languages according to country
p2 = pd.pivot_table(data=data2, index=['CountryCode', 'Endangered'], aggfunc=np.sum)
p2 = pd.pivot_table(data=data2, index=['CountryCode', 'Endangered'], values=['Number of speakers'], aggfunc=np.sum)


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