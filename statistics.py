import pandas as pd

#load the flights dataset
flights = pd.read_csv('Masters-DataSciencs/dataAnalysisPandas/data-analysis/flights.csv', index_col=False)

#get basic statistics
print(flights.describe())

#compute the mean and STD for a column
print(flights['DISTANCE'].mean())
print(flights['DISTANCE'].std())

#mean of the difference of CRS departure and actual departure times
print((flights['CRS_DEP_TIME']-flights['DEP_TIME']).mean())

