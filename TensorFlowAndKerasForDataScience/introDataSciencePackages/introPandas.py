import pandas as pd


# Data structures in Pandas:
    # 1. Series: Represents data in a 1D form
    # 2. Data frames: Represent data in a 2D tabular form


# Pandas Series: One-dimensional array with labels

# Obtain series from Python dictionaries 
dict = {'a': 3, 'b': 'cat', 'c': 2.5}
print(pd.Series(dict))

oneD = pd.Series([100, 'cat', 310, 'gog', 500], ['Ammy', 'Bobby', 'Cat', 'Don', 'Emma'])   # Second list contains labels
print("oneD pandas series:\n" + str(oneD))

# loc is a label-location based indexer
print(oneD.loc[['Cat', 'Emma']])   # Selection by labels-Cat and Emma
print(oneD[[0, 3, 4]])  # Extract data at index 0, 3 and 4

# iloc is the integar-position based (from 0 to length-1 of the axis). access index 1
print(oneD.iloc[1])

# Check if there is cat in the series index
print(" 'cat' in series index ?:", 'cat' in oneD)
print(" 'Cat' in series index ?:", 'Cat' in oneD)



# DataFrames- 2D data structure. Stores data in tabular form (rows and columns)
# <class 'pandas.core.frame.DataFrame'>
d = {'A' : pd.Series([100., 200., 300.], index=['apple', 'pear', 'orange']),
     'B' : pd.Series([111.,222.,333.,4444.], index=['apple', 'pear', 'orange', 'melon'])}

df = pd.DataFrame(d)     # df stands for a dataFrame
print('dataFrame:', df)
print("of type:", type(df))
print("List of indices:", df.index)
print("List of columns:", df.columns)

# Specify which row/index and column we want to retain
print(pd.DataFrame(df, index=['orange', 'melon', 'apple'], columns=['A']))


# Read in CSV files
import os 
csvFilename = 'sampleCSVFile.csv'
csvFile = os.getcwd() + "\\introDataSciencePackages\\" + csvFilename   # creating a string of global file path 
dfFromCSV = pd.read_csv(csvFile)
print(dfFromCSV.head())


# Read in xls files
xlsFilename = 'sampleXLSFile.xls'
xlsFile = os.getcwd() + "\\introDataSciencePackages\\" + xlsFilename   # creating a string of global file path 
xl = pd.ExcelFile(xlsFile)
print("Sample excel file sheet name:", xl.sheet_names)
dfFromExcel = xl.parse('Sheet1')     # Loading a sheet into a DataFrame- dfFromExcel
print(dfFromExcel.head())


# Data cleaning using Pandas


dataFrame2 = pd.read_csv("introDataSciencePackages/titanic.csv")
print(dataFrame2.head())

train_df = dataFrame2.drop(['PassengerId', 'Name', 'Ticket'], axis=1)   # drop specific columns
print(train_df.head())

print("Train")
print(train_df.isnull().sum())   # NaN values count

# Fill in nans with mean values
maleMeanAge = train_df[train_df["Sex"]=='male']['Age'].mean()
femaleMeanAge = train_df[train_df["Sex"]=='female']['Age'].mean()
print("Male mean age: %1.0f" %maleMeanAge)
print("Female mean age: %1.0f" %femaleMeanAge)

# Set the nan values with mean values
train_df.loc[ (train_df['Sex'] == 'male') & (train_df['Age'].isnull()), 'Age'] = maleMeanAge
train_df.loc[ (train_df['Sex'] == 'female') & (train_df['Age'].isnull()), 'Age'] = femaleMeanAge
meanFare = train_df['Fare'].mean()

train_df["Cabin"] = train_df["Cabin"].fillna("X")
train_df["Embarked"] = train_df["Embarked"].fillna("S")

train_df.to_csv("introDataSciencePackages/cleanedUpTitanic.csv")