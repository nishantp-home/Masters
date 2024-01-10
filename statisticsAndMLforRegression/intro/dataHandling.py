import pandas as pd


# Read data from CSV file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section3'
filePath += '\\endangeredLang.csv'
data = pd.read_csv(filePath)


print(data[['Countries', 'Name in English']])   # Isolate 2 column
print(data[3:10])    # Isolate rows
print(data[['Countries', 'Name in English']][3:10]) # Isolate rows and columns
italyData = data[data['Countries'] == 'Italy'] # Isolate by row names
italyGermanyData = data.loc[data['Countries'].isin(['Italy', 'Germany'])] # Isolate two row labels

# Conditional selection
selectedData = data[data['Number of speakers'] < 5000]

# Basic grouping
large = data.sort_values(by='Number of speakers', ascending=False)  # Sorting data 

byStatus = data.groupby('Degree of endangerment')
byStatus.count()

byTwoAttributes = data.groupby(['Countries', 'Degree of endangerment'])
x = data['Number of speakers'].groupby(data['Countries'])

# Implementing lambda functions

# Identify which languages have lesser than 5k speakers
data['vfew'] = data['Number of speakers'].apply(lambda x: x <= 5000)   # adds a column with boolean values
group_by_5k = data.groupby(['vfew'])
print(group_by_5k.size())


# Some more grouping 
group_by_5kc = data.groupby(['vfew', 'Countries'])
print(group_by_5kc.size())
data['Degree of endangerment'].value_counts()

# Cross tabulation
data2 = data.rename(columns={data.columns[3]: 'CountryCode', data.columns[4]: 'Endangerment'})  # rename column name to better one
crossTabulatedData = pd.crosstab(data2["CountryCode"], data2['Endangerment'],margins=True)

# Remove rows or columns
data.drop(data.index[:2], inplace=True)  # Romove first 2 rows
data.drop(['Latitude'], axis=1, inplace=True) # Drop column 'Latitude'

# Melting
patient = pd.DataFrame({'FirstName': ['Bill', 'Jane'], 
                        'LastName' : ['Shakespeare', 'Austen'],
                        'BloodType': ['o+', 'B+'],
                        'Wt': [85, 62]})

pd.melt(patient, id_vars=['FirstName', 'LastName'], var_name='Measurements')