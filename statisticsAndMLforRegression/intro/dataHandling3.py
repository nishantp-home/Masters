# Concatenation of data

import pandas as pd
import numpy as np

df1 = pd.DataFrame([['a',1], ['b', 2]], 
                   columns=['letter', 'number'])

df2 = pd.DataFrame([['c',3], ['d', 4]], 
                   columns=['letter', 'number'])

df3 = pd.DataFrame([['c',3, 'cat'], ['d', 4, 'dog']], 
                   columns=['letter', 'number', 'animal'])

dfConcat = pd.concat([df1, df2])
dfConcat2 = pd.concat([df1, df3])   # Fill non-existing values with NaNs
dfConcat3 = pd.concat([df1, df3], join='inner')   # only columns common for both data 
                                                  # frames are joined together


# Read data from CSV file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section3'
foodFilePath = filePath + '\\starbucks-food2.csv'
drinksFilePath = filePath + '\\starbucks-menu-nutrition-drinks.csv'
food = pd.read_csv(foodFilePath)
drinks = pd.read_csv(drinksFilePath)

# Make columns consistent
food.columns = ['Items', 'Calories', 'Fat', 'Carb', 'Fiber', 'Protein']
drinks.columns = ['Items', 'Calories', 'Fat', 'Carb', 'Fiber', 'Protein', 'Na']
concat = pd.concat([drinks, food])

# Merge data
gfpFilePath = filePath + '\\GlobalFirePower.csv'
gdpFilePath = filePath + '\\countryGDP.csv'
gfp = pd.read_csv(gfpFilePath)
gdp = pd.read_csv(gdpFilePath)

merged = pd.merge(gdp, gfp, on='Country')  # Merge on column with common entries
# default 'inner values

# Merge selected columns
y = pd.merge(gdp, gfp[['Country', 'Manpower Available', 'Railway Coverage (km)']], on='Country')

# Database type merging

# Keep every row in the left data frame. Where there are missing values of the "on" variable
# in the right data frame, fill with Nan
y = pd.merge(gdp, gfp[['Country', 'Manpower Available', 
                       'Railway Coverage (km)']], 
                       on='Country', how='left')   # Similarly for 'right


# A full outer join returns all the rows from left df
# all the rows from the right df, and matches up rows wherever possible with NaNs elsewhere,
# indicator tells how the joining was done
y = pd.merge(gdp, gfp[['Country', 'Manpower Available', 
                       'Railway Coverage (km)']], 
                       on='Country', how='outer', indicator=True)
