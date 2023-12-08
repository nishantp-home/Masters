# Binary response variable
# Xs can be numerical or categorical

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# Load data from csv file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section5\\'
filePath += "trainT.csv"    # titanic data set
df = pd.read_csv(filePath)


# Check for data cleaning
#df.shape
#df.isnull().sum()

df = df[['Survived', 'Pclass', 'Age', 'Fare']]
df = df.dropna()

plt.figure(figsize=(6,4))
fig, ax = plt.subplots()
df.Survived.value_counts().plot(kind='barh', color='blue', alpha=0.65)
plt.title("Survival Breakdown (1: Survived, 0: Died)")


# create response and predictors
y = df[['Survived']]
x = df[['Pclass', 'Age', 'Fare']]

# Make the model
logit = sm.Logit(y, x.astype(float))

# fit the model
result = logit.fit()
# Equation from results.summary()
# Log [p/(1-p)] = -0.28*Pclass + 0.0146*Fare - 0.0108*Age

# How a 1 unit increase or decresse in a variable affects the odds of surviving
# Number of successes: 1 failure

# odds
print(np.exp(result.params))
#Pclass    0.755585
#Age       0.989267    # odds that passengers die increase by a factor of 0.98 for each unit change in age.
#Fare      1.014718

# probability = odds/(1+odds)



# Another example
from patsy import dmatrices
import statsmodels.discrete.discrete_model as smdm

df2 = pd.read_csv(filePath)

df2 = df2[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df2 = df2.dropna()

# C indicates categorical
y, X = dmatrices('Survived ~ C(Pclass) + C(Sex) + Age + Fare', df2, return_type= 'dataframe')

# sklearn output
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(fit_intercept=False, C=1e9)
md1 = model.fit(X, y)
model.coef_

logit = smdm.Logit(y, X)
results = logit.fit()
results.params
results.summary()