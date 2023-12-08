import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model


# Load dataset
iris = sns.load_dataset('iris')

X = iris.petal_length   # predictor
y = iris.petal_width    # response


# Linear regression
model = sm.OLS(y, X)
results = model.fit()
# Statsmodels gives R-like statistical output

X2 = np.vander(X, 2)  # adds a constant column for intercepts

model2 = sm.OLS(y, X2)
results2 = model2.fit()

# Intercept = 0.41
# Slope = -0.36
# Petal_width = 0.41 - 0.36*(Petal_length)


# Multilinear regression
X = iris[['petal_length', 'sepal_length']]
X = sm.add_constant(X)    # adds a column of contants for intercept
y = iris['petal_width']

model = sm.OLS(y, X)
results = model.fit()
# results.summary()


# Use categorical variables
iris = sns.load_dataset('iris')
dummies = pd.get_dummies(iris['species'])
# Add to the original dataframe
iris2 = pd.concat([iris, dummies], axis=1) #assign numerical values to different species

X = iris2[['petal_length', 'sepal_length', 'setosa', 'versicolor', 'virginica']]
X = sm.add_constant(X)
y = iris2['petal_width']
model = sm.OLS(y, X)
results = model.fit()
# results.summary()


# Fit the linear model using sklearn
model = linear_model.LinearRegression()
results = model.fit(X, y)
print(results.intercept_, results.coef_)


# Conditions of linear regression
# Linear relationship between Y and Xs
sns.pairplot(iris[['petal_width', 'petal_length', 'sepal_length']].dropna(how='any', axis=0))

# Multilinear regression
X = iris[['petal_length', 'sepal_length']]
X = sm.add_constant(X)    # adds a column of contants for intercept
y = iris['petal_width']

model = sm.OLS(y, X)
results = model.fit()


# Are the residuals normally distributed ?
# Jarque-Bera (JB) test: test for normal distribution of residuals
## H0: The null hypothesis for the test is that the data is normally distributed (in this case residuals)

# Unfortunately, with small samples, the JB test is prone rejecting the null hypothesis
# that the distribution is normal- when it is infact true

# prob(JB) = 0.0357 (< 0.05) implies residuals are not normally distributed


# QQ-Plot the residuals 
res = results.resid
sm.qqplot(res)  # a linear correlation with slope 1 implies normally distributed residuals

# Durbin-watson: used for measuring autocorrelation, 
# approximately equal to 2(1-r), where r is the sample autocorrelation
# ranges from (0-4):
    # a value around 2 suggests no autocorrelation
    # values > 2 suggest negative correlation
    # values < 1 suggest positive correlation  

    # Our DB value = 1.414, implies no issue of autocorrelation

# Multicollinearity
#   condition number (cond. no.) : used for measuring multicollinearity
#   cond. no. > 30 implies multicollinearity
#   influences the stability and reliability of coefficients
    # if correlation between predictors exist, multicollinearity exists
    # our cond. no. = 80.7, implies our predictors are highly correlated with each other

corr = X.corr()   # correlation bw predictors
# rule of thumb: drop a predictor if correlation value > 0.75
# Our value = 0.871754, use one predictor among petal_length and sepal_length


# Heteroscedasticity
    # Test whether the variance of the errors from a regression is dependent on 
    # the values of the independent variables
    # There should be no relation between residuals and fitted values, i.e. we want homoscedasticity
    # Breusch-pagan (BP) test
    # H0: null hypothesis of the BP test is homoscedasticity.


import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']   # List of measures computed

test = sms.het_breuschpagan(results.resid, results.model.exog)
lzip(name, test)
#[('Lagrange multiplier statistic', 23.53447941749097),
# ('p-value', 7.754480840216067e-06),
# ('f-value', 13.67791180725054),
# ('f p-value', 3.5665183433603395e-06)]

# Our p-value < 0.05, implies we reject the null hypothesis that 
# the variance of the residuals is constant, and infer that heteroscedasticity
# is indeed present


# Influence test
# Plot helps us to find influential cases (i.e. subjects) if any. 
# Not all outliers are influential in linear regression analysis
#

from statsmodels.graphics.regressionplots import * 
plot_leverage_resid2(results)
influence_plot(results)