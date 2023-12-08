import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import seaborn as sns
from scipy.stats import norm, shapiro, skewnorm, zscore
from sklearn import datasets

# Load data
data = datasets.load_iris()   # load the inbuilt iris dataset
df = pd.DataFrame(data.data, columns=data.feature_names)

# Important methods
statCount = df.count()
statDesc = df.describe()    # summary statistics
# Visual box plots
df.boxplot()

iris = sns.load_dataset('iris')   # inbuilt seaborn dataset


# Grouping and Summarizing data
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section4'
fileName = "\\cwurData.csv"
file = filePath + fileName
uni = pd.read_csv(file)
# uni.count(), uni.describe()

# grouping
groupDf = uni.groupby('country')
# groupDf.describe()

g1 = uni.groupby(['country', 'influence']).count()
x = pd.DataFrame(groupDf.size().reset_index(name="InfluenceAggregate"))


fileName2 = '\\rainfall1901_2015.csv'
file2 = filePath + fileName2
r = pd.read_csv(file2)
m1 = r.groupby('SUBDIVISION').mean()
m2 = r.groupby(['SUBDIVISION', 'ANNUAL']).mean()  # average ANNUAL rainfall for different subdivisions
g2 = r.groupby(['SUBDIVISION', 'YEAR', 'ANNUAL']).size().to_frame(name='count').reset_index()


# Check for normal distribution
    # Normal distributions have a bell-shaped curve
    # The mean, median and mode of a normal distribution are equal. The area under the normal curve is one.
    # An important condition for most common (parametric) statistical tests

# By plotting 
x = np.linspace(-10, 10, 1000)
y = norm.pdf(x, loc=2.5, scale=1.5)
pylab.plot(x,y)
pylab.show()

a = 4
fig, ax = plt.subplots(1, 1)
r = skewnorm.rvs(a, size=1000)
ax.hist(r, histtype='stepfilled', alpha=0.2, density=True)
ax.legend(loc='best', frameon=False)
plt.show()

# Check for data normality using other techniques
i = sns.load_dataset('iris')   # inbuilt seaborn dataset
sl = i['sepal_length']
sl.hist()

# Shapiro-Wilks test
    # Test for normality
    # H0 : Indicator, if the data are normally distributed or not
    # Accepted H0 is p > 0.05
shapiro_results = shapiro(sl)



# Standard Normal distribution and Z-score
    # Z-score is a standard score: z = (value-mean)/standard deviation
    # standard normal distribution (or the unit normal distribution) is 
    # a special normal curve made up of z-scores
    # std normal distribution has a mean=0 and sd=1

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu-3*variance, mu+3*variance, 100)
y = norm.pdf(x=x, loc=mu, scale=variance)
plt.plot(x, y)
plt.show()



# Z-scores
    # Generate data frame with five rows 
    # and three columns and values between 100-200
df = pd.DataFrame(np.random.randint(100, 200, size=(5,3)), 
                  columns=['A', 'B', 'C'])

df.apply(zscore)     #Compute Z-scores



# Confidence interval
    # Point estimates: Estimates of population parameters based on sample data.
                    #  If we want to know the average weight of students in a school, we will collect a sample
                    # to estimate the average weight of all the students in a school
    # The sample mean is usually not exactly equal to population mean
    # Confidence interval (CI): It is a range of values above and below a point estimate that captures 
    #                        the true population parameter at some predetermined confidence interval

    # CI = Mean +-  Margin of error
    # Margin of error (MOE) = Z*sd/sqrt(n)


import random
import scipy.stats as stats

np.random.seed(10)

# Randomly create population weight data for 150k people
population_wt = stats.poisson.rvs(loc=18, mu=35, size=150000)

sample_size = 1000
sample = np.random.choice(a=population_wt, size=sample_size)
sample_mean = sample.mean()   # Mean of sample of sample_size (=1000)
                              # Point estimate

z_critical = stats.norm.ppf(q=0.975)  # Get the z-critical value at 95% level
# z-value at 95% I is 1.96
# q = (1+0.95) / 2


pop_stdev = population_wt.std()  # Get population standard deviation
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error, 
                       sample_mean + margin_of_error)

from numpy import average, std
from math import sqrt
from scipy.stats import t, sem

s_stdev = sample.std()

# Major difference b/w Z-score and T statistic is
# that z-score needs a population standard deviation
# The T test is also used if you have small sample size (e.g. < 30)


# E.g. data: Weight of weight-lifters
data=[63.5, 81.3, 88.9, 63.5, 76.2, 67.3, 66.0, 64.8, 74.9, 81.3, 76.2,
      72.4, 76.2, 81.3, 71.1, 80.0, 73.7, 74.9, 76.2, 86.4, 73.7, 81.3,
      68.6, 71.1, 83.8, 71.1, 68.6, 81.3, 73.7, 74.9]

mean = average(data)   # Point estimate: mean weight

confidence = 0.95
n = len(data)
m = mean
std_err = sem(data)     # Standard error

h = std_err * t.ppf((1+confidence) / 2, n - 1)

t_interval = (m - h, 
              m + h)