# Correlation
# Test the strength of association b/w quantitative variables

import seaborn as sns
import matplotlib.pyplot as plt

i = sns.load_dataset('iris')
corr = i.corr()

plt.scatter(i['sepal_length'], i['petal_width'],
            marker='x',
            color='b',
            alpha=0.7,
            s=124)

plt.title('Sepal length Vs Petal width')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.show()


# Visualize the correlation-corr plot
from statsmodels import api as sm

sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


import pandas as pd
# Work with real data
# Read data from CSV file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section4'
filePath += '\\happy2015.csv'
df = pd.read_csv(filePath)
df3 = df[['Happiness Score','Family', 
          'Economy (GDP per Capita)', 'Freedom', 'Generosity']]


corr = df3.corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()