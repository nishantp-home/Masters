import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from csv file
filePath = 'e:\\Eskills-Academy-projects\\StatisticsAndMLforRegressionData\\section5\\'

trainFileName = "trainT.csv"    # titanic data set
trainFile = filePath + trainFileName
train = pd.read_csv(trainFile)

testFileName = 'testT.csv'
testFile = filePath + testFileName
test = pd.read_csv(testFile)

# If test data is not available
# Split dataset into training and testing data
# Train-test split could be e.g. 80%-20%, 75%-25% 

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load the Diabetes Housing dataset
columns = "age sex bmi map tc ldl hdl tch ltg glu".split()  # declare column names
diabetes = datasets.load_diabetes() 
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Splitting data can lead to unstable/unreliable results when we have a small dataset
    # Use Cross-validation for better results
# Cross-Validation
    # Divide your data into folds (each fold is container that holds an even distribution of cases)
    # usually 5 fold CV or 10 fold CV
    # Hold out one fold as test set, and use the others as train sets
    # Train and record the test set result
    # Perform tranining and recording using each fold in teurn as a test set
    # Calculate the average and the standard deviation of all the fold's test results


from sklearn.model_selection import cross_val_score

# Load the inbuilt iris-dataset from sklearn
iris_data = datasets.load_iris()

data_input = iris_data.data
data_output = iris_data.target

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
X = np.arange(10)
for train_set, test_set in kf.split(X):
    print(train_set, test_set)