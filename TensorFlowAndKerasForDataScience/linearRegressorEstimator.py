import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os 

plt.style.use("seaborn-colorblind")
from sklearn.model_selection import train_test_split

used_features = ['property_type','room_type','bathrooms','bedrooms','beds','bed_type','accommodates','host_total_listings_count'
                ,'number_of_reviews','review_scores_value','neighbourhood_cleansed','cleaning_fee','minimum_nights','security_deposit','host_is_superhost',
                 'instant_bookable', 'price']


## Use file path correctly 
filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section5"
filePath += "\\listings.csv"

boston = pd.read_csv(filePath, usecols=used_features)  #actual data
print(boston.shape)
#print(boston.head(10))

for feature in ['cleaning_fee', 'security_deposit', 'price']:
    boston[feature] = boston[feature].map(lambda x: x.replace("$", '').replace(",", ''), na_action='ignore')
    # remove $ sign from prices
    boston[feature] = boston[feature].astype(float)
    boston[feature].fillna(boston[feature].median(), inplace=True)   # fill in missing vlaues with median values

#Fill in NAs
for feature in ["bathrooms", "bedrooms", "beds", "review_scores_value"]:
    boston[feature].fillna(boston[feature].median(), inplace=True)

boston['property_type'].fillna('Apartment', inplace=True)

# Response variable
boston = boston[(boston["price"]>50) & (boston["price"]<500)]
target = np.log(boston.price)

# predictors
features = boston.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)  ## split dataset

# Get all the numeric feature names
numeric_columns = ['host_total_listings_count','accommodates','bathrooms','bedrooms','beds', 
                   'security_deposit','cleaning_fee','minimum_nights','number_of_reviews',
                   'review_scores_value']

# Get all the categorical feature names that contains strings
categorical_columns = ['host_is_superhost','neighbourhood_cleansed','property_type',
                       'room_type','bed_type','instant_bookable']


# Convert numerical data to feature columns
numeric_features = [tf.feature_column.numeric_column(key=column) for column in numeric_columns]

# Convert categorical data to feature columns
categorical_features = [tf.feature_column.categorical_column_with_vocabulary_list(key=column, vocabulary_list=features[column].unique()) for column in categorical_columns]

linear_features = numeric_features + categorical_features

# Create training input function
training_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,
                                                                  y=y_train,
                                                                  batch_size=32,
                                                                  shuffle=True,
                                                                  num_epochs=None)

# Create testing input function
eval_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,
                                                              y=y_test,
                                                              batch_size=32,
                                                              shuffle=False,
                                                              num_epochs=1)
                                    
# Linear regressor estimator
linear_regressor = tf.estimator.LinearRegressor(feature_columns=linear_features, model_dir="linear_rgressor")
linear_regressor.train(input_fn=training_input_fn, steps=2000)

# Evaluate on testing data
linear_regressor.evaluate(input_fn=eval_input_fn)

pred = list(linear_regressor.predict(input_fn=eval_input_fn))   # predict on testing data
pred = [p['predictions'][0] for p in pred]

prices = np.exp(pred)
print(prices)

