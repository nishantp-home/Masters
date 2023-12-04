# Implement Wide & Deep learning
    # The wide model is able to memorize interactions with data with a large number of features
    # but not able to generalize these learned interactions on new data
    # The deep model generalizes well but is unable to learn exceptions within data
    # The wide and deep model combines the two models and is able to generalize while learning exceptions

import time
import numpy as np
import tensorflow as tf
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

# clean data
del X["fnlwgt"]
y = y.income.apply(lambda x: ">50K" in x).astype(int)   # Convert to  binary response variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
BATCH_SIZE = 40
num_epochs = 1
shuffle = True


CATEGORICAL_COLUMNS = ["workclass", "education",
                       "marital-status", "occupation",
                       "relationship", "race",
                       "sex", "native-country"]

# Columns of the input csv file
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", 
           "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# Feature columns for input into the model
FEATURE_COLUMNS = ["age", "workclass", "education", "education-num",
                   "marital-status", "occupation", "relationship", "race", 
                   "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]


# Make data input ready
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    num_epochs=num_epochs,
    shuffle=shuffle)

test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=y_test,
    batch_size=BATCH_SIZE,
    num_epochs=num_epochs,
    shuffle=shuffle)

# Categorical base columns
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    key="sex",
    vocabulary_list=["female", "male"])

race = tf.feature_column.categorical_column_with_vocabulary_list(
    key="race",
    vocabulary_list=["Amer-Indian-Eskimo", 
                     "Asian-Pac-Islander", 
                     "Black", "Other", "White"])

education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)

marital_status = tf.feature_column.categorical_column_with_hash_bucket(
    "marital-status", hash_bucket_size=100)

relationship = tf.feature_column.categorical_column_with_hash_bucket(
    "relationship", hash_bucket_size=100)

workclass = tf.feature_column.categorical_column_with_hash_bucket(
    "workclass", hash_bucket_size=100)

occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)

native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native-country", hash_bucket_size=1000)

# Continuous base columns
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education-num")
capital_gain = tf.feature_column.numeric_column("capital-gain")
capital_loss = tf.feature_column.numeric_column("capital-loss")
hours_per_week = tf.feature_column.numeric_column("hours-per-week")

# The wide columns are sparse, categorical columns that we specified, as well as our hashed, 
# bucket, and feature crossed columns

# wide columns and deep columns
wide_columns = [sex, race, native_country, 
                education, occupation, workclass,
                marital_status, relationship]

deep_columns = [
    # Multi-hot indicator columns for columns with fewer possibilities
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(sex),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.indicator_column(race),
    # Embeddings for categories with more possibilities. Should have atleast (possibilities)**(0.25) dims
    tf.feature_column.embedding_column(education, dimension=8),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    # Numerical columns
    age, education_num, capital_gain, capital_loss, hours_per_week,]

# wide: Linear Classifier

# Deep: Deep Neural Net Classifier
# Wide & Deep: Combined Linear and Deep Classifier

def create_model_dir(model_type):
    return 'model/model_' + model_type + '_' + str(int(time.time()))

# If new_model= False, pass in the desired model_dir
def get_model(model_type, wide_columns=None, deep_columns=None, new_model=False, model_dir=None):
    if new_model or model_dir is None:
        model_dir=create_model_dir(model_type)
    print("Model directory = %s" % model_dir)

    m = None

    # Linear Classifier
    if model_type == "WIDE":
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir,
            feature_columns=wide_columns)
        
    # Deep Neural Network
    if model_type == "DEEP":
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])
        
    # Combined Linear and Deep Classifier
    if model_type == "WIDE_AND_DEEP":
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 70, 50, 25]   # 4 layers
        )
    
    print('estimator built')

    return m, model_dir


MODEL_TYPE = "WIDE_AND_DEEP"
model_dir = create_model_dir(model_type=MODEL_TYPE)
m, model_dir = get_model(model_type=MODEL_TYPE, wide_columns=wide_columns, deep_columns=deep_columns, model_dir=model_dir)



m.train(input_fn=train_input_fn)
results = m.evaluate(input_fn=test_input_fn)
print("Evaluation done")
print("\nAccuracy: %s" % results['accuracy'])