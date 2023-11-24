import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section5"
filePath += "\\titanic.csv"
df = pd.read_csv(filePath)

# Data cleaning 
# Drop unrequired columns 
train_df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
nullValueCount = train_df.isnull().sum()   # Computes # of NANs in each feature column

# replace NAs with mean values
# Identify mean ages
male_mean_age = train_df[train_df['Sex'] == 'male']['Age'].mean()
female_mean_age = train_df[train_df['Sex'] == 'female']['Age'].mean()

# fill the nan values
train_df.loc[ (train_df['Sex'] == 'male') & (train_df['Age'].isnull()), 'Age'] = male_mean_age
train_df.loc[ (train_df['Sex'] == 'female') & (train_df['Age'].isnull()), 'Age'] = female_mean_age

# Mean fare
mean_fare = train_df['Fare'].mean()

# Replace NANs in Cabin and Embarked with X and S
train_df['Cabin'] = train_df['Cabin'].fillna('X')
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# Response
y = train_df.Survived

# Predictors
x = train_df.drop(['Survived'], axis=1) 

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=101)

# Tensorflow instantiation of Feature Columns
    # numeric_column: it defines that the feature will be a float32 number
    # bucketized_column: It defines a feature that will be bucketized. You can define the range of the bucket.
    # categorical_column_with vocabulary_list: As the name says, it does a one-hot-encoding for the column using a vocabulary list.
    # categorical_column_with_hash_bucket: It encodes the categorical values using hash bucket. You define the number of hashes it will have.

# Define numeric columns
pclass_feature = tf.feature_column.numeric_column('Pclass')
parch_feature = tf.feature_column.numeric_column('Parch')
age_feature = tf.feature_column.numeric_column('Age')
fare_feature = tf.feature_column.numeric_column('Fare')

# Define buckets for children, teens, adults and elders
age_bucket_feature = tf.feature_column.bucketized_column(age_feature, [12, 21, 60])

# Define a categorical colum with predicted values
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['female', 'male']
)

# Define a categorical column with dynamic values
embarked_feature = tf.feature_column.categorical_column_with_hash_bucket('Embarked', 3)

cabin_feature = tf.feature_column.categorical_column_with_hash_bucket('Cabin', 100)


feature_columns = [ pclass_feature, age_feature, age_bucket_feature, parch_feature, 
                   fare_feature, embarked_feature, cabin_feature]

# Linear Classifier Estimator
estimator = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Train input function
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,
                                                               y=y_train,
                                                               num_epochs=None,   # It uses the necessary number of epochs
                                                               shuffle=True,
                                                               target_column='target',
                                                               )

estimator.train(input_fn=train_input_fn, steps=450)

# Test on training data
test_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train,
                                                                    y=y_train,
                                                                    batch_size=10,
                                                                    num_epochs=1,   
                                                                    shuffle=False
                                                                    )
results = estimator.evaluate(test_train_input_fn)

eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test,
                                                                y=y_test,
                                                                batch_size=10,
                                                                num_epochs=1,   
                                                                shuffle=False
                                                                )
test_results = estimator.evaluate(eval_input_func)

