import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

# Load data from csv file
filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section9"
filePath += "\\titanic.csv"
df = pd.read_csv(filePath)

# dropping unrequired columns 
train_df = df.drop(["Name", "Ticket", "PassengerId"], axis=1)

# Fill in NANs

# identify mean ages 
male_mean_age = train_df[train_df["Sex"] == "male"]["Age"].mean()
female_mean_age = train_df[train_df["Sex"] == "female"]["Age"].mean()

# Fill NaN values
train_df.loc[(train_df["Sex"]=="male") & (train_df["Age"].isnull()), "Age"] = male_mean_age
train_df.loc[(train_df["Sex"]=="female") & (train_df["Age"].isnull()), "Age"] = female_mean_age

mean_fare = train_df["Fare"].mean()

train_df["Cabin"] = train_df["Cabin"].fillna("X")
train_df["Embarked"] = train_df["Embarked"].fillna("S")

y = train_df.Survived # response
x = train_df.drop(["Survived"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=101)


# Define inputs## 
# defining numeric columns
pclass_feature = tf.feature_column.numeric_column('Pclass')
age_feature = tf.feature_column.numeric_column('Age')
fare_feature = tf.feature_column.numeric_column('Fare')
parch_feature = tf.feature_column.numeric_column('Parch')

# defining buckets for children, teens adults and elders
age_bucket_feature = tf.feature_column.bucketized_column(age_feature, [12, 21, 60])

# Defining categorical column with predefined values
sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(
    'Sex', ['female', 'male']
)

# Defining a categorical column with dynamic values
embarked_feature = tf.feature_column.categorical_column_with_hash_bucket(
    'Embarked', 3
)

cabin_feature = tf.feature_column.categorical_column_with_hash_bucket(
    'Cabin', 100
)

# DNN doesn't support categorical with hash buckets
embarked_embedding = tf.feature_column.embedding_column(
    categorical_column=embarked_feature,
    dimension=3
)

cabin_embedding = tf.feature_column.embedding_column(
  categorical_column=cabin_feature,
  dimension=300
)

feature_columns = [ pclass_feature, age_feature, age_bucket_feature, parch_feature,
                   fare_feature, embarked_embedding, cabin_embedding]


# Instantiate the DNN estimator
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 30, 10]   # 3 hidden layers
)

# Train input function
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    num_epochs=None,   # It can use automatically the necessary number of epochs
    shuffle=True,
    target_column='target',
)

estimator.train(input_fn=train_input_fn, steps=1000)

# Test on training data
eval_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x = X_train,
    y=y_train,
    batch_size=10, 
    num_epochs=1, 
    shuffle=False
)
train_eval_results = estimator.evaluate(eval_train_input_fn)

eval_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=y_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False
)

results = estimator.evaluate(eval_input_fn)



