# Softmax regression (or multinomial logistic regression) is a generalization of 
# logistic regression to hte case where we want to handle multiple classes

import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.utils
from sklearn.model_selection import train_test_split


filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section9"
filePath += "\\Iris.csv"
df = pd.read_csv(filePath, usecols=[1,2,3,4,5])
#df.describe()

# Modify column names
# map species names
df.columns = ['f1', 'f2', 'f3', 'f4', 'f5']
# map data into arrays
s = np.asarray([1, 0, 0])
ve = np.asarray([0, 1, 0])
vi = np.asarray([0, 0, 1])
# Hot encoding
df['f5'] = df['f5'].map({'Iris-setosa': s, 'Iris-versicolor': ve, 'Iris-virginica': vi})

# Shuffle Pandas data frame
df = sklearn.utils.shuffle(df)

# Reser the index column 
df = df.reset_index(drop=True)

# Define predictors and response variables
x_input = df.loc[:, ['f1', 'f2', 'f3', 'f4']] # Predictors
y_input = df['f5'] #df.loc[:, ['f5']]

X_train, X_test, y_train, y_test = train_test_split(x_input, y_input, test_size=0.30, random_state=42)

x_input = X_train
y_input = y_train

# placeholders and variables. Input has 4 features, output has 3 classes
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

# Weights and Bias
W = tf.Variable(tf.compat.v1.random_normal([4,3]))
b = tf.Variable(tf.compat.v1.random_normal([3]))

# Softmax function for multiclass classification
y = tf.compat.v1.nn.softmax(tf.matmul(x, W) + b)

# Loss funciton
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.compat.v1.log(y), axis=1))

# Optimizer
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)

# Calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Session parameters
sess = tf.compat.v1.Session()

#initializing variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

# number of iterations
epoch = 2000

for step in range(2, epoch):
    _, c = sess.run([train_step, cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input]})
    if step%500 ==0:
        print(c)

print("Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y_: [t for t in y_test]}))