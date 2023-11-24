import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as skds
from sklearn import datasets
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

boston = load_boston()      # Inbuilt dataset
print(boston.DESCR)         # description of the dataset

X = boston.data.astype(np.float32)
y = boston.target.astype(np.float32)
if (y.ndim == 1):
    y = y.reshape(-1, 1)
X = StandardScaler().fit_transform(X)   # Standardize features by removing mean and scaling to unit variance


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
#print(X_train.shape)


num_outputs = y_train.shape[1]
num_inputs = X_train.shape[1]

tf.compat.v1.disable_eager_execution()   # tf.compat.v1 is the compatible api version with the specific methods

x_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_inputs], name='x')
y_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='y')

w = tf.Variable(tf.zeros([num_inputs, num_outputs]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros([num_outputs]), dtype=tf.float32, name='b')
model = tf.matmul(x_tensor, w) + b

# Define loss, MSE and R2
loss = tf.reduce_mean(tf.square(model-y_tensor))
mse = loss
y_mean = tf.reduce_mean(y_tensor)
total_error = tf.reduce_sum(tf.square(y_tensor - y_mean))
unexplained_error = tf.reduce_sum(tf.square(model - y_tensor))
rs = 1 - tf.divide(unexplained_error, total_error)

learning_rate = 0.001
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

num_epochs = 1500
loss_epochs = np.empty(shape=[num_epochs], dtype=np.float32)
mse_epochs = np.empty(shape=[num_epochs], dtype=np.float32)
rs_epochs = np.empty(shape=[num_epochs], dtype=np.float32)

mse_score = 0.0
rs_Score = 0.0

with tf.compat.v1.Session() as tfs:
    tfs.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(num_epochs):
        feed_dict = {x_tensor: X_train, y_tensor: y_train}
        loss_val, _ = tfs.run([loss, optimizer], feed_dict)
        loss_epochs[epoch] = loss_val

        feed_dict = {x_tensor: X_test, y_tensor: y_test}
        mse_score, rs_score = tfs.run([mse, rs], feed_dict)
        mse_epochs[epoch] = mse_score
        rs_epochs[epoch] = rs_score

print('For test data: MSE = {0:.8f}, R2 = {1:.8f}'.format(mse_score, rs_score))

