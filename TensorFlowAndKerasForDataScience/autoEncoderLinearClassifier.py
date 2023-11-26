# Implements an autoencoder on a dataset with 2 classes
# Autoencoder: Unsupervised learning technique

import numpy as np
import pandas as pd
import tensorflow as tf

# Import data in a pandas dataframe
filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section5"
filePath += "\\creditcard.csv"
df = pd.read_csv(filePath)

# Esplore data
fraud_indices = df[df['Class'] == 1].index
number_records_fraud = len(fraud_indices)
normal_indices = df[df['Class'] == 0].index
number_records_normal = len(normal_indices)

# Split dataset in training (75%) and test(25%) datasets
train_Set = df.sample(frac=0.75, replace=False, random_state=123)
test_Set = df.loc[set(df.index) - set(train_Set.index)]   

from sklearn.preprocessing import MinMaxScaler  # data scaling

scaler = MinMaxScaler()
scaler.fit(df.drop(['Class', 'Time'], axis=1))

# Scale the predictors
scaled_data = scaler.transform(train_Set.drop(['Class', 'Time'], axis=1))
scaled_test_data = scaler.transform(test_Set.drop(['Class', 'Time'], axis=1))

num_inputs = len(scaled_data[1])   # number of columns
num_hidden = 2   # number of hidden layers
num_outputs = num_inputs

learning_rate = 0.001
keep_prob = 0.5    
tf.compat.v1.reset_default_graph()


# define placeholders
tf.compat.v1.disable_eager_execution()

# placeholder X
X = tf.compat.v1.placeholder(tf.float32, shape=[None, num_inputs])  # Column vector

# Weights
initializer = tf.compat.v1.variance_scaling_initializer()
w = tf.compat.v1.Variable(initializer([num_inputs, num_hidden]), dtype=tf.float32)
w_out = tf.Variable(initializer([num_hidden, num_outputs]), dtype=tf.float32)

# bias 
b = tf.Variable(tf.zeros(num_hidden))
b_out= tf.Variable(tf.zeros(num_outputs))

# Activation
act_func = tf.nn.tanh

# Layers
hidden_layer = act_func(tf.matmul(X, w) + b)
dropout_layer = tf.compat.v1.nn.dropout(hidden_layer, keep_prob=keep_prob)
output_layer = tf.matmul(dropout_layer, w_out) + b_out


# Loss function
loss = tf.reduce_mean(tf.abs(output_layer - X))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.compat.v1.global_variables_initializer()

def next_batch(x_data, batch_size):
    rindx = np.random.choice(x_data.shape[0], batch_size, replace=False)
    x_batch = x_data[rindx, :]
    return x_batch

# Training
num_steps = 10
batch_size = 150
num_batches = len(scaled_data) // batch_size   # // : floor division

with tf.compat.v1.Session() as sess:
    sess.run(init)
    for step in range(num_steps):
        for iteration in range(num_batches):
            X_batch = next_batch(scaled_data, batch_size)
            sess.run(train, feed_dict={X: X_batch})

        if step % 1 == 0:
            err = loss.eval(feed_dict = {X: scaled_data})
            print(step, '\tLoss:', err)
            output_2d = hidden_layer.eval(feed_dict= {X: scaled_data})

    
    output_2d_test = hidden_layer.eval(feed_dict = {X: scaled_test_data})

# Visualization of results

import matplotlib.pyplot as plt

plt.figure(figsize = (20, 8))
plt.scatter(output_2d[:, 0], output_2d[:, 1], c= train_Set['Class'], alpha=0.7)
