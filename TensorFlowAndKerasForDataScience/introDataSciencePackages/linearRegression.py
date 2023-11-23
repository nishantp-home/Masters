import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets as skds
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Create dummy data
X, y = skds.make_regression(n_samples=200, n_features=1, n_informative=1, n_targets=1, noise=20.0)  # 200 samples, each for the targer (yi) and predictor (Xi)

# Reshape numpy array to have 2 dimensions
if (y.ndim == 1):
    y = y.reshape(len(y), 1)

# Plotting
plt.figure(figsize=(7, 4))
plt.plot(X, y, '.b')
plt.show()


# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)  # set aside 30% dataset for testing

# Defining inputs, parameters and other variables
num_outputs = y_train.shape[1]    # One response variable
num_inputs = X_train.shape[1]     # One predictor variable

# Define eqn: y = W*x + b
# Use placeholders
tf.compat.v1.disable_eager_execution()   # tf.compat.v1 is the compatible api version with the specific methods

x_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_inputs], name='x')
y_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_outputs], name='y')
w = tf.Variable(tf.zeros([num_inputs, num_outputs]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros([num_outputs]), dtype=tf.float32, name='b')

model = tf.matmul(x_tensor, w) + b

# Defining the loss funtion
# Mean squared error/ residuals (MSE)
# residual = (Y)predict - (Y)actual; residual = model - ytensor
loss = tf.reduce_mean(tf.square(model - y_tensor))   # Mean squared arror 

# Compute MSE and R2
mse = loss
y_mean = tf.reduce_mean(y_tensor)
total_error = tf.reduce_sum(tf.square(y_tensor - y_mean))
unexplained_error = tf.reduce_sum(tf.square(y_tensor-model))
rsq = 1 - tf.divide(unexplained_error, total_error)   # R2 = 1 - unexplained error / total error, signifies the goodness of fit

 
# Define optimizer function
learning_rate = 0.001
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss)
# Gradient descent is an algorithm that minimizes functions (loss funtion in this example)
# learning rate is  the step size we take per iteration towards the solution

# Train the model
num_epochs = 1800   # number of iterations to run the training

# w_hat and b_hat: estimates of w and b
w_hat = 0
b_hat = 0

loss_epochs = np.empty(shape=[num_epochs], dtype=float)
mse_epochs = np.empty(shape=[num_epochs], dtype=float)
rs_epochs = np.empty(shape=[num_epochs], dtype=float)

#initial values
mse_score = 0
rsq_score = 0

with tf.compat.v1.Session() as tfs:
    tfs.run(tf.compat.v1.global_variables_initializer())  # run optimizer / loop on training data
    for epoch in range(num_epochs):
        feed_dict = {x_tensor: X_train, y_tensor: y_train}
        loss_val, _ = tfs.run([loss, optimizer], feed_dict=feed_dict)
        loss_epochs[epoch] = loss_val   #calculate and store error
        feed_dict = {x_tensor: X_test, y_tensor: y_test}
        mse_score, rsq_score = tfs.run([mse, rsq], feed_dict=feed_dict)
        mse_epochs[epoch] = mse_score
        rs_epochs[epoch] = rsq_score

    w_hat, b_hat = tfs.run([w, b])  # final values of w and b obtained after all iterations
    w_hat = w_hat.reshape(1)

print('model: Y = {0:.8f} X + {1:.8f}'.format(w_hat[0], b_hat[0]))
print('For test data: MSE = {0:.8f}, R2 = {1:.8f}'.format(mse_score, rsq_score))

# Visulatization of resulting linear regression fit
plt.figure(figsize=(14, 8))
plt.title('original data and Trained model')

x_plot = [np.min(X) - 1,  np.max(X) + 1] #Range of X values
y_plot = w_hat * x_plot + b_hat # w_hat and b_hat predicited before
plt.axis([x_plot[0], x_plot[1], y_plot[0], y_plot[1]])
plt.plot(X, y, '.b', label='Original Data')
plt.plot(x_plot, y_plot, 'r-', label='Trained Model')
plt.legend()
plt.show()

# plot loss and mse with epochs
plt.figure(figsize=(14, 8))
plt.axis([0, num_epochs, 0, np.max(loss_epochs)])
plt.title('Loss in Iterations')
plt.xlabel('# Epoch')
plt.ylabel('# MSE')

plt.axis([0, num_epochs, 0, np.max(mse_epochs)])
plt.plot(mse_epochs, label='MSE on X_test')
plt.xlabel('# Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()