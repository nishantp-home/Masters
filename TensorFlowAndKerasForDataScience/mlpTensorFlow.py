import numpy as np
import tensorflow as tf

from keras.datasets import mnist

def next_batch(x_data, y_data, index_in_epoch, batch_size):
    start_index = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > trainSampleCount:
        # epochs_completed += 1
        # shuffle the data
        perm = np.arange(trainSampleCount)
        np.random.shuffle(perm)
        x_data = x_data[perm, :]
        y_data = y_data[perm]
        # Start next epoch
        start_index = 0
        index_in_epoch = batch_size
        # assert batch_size <= trainSampleCount
    end_index = index_in_epoch
    updated_index_in_epoch = index_in_epoch

    return x_data[start_index:end_index, :], y_data[start_index:end_index], updated_index_in_epoch



(X_train, y_train), (X_test, y_test) = mnist.load_data()
trainSampleCount = len(X_train)
testSampleCount = len(X_test)
sampleVectorLength = np.prod(X_train.shape[1:])

# Reshape data sets.  Sample 28*28 matrix -> 784 1D row vectorarray 
X_train, X_test = X_train.reshape([trainSampleCount, sampleVectorLength]), X_test.reshape([testSampleCount, sampleVectorLength])

# Convert to float values for scaling 
X_train, X_test = np.array(X_train, dtype=np.float32), np.array(X_test, dtype=np.float32)




# Training parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

# Network parameters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = sampleVectorLength  #784
n_classes = 10

y_train_matrix, y_test_matrix = np.zeros([len(y_train), n_classes]), np.zeros([len(y_test), n_classes])

for i in range(len(y_train)):
    y_train_matrix[i][y_train[i]] = 1
for i in range(len(y_test)):
    y_test_matrix[i][y_test[i]] = 1


# tf Graph input
tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder('float', [batch_size, n_input])
Y = tf.compat.v1.placeholder('int16', [batch_size, n_classes])  # response

# Store layers weight and bias
weights = {
    'h1': tf.Variable(tf.compat.v1.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.compat.v1.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.compat.v1.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.compat.v1.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.compat.v1.random_normal([n_classes]))
}

# Define Multilevel Perceptron (MLP)
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer

# Construct model
logits = multilayer_perceptron(X)  # feed the predictors

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing variables
init = tf.compat.v1.global_variables_initializer()

# Invoke the session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(trainSampleCount / batch_size)

        index_in_epoch = 0
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y, updated_index_in_epoch = next_batch(X_train, y_train_matrix, index_in_epoch, 
                                          batch_size)
            index_in_epoch = updated_index_in_epoch

        # Run optimization op (backprop) and cost op (to get the loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch


        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", "%02d" % (epoch + 1), "cost={:.3f}".format(avg_cost))
    
    print("Optimization finished !")



