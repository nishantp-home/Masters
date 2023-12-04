import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# This one is not working # Debug it !!

def next_batch(x_data, batch_size):
    rindx = np.random.choice(x_data.shape[0], batch_size, replace=False)
    x_batch = x_data[rindx, :]
    return x_batch

#from keras.datasets import mnist

# Import MNIST data
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256
num_batches = len(X_train) // batch_size
display_step = 1000
examples_to__show = 10

# Network parameters 
num_hidden_1 = 256 # 1st layer feature count
num_hidden_2 = 128 # 2nd layer feature count
num_input = 784 # Input layer data input (MNIST image data, size 28*28 pixel)

# tf Graph input (only pictures)
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder("float", [None, num_input])

weights = {
    "encoder_h1": tf.Variable(tf.compat.v1.random_normal([num_input, num_hidden_1])),
    "encoder_h2": tf.Variable(tf.compat.v1.random_normal([num_hidden_1, num_hidden_2])),
    "decoder_h1": tf.Variable(tf.compat.v1.random_normal([num_hidden_2, num_hidden_1])),
    "decoder_h2": tf.Variable(tf.compat.v1.random_normal([num_hidden_1, num_input])),
}

biases = {
    "encoder_b1": tf.Variable(tf.compat.v1.random_normal([num_hidden_1])),
    "encoder_b2": tf.Variable(tf.compat.v1.random_normal([num_hidden_2])),
    "decoder_b1": tf.Variable(tf.compat.v1.random_normal([num_hidden_1])),
    "decoder_b2": tf.Variable(tf.compat.v1.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))

    #Encoder hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

# Building a decoder
def decoder(x):
    # Decoder hidden layer with signoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['decoder_h1']), biases['decoder_b1']))

    # Decoder hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    return layer_2

# Construct a model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

# Train autoencoder
sess = tf.compat.v1.Session()

# Run the initializer
sess.run()

# Training
for i in range(1, num_steps + 1):
    # Prepare data
    # Get the next batch of MNIST data (only images are needed, not labels)
    for iter in range(num_batches):
        batch_x = next_batch(X_train, batch_size)

    # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, 1))








