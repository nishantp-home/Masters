import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist data



learning_rate = 0.001
# Input and target placeholders
inputs_ = tf.compat.v1.placeholder(tf.float32, (None, 28, 28, 1), name='input')
targets_ = tf.compat.v1.placeholder(tf.float32, (None, 28, 28, 1), name='target')


# Encoder
conv1 = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3), padding='same', activation=tf.nn.relu)(inputs_)
# now 29*28*16
maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv1)
# now 14 * 14 * 16
conv2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)(maxpool1)
# now 14 * 14 * 8
maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv2)
# Now 7 *7 * 8
conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='same', activation=tf.nn.relu)(maxpool2)
# Now 7 *7 * 8
encoded = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv3)

# Decoder
upsample1 = tf.compat.v1.image.resize_images(encoded, size=(7,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# resize images
# now 7 * 7 * 8
conv4 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(upsample1)
# now 7 * 7 * 8
upsample2 = tf.compat.v1.image.resize_images(conv4, size=(14,14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# Now 14 * 14 * 8
conv5 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(upsample2)
# now 14 * 14 * 8
upsample3 = tf.compat.v1.image.resize_images(conv5, size=(28,28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# now 28 * 28 * 8
conv6 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(upsample3)

logits = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation=None)(conv6)


# get reconstructed image

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)

# get cost and define the optimizer
cost = tf.reduce_mean(loss)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Training
sess = tf.compat.v1.Session()
epochs = 20
batch_size = 