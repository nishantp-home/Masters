from keras.layers import Input, Dense
from keras.models import Model

# This is the size of our encoded representations
# hidden layer
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming 784 floats as input

# Input placeholder
input_img = Input(shape=(784,))

# 'encoded' is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# 'decoded' is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Simple encoder model
encoder = Model(input_img, encoded)

# Simple decoder model
# Create a placeholder for an encoded (32- dimensional) input
encoded_input = Input(shape=(encoding_dim, ))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# Read in data
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print('Training data set shape:', x_train.shape)
print('Testing data set shape:', x_test.shape)

# Fit an autoencoder model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
# note that we take them from test dataset
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# plot images

import matplotlib.pyplot as plt

n = 10   # number of digits to display
plt.figure(figsize=(20,4))
for i in range(n):
    # displace original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


