import matplotlib.pyplot as plt
from IPython.display import Image, SVG

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, InputLayer
from keras import regularizers

# Load the training and test data sets (ignoring class labels)
(x_train, _), (x_test, _) = mnist.load_data()

# Scale the training and test data to range between 0 and 1
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# Convert 28*28 images to vectors of length 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Reshape images back to 28 * 28 * 1 for the CNN model
# 1 is for channel, as we have black and white images
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Autoencoder model
autoencoder = Sequential()

# encoder layers
# Use Conv2D and MaxPooling2S layers for the encoder
autoencoder.add(InputLayer(input_shape=(x_train.shape[1:])))
autoencoder.add(Conv2D(16, (3,3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3,3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D((2, 2), padding='same'))
autoencoder.add(Conv2D(8, (3,3), strides=(2, 2), activation='relu', padding='same'))

# Flatten encoding for visualization
autoencoder.add(Flatten())
autoencoder.add(Reshape((4, 4, 8)))

# Decoder model
# Conv2D and UpSampling2D layers for the decoder
autoencoder.add(Conv2D(8, (3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(8, (3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(16, (3,3), activation='relu'))
autoencoder.add(UpSampling2D((2, 2)))
autoencoder.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))

autoencoder.summary()
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten').output)
encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=100,batch_size=128,
                validation_data=(x_test, x_test))

