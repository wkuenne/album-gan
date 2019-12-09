import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tf.nn import batch_normalization

def make_noise_scale_net(num_channels, noise_shape):
    """
    Analyzes a noise matrix to determine a scaling for the noise (scalar value)
    """
    model = Sequential()

    initial_shape = 512

    model.add(Dense(initial_shape, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 2, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 4, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 8, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 16, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 32, use_bias=True, activation='softmax'))
    model.add(Dense(initial_shape // 64, use_bias=True, activation='softmax'))
    model.add(Dense(1, use_bias=True))

    return model

def make_affine_transform_net(num_channels, z_dim):
    """
    Given a style vector of shape (z_dim), learns scale and bias channels for ADAin
    """
    model = Sequential()

    upspace = num_channels * 4

    model.add(Dense(upspace, use_bias=True, input_shape=(z_dim,), activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(upspace, use_bias=True, activation='softmax'))
    model.add(Dense(2 * num_channels, use_bias=True))
    model.add(Reshape(2, num_channels))

    return model

def make_mapping_net(mapping_dim, z_dim):
    """
    Uses a random vector from latent space to learn styles of photos and map features.
    Its output is fed into the affine transform layer.
    """
    # instantiate model
    model = Sequential()

    # 8 layer MLP (f)
    model.add(Dense(mapping_dim, use_bias=True, input_shape=(z_dim,), activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(mapping_dim, use_bias=True, activation='softmax'))
    model.add(Dense(z_dim, use_bias=True))

    return model

class ADAin(Layer):
    def __init__(self):
        super(ADAin, self).__init__()

    # def build(self):

    def call(self, y_s, y_b, x):
        return ADAin(y_s, y_b, x)

    def ADAin(y_s, y_b, x):
        """
        Performs the ADAin calculation.
        y_s = scale vector, shape (1, n)
        y_b = bias vector, shape (1, n)
        x = convolutional output, shape (batch_size, width, height, out_channels\)
        """
        (mean, variance) = tf.nn.moments(x, (1, 2), keep_dims=True)
        return batch_normalization(x=x, mean=mean, variance=variance, offset=y_b, scale=y_s, variance_epsilon=0.00001)

class ScaledGaussianNoise(Layer):
    def __init__(self, noise_shape):
        super(ADAin, self).__init__()
        self.noise_shape = noise_shape

    # def build(self):

    def call(self, a):
        return a * tf.random.normal(self.noise_shape)

def make_generator(num_channels):
    """
    The model for the generator network is defined here. 
    """
    # instantiate model
    model = Sequential()

    # starts with shape (z_dim * 4 * 4) random input
    # add noise
    model.add(GaussianNoise(stddev=1), input_size=(4, 4, z_dim))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())
    # do Conv 3x3
    model.add(Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())

    # do upsampling
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())
    # do Conv 3x3
    model.add(Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    # add noise
    model.add(GaussianNoise(stddev=1))
    # do ADAin with vector in W space (from mapping net)
    model.add(ADAin())

    # continue...

    return model


def make_discriminator(num_channels):
    """
    The model for the discriminator network is defined here. 
    """
    # instantiate model
    model = Sequential()

    # conv layers
    model.add(Conv2D(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))

    model.add(Conv2D(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # condense into a decision
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model