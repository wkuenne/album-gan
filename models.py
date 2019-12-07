import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

def make_generator(num_channels, z_dim):
    """
    The model for the generator network is defined here. 
    """
    # instantiate model
    model = Sequential()

    return model


def make_discriminator(num_channels):
    """
    The model for the discriminator network is defined here. 
    """
    # instantiate model
    model = Sequential()

    return model