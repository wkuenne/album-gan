import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, Layer, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from tensorflow.nn import batch_normalization

def make_noise_scale_net(num_channels):
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
	model.add(Reshape((2, num_channels)))

	return model

def make_mapping_net(mapping_dim, z_dim):
	"""
	Uses a random vector from latent space to learn styles of genres, outputs embeddings.
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

	def call(self, x, y_s, y_b):
		return self.ADAin_calc(x, y_s, y_b)

	def ADAin_calc(self, x, y_s, y_b):
		"""
		Performs the ADAin calculation.
		y_s = scale vector, shape (1, n)
		y_b = bias vector, shape (1, n)
		x = convolutional output, shape (batch_size, width, height, out_channels\)
		"""
		(mean, variance) = tf.nn.moments(x, (1, 2), keepdims=True)
		print(y_s)
		print(x)
		print(tf.matmul(batch_normalization(x=x, mean=mean, variance=variance, offset=0, scale=1, variance_epsilon=0.00001)), )
		return batch_normalization(x=x, mean=mean, variance=variance, offset=y_b, scale=y_s, variance_epsilon=0.00001)

class ScaledGaussianNoise(Layer):
	def __init__(self, stddev=1):
		super(ScaledGaussianNoise, self).__init__()
		self.stddev = stddev

	# def build(self):

	def call(self, x, scale=1):
		return x + scale * tf.random.normal(x.shape, stddev=self.stddev)

class Generator_Model(Model):
	def __init__(self, num_channels):
		"""
		The model for the generator network is defined here. 
		"""
		super(Generator_Model, self).__init__()

		self.deconv1 = Conv2DTranspose(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv2 = Conv2DTranspose(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv3 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv4 = Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)

		self.noise = ScaledGaussianNoise(stddev=1)
		self.adain = ADAin()
		self.activation = LeakyReLU(0.2)
		

	@tf.function
	def call(self, adain_net, w, noise_scale=1):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		(batch_size, z_dim) = w.shape
		(scale, bias) = self.get_adain_params(adain_net, w)
		rand_const = tf.random.normal((batch_size, 4, 4, z_dim))
		
		adain1 = self.adain(self.noise(rand_const), scale, bias)
		synthesized1 = self.deconv1(adain1)
		adain2 = self.adain(self.noise(synthesized1), scale, bias)
		activated1 = self.activation(adain2)

		adain3 = self.adain(self.noise(activated1), scale, bias)
		synthesized2 = self.deconv2(adain3)
		adain4 = self.adain(self.noise(synthesized2), scale, bias)
		activated2 = self.activation(adain4)

		adain5 = self.adain(self.noise(activated2), scale, bias)
		synthesized3 = self.deconv2(adain5)
		adain6 = self.adain(self.noise(synthesized3), scale, bias)

		return adain6

	def get_adain_params(self, model, w):
		adain_params = model(w)
		print(adain_params)
		batch_size = w.shape[0]
		scale = tf.slice(adain_params, [0, 0, 0], (batch_size, 1, -1))
		bias = tf.slice(adain_params, [0, 1, 0], (batch_size, 1, -1))
		return scale, bias


class Discriminator_Model(tf.keras.Model):
	def __init__(self, num_channels, n_genres, img_shape):
		super(Discriminator_Model, self).__init__()
		"""
		The model for the discriminator network is defined here. 
		"""
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size)
		self.embed_dense = Dense(img_shape, activation='softmax')

		# conv layers
		self.conv1 = Conv2D(num_channels // 8, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv2 = Conv2D(num_channels // 4, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv3 = Conv2D(num_channels // 2, (5, 5), strides=(2, 2), padding='same', use_bias=False)
		self.conv4 = Conv2D(num_channels, (5, 5), strides=(2, 2), padding='same', use_bias=False)

		# condense into a decision
		self.decision = Dense(1, activation='sigmoid')

		self.norm = BatchNormalization()
		self.activate = LeakyReLU(0.2)
		self.flat = Flatten()

	@tf.function
	def call(self, inputs, genre):
		"""
		Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

		:param inputs: a batch of images, shape=[batch_size, height, width, channels]
		:param genre: a genre string, shape=[1,]

		:return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
		"""
		embed = self.embedding(genre)
		embed = tf.reshape(self.embed_dense(embed), inputs.shape[1:])
		out = self.conv1(inputs)

		out = tf.concat([embed, out], [-1])
		
		out = self.conv2(self.activate(self.norm(out)))
		out = self.conv3(self.activate(self.norm(out)))
		out = self.conv4(self.activate(self.norm(out)))

		flat = self.flat(out)
		return self.decision(flat)



def make_generator(num_channels):
	"""
	The model for the generator network is defined here. 
	"""
	# # instantiate model
	# model = Sequential()

	# # starts with shape (z_dim * 4 * 4) random input
	# # add noise
	# model.add(GaussianNoise(stddev=1), input_size=(4, 4, z_dim))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())
	# # do Conv 3x3
	# model.add(Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())

	# # do upsampling
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())
	# # do Conv 3x3
	# model.add(Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# # add noise
	# model.add(GaussianNoise(stddev=1))
	# # do ADAin with vector in W space (from mapping net)
	# model.add(ADAin())

	# # continue...

	# # return model
	return Generator_Model(num_channels)


def make_discriminator(num_channels, n_genres, img_shape):
	"""
	The model for the discriminator network is defined here. 
	"""
	# # instantiate model
	# model = Sequential()

	# # conv layers
	# model.add(Conv2D(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False))

	# model.add(Conv2D(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# model.add(Conv2D(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# model.add(Conv2D(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False))
	# model.add(BatchNormalization())
	# model.add(LeakyReLU(0.2))

	# # condense into a decision
	# model.add(Flatten())
	# model.add(Dense(1, activation='sigmoid'))

	return Discriminator_Model(num_channels, n_genres, img_shape)


# class Affine_Transform(Model):
# 	def __init__(self, num_channels):
# 		"""
# 		The model that learns scale and bias for ADAin is defined here. 
# 		"""
# 		super(Affine_Transform, self).__init__()
		
# 		upspace = num_channels * 4
# 		self.dense1 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense2 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense3 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense4 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense5 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense6 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense7 = Dense(upspace, use_bias=False, activation='softmax')
# 		self.dense8 = Dense(2 * num_channels, use_bias=False, activation='softmax'))=
# 		model.add(Reshape((2, num_channels)))
		

# 	@tf.function
# 	def call(self, adain_net, w, noise_scale=1):
# 		"""
# 		Executes the generator model on the random noise vectors.

# 		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

# 		:return: prescaled generated images, shape=[batch_size, height, width, channel]
# 		"""
# 		(batch_size, z_dim) = w.shape
# 		(scale, bias) = self.get_adain_params(adain_net, w)
# 		rand_const = tf.random.normal((batch_size, 4, 4, z_dim))
		
# 		adain1 = self.adain(self.noise(rand_const), scale, bias)
# 		synthesized1 = self.deconv1(adain1)
# 		adain2 = self.adain(self.noise(synthesized1), scale, bias)
# 		activated1 = self.activation(adain2)

# 		adain3 = self.adain(self.noise(activated1), scale, bias)
# 		synthesized2 = self.deconv2(adain3)
# 		adain4 = self.adain(self.noise(synthesized2), scale, bias)
# 		activated2 = self.activation(adain4)

# 		adain5 = self.adain(self.noise(activated2), scale, bias)
# 		synthesized3 = self.deconv2(adain5)
# 		adain6 = self.adain(self.noise(synthesized3), scale, bias)

# 		return adain6