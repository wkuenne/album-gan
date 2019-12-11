import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate, Embedding, Layer, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
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
	def __init__(self, num_channels, n_genres, image_side_len, batch_size=1):
		"""
		The model for the generator network is defined here. 
		"""
		super(Generator_Model, self).__init__()

		self.smallest_img_side_len = (image_side_len // 8)
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size)
		self.embed_dense = Dense(self.smallest_img_side_len**2, activation='softmax')
		self.embed_reshape = Reshape((-1, self.smallest_img_side_len, self.smallest_img_side_len, 1))

		self.initial_dense = Dense(image_side_len * self.smallest_img_side_len**2, activation='softmax')
		self.initial_reshape = Reshape((-1, self.smallest_img_side_len, self.smallest_img_side_len, num_channels))

		self.deconv1 = Conv2DTranspose(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv2 = Conv2DTranspose(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv3 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv4 = Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)

		self.noise = ScaledGaussianNoise(stddev=1)
		self.adain = ADAin()
		self.activation = LeakyReLU(0.2)
		self.merge = Concatenate()
		

	@tf.function
	def call(self, adain_net, w, genres, noise_scale=1):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		inputs = tf.convert_to_tensor(inputs)
		genres = tf.convert_to_tensor(genres)

		(batch_size, z_dim) = w.shape
		(scale, bias) = self.get_adain_params(adain_net, w)

		# in lieu of input, use a random constant as base
		rand_const = tf.random.normal((batch_size, self.smallest_img_side_len, self.smallest_img_side_len, z_dim))

		# get embedding of genre label
		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))
		out = self.conv1(inputs)

		# merge embedding information and image information
		combined = self.merge(rand_const, out)
		
		# proceed with StyleGAN
		adain1 = self.adain(self.noise(combined), scale, bias)
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
		batch_size = w.shape[0]
		scale = tf.slice(adain_params, [0, 0, 0], (batch_size, 1, -1))
		bias = tf.slice(adain_params, [0, 1, 0], (batch_size, 1, -1))
		return scale, bias


class Discriminator_Model(tf.keras.Model):
	def __init__(self, num_channels, n_genres, image_side_len):
		super(Discriminator_Model, self).__init__()
		"""
		The model for the discriminator network is defined here. 
		"""
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size)
		self.embed_dense = Dense(image_side_len**2, activation='softmax')
		self.embed_reshape = Reshape((-1, image_side_len, image_side_len, 1))

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
		self.merge = Concatenate()

	@tf.function
	def call(self, inputs, genres):
		"""
		Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

		:param inputs: a batch of images, shape=[batch_size, height, width, channels]
		:param genre: a genre string, shape=[1,]

		:return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
		"""
		inputs = tf.convert_to_tensor(inputs)
		genres = tf.convert_to_tensor(genres)

		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))
		out = self.conv1(inputs)

		out = self.merge([embed, out])
		
		# proceed with normal 
		out = self.conv2(self.activate(self.norm(out)))
		out = self.conv3(self.activate(self.norm(out)))
		out = self.conv4(self.activate(self.norm(out)))

		flat = self.flat(out)
		return self.decision(flat)

