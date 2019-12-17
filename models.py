import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Concatenate, Embedding, Layer, GaussianNoise, Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

from get_args import get_args
args = get_args()
z_dim = args.z_dim
batch_size = args.batch_size
num_channels = args.num_channels
num_genres = args.num_genres
num_gen_updates = args.num_gen_updates
out_dir = args.out_dir
mapping_dim = args.mapping_dim
image_side_len = args.image_side_len
n_genres = args.num_genres

num_convolutions = 4

class Generator_Model(Model):
	def __init__(self):
		"""
		The model for the generator network is defined here. 
		"""
		super(Generator_Model, self).__init__()

		# side length image is reduced to by discriminator
		self.smallest_img_side_len = (image_side_len // (num_convolutions**2))

		# cGAN embeddings of genres
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size)
		self.embed_dense = Dense(self.smallest_img_side_len ** 2, activation=tf.nn.leaky_relu)
		self.embed_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, 1))

		# cGAN prepping for latent input
		self.initial_dense = Dense(num_channels * self.smallest_img_side_len ** 2, activation=tf.nn.leaky_relu)
		self.initial_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, num_channels))

		# GAN layers
		self.deconv1 = Conv2DTranspose(num_channels // 2, kernel_size=7, strides=1, padding='same')
		self.deconv2 = Conv2DTranspose(num_channels // 4, kernel_size=5, strides=2, padding='same')
		self.finaldeconv = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')

		# various layers
		self.merge = Concatenate()

	@tf.function
	def call(self, z, genres):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		genres = tf.convert_to_tensor(genres)

		# get embedding of genre label
		embedding = self.embedding(genres)
		embedding_reshaped = self.embed_reshape(self.embed_dense(embedding))

		# dense for z
		z_processed = self.initial_reshape(self.initial_dense(z))

		# merge embedding information and image information
		combined = self.merge([embedding_reshaped, z_processed])

		# proceed with GAN
		out1 = self.deconvolve(self.deconv1, combined)
		out2 = self.deconvolve(self.deconv2, out1)

		return self.finaldeconv(out3)

	def deconvolve(self, layer, x):
		out = layer(x)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = tf.nn.batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		return tf.nn.leaky_relu(out)

class Discriminator_Model(tf.keras.Model):
	def __init__(self):
		super(Discriminator_Model, self).__init__()
		"""
		The model for the discriminator network is defined here. 
		"""
		# cGAN embedding of labels
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size, name='DiscEmbedding')
		self.embed_dense = Dense(image_side_len**2, activation=tf.nn.leaky_relu)
		self.embed_reshape = Reshape((image_side_len, image_side_len, 1))

		# GAN conv layers
		self.conv1 = Conv2D(num_channels // 4, kernel_size=5, strides=2, padding='same')
		self.conv2 = Conv2D(num_channels // 2, kernel_size=5, strides=2, padding='same')
		self.finalconv = Conv2D(num_channels, kernel_size=7, strides=1, padding='same')

		# condense into a decision
		self.decision = Dense(1, activation='sigmoid')

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
		genres = tf.convert_to_tensor(genres)

		# set up genre embedding
		embedding = self.embedding(genres)
		embedding_reshaped = self.embed_reshape(self.embed_dense(embedding))

		# merge information
		merged = self.merge([inputs, embedding_reshaped])

		# proceed with normal discriminator processes
		out1 = tf.nn.leaky_relu(self.conv1(merged))
		out2 = self.convolve(out1, self.conv2)
		out3 = self.convolve(out2, self.finalconv)
		flat = self.flat(out3)

		return self.decision(flat)

	def convolve(self, x, layer):
		"""
		Factors out convolution. Applies convolution layer, batch normalizes, and leaky ReLU activates
		"""
		out = layer(x)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = tf.nn.batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		return tf.nn.leaky_relu(out)
