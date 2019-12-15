import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Reshape, Concatenate, Embedding, Layer, GaussianNoise, Dense, Conv2D, BatchNormalization, Conv2DTranspose

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

		# learned constant in StyleGAN
		self.learned_const = tf.Variable(tf.random.normal((batch_size, self.smallest_img_side_len, self.smallest_img_side_len, num_channels)))

		# cGAN embeddings of genres
		self.embedding_size = 50
		self.embedding = Embedding(n_genres, self.embedding_size, name="generator")
		self.embed_dense = Dense(self.smallest_img_side_len ** 2, activation=tf.nn.leaky_relu, name="embedding_dense")
		self.embed_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, 1))

		# cGAN prepping for random constant
		self.initial_dense = Dense(num_channels, activation=tf.nn.leaky_relu)
		self.initial_reshape = Reshape((self.smallest_img_side_len, self.smallest_img_side_len, num_channels))

		# StyleGAN layers
		self.deconv1 = Conv2DTranspose(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv2 = Conv2DTranspose(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.deconv3 = Conv2DTranspose(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.finaldeconv = Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh')

		# various layers
		self.noise = ScaledGaussianNoise(stddev=1)
		self.adain = ADAin()
		self.merge = Concatenate()

	@tf.function
	def call(self, adain_net, w, genres, noise_scale=1):
		"""
		Executes the generator model on the random noise vectors.

		:param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

		:return: prescaled generated images, shape=[batch_size, height, width, channel]
		"""
		genres = tf.convert_to_tensor(genres)

		# get embedding of genre label
		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))
		out = self.initial_reshape(self.initial_dense(self.learned_const))

		# merge embedding information and image information
		combined = self.merge([embed, out])

		# StyleGAN
		# combined = self.learned_const
		out1 = self.deconvolve(self.deconv1, combined, adain_net, w)
		out2 = self.deconvolve(self.deconv2, out1, adain_net, w)
		out3 = self.deconvolve(self.deconv3, out2, adain_net, w)
		return self.finaldeconv(out3)

	def deconvolve(self, layer, x, a, w):
		# get ADAin scale, bias vectors
		(scale, bias) = self.get_adain_params(a, w, x.shape[-1])
		# apply ADAin operation
		adain1 = self.adain(self.noise(x), scale, bias)
		# deconvolve
		synthesized = layer(adain1)
		# get ADAin scale, bias vectors
		(scale, bias) = self.get_adain_params(a, w, synthesized.shape[-1])
		# apply ADAin operation
		adain2 = self.adain(self.noise(synthesized), scale, bias)
		# activate
		return tf.nn.leaky_relu(adain2)


	def get_adain_params(self, a, w, n):
		adain_params = a(w)
		scale = tf.reshape(tf.slice(adain_params, [0, 0, 0], (batch_size, 1, n)), (batch_size, -1))
		bias = tf.reshape(tf.slice(adain_params, [0, 1, 0], (batch_size, 1, n)), (batch_size, -1))
		return scale, bias

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
		self.conv1 = Conv2D(num_channels // 8, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.conv2 = Conv2D(num_channels // 4, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.conv3 = Conv2D(num_channels // 2, (3, 3), strides=(2, 2), padding='same', use_bias=False)
		self.finalconv = Conv2D(num_channels, (3, 3), strides=(2, 2), padding='same', use_bias=False)

		# condense into a decision
		self.decision = Dense(1, activation='sigmoid')

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
		embed = self.embedding(genres)
		embed = self.embed_reshape(self.embed_dense(embed))

		# merge information
		out = self.merge([embed, inputs])

		out1 = self.conv1(out)
		out2 = self.convolve(out1, self.conv2)
		out3 = self.convolve(out2, self.conv3)
		out4 = self.convolve(out3, self.finalconv)

		return self.decision(tf.reshape(out4, (batch_size, -1)))

	def convolve(self, x, layer):
		"""
		Factors out convolution. Applies convolution layer, batch normalizes, and leaky ReLU activates
		"""
		out = layer(x)
		(mean, variance) = tf.nn.moments(out, (0, 1, 2))
		out = tf.nn.batch_normalization(out, mean, variance, offset=0, scale=1, variance_epsilon=0.00001)
		return tf.nn.leaky_relu(out)

class Mapping_Model(tf.keras.Model):
	def __init__(self):
		super(Mapping_Model, self).__init__()
		"""
		Uses a random vector from latent space to learn styles of genres, outputs embeddings.
		Its output is fed into the affine transform layer.
		"""
		self.dense1 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping1')
		self.dense2 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping2')
		self.dense3 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping3')
		self.dense4 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping4')
		self.dense5 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping5')
		self.dense6 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping6')
		self.dense7 = Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping7')
		self.dense8 = Dense(z_dim, use_bias=True, name='Mapping8')

	@tf.function
	def call(self, r):
		return self.dense8(self.dense7(self.dense6(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(r))))))))

class ADAin_Model(tf.keras.Model):
	def __init__(self):
		super(ADAin_Model, self).__init__()
		"""
		Uses a random vector from latent space to learn styles of genres, outputs embeddings.
		Its output is fed into the affine transform layer.
		"""
		self.final_size = num_channels * 2
		upspace = num_channels * 4
		self.dense1 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin1')
		self.dense2 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin2')
		self.dense3 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin3')
		self.dense4 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin4')
		self.dense5 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin5')
		self.dense6 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin6')
		self.dense7 = Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu, name='ADAin7')
		self.dense8 = Dense(self.final_size * 2, use_bias=True, name='ADAin8', activation='tanh')

	@tf.function
	def call(self, w):
		return tf.reshape(self.dense8(self.dense7(self.dense6(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(w)))))))), (batch_size, 2, self.final_size))

class ADAin(Layer):
	def __init__(self):
		super(ADAin, self).__init__()

	def call(self, x, y_s, y_b):
		return self.ADAin_calc(x, y_s, y_b)

	def expand(self, x, dims):
		for i in dims:
			x = tf.expand_dims(x, i)
		return x

	def ADAin_calc(self, x, y_s, y_b):
		"""
		Performs the ADAin calculation.
		y_s = scale vector, shape (batch_size, n)
		y_b = bias vector, shape (batch_size, n)
		x = convolutional output, shape (batch_size, width, height, out_channels)
		"""
		(mean, variance) = tf.nn.moments(x, (1, 2), keepdims=True)
		correction = 0.000001
		return ((x - mean)/tf.sqrt(variance + correction)) * self.expand(y_s, [1, 2]) + self.expand(y_b, [1, 2])

class ScaledGaussianNoise(Layer):
	def __init__(self, stddev=1):
		super(ScaledGaussianNoise, self).__init__()
		self.stddev = stddev

	def call(self, x, scale=1):
		return x + scale * tf.random.normal(x.shape, stddev=self.stddev)
