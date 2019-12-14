

def make_noise_scale_net():
	"""
	Analyzes a noise matrix to determine a scaling for the noise (scalar value)
	"""
	model = Sequential()

	initial_shape = 512

	model.add(Dense(initial_shape, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 2, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 4, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 8, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 16, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 32, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(initial_shape // 64, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(1, use_bias=True))

	return model

def make_affine_transform_net():
	"""
	Given a style vector of shape (z_dim), learns scale and bias channels for ADAin
	"""
	model = Sequential()

	upspace = num_channels * 4

	model.add(Dense(upspace, use_bias=True, input_shape=(z_dim,), activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	model.add(Dense(upspace, use_bias=True, activation=tf.nn.leaky_relu))
	# model.add(Dense(2 * num_channels, use_bias=True))
	# model.add(Reshape((2, num_channels)))
	model.add(Dense(4 * num_channels, use_bias=True))
	model.add(Reshape((2, 2 * num_channels)))
	model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model
	# return ADAin_Model()

def make_mapping_net():
	"""
	Uses a random vector from latent space to learn styles of genres, outputs embeddings.
	Its output is fed into the affine transform layer.
	"""
	# instantiate model
	model = Sequential()

	# 8 layer MLP (f)
	model.add(Dense(mapping_dim, use_bias=True, input_shape=(z_dim,), activation=tf.nn.leaky_relu, name='Mapping1'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping2'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping3'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping4'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping5'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping6'))
	model.add(Dense(mapping_dim, use_bias=True, activation=tf.nn.leaky_relu, name='Mapping7'))
	model.add(Dense(z_dim, use_bias=True, name='Mapping8'))

	return model