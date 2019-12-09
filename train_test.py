import tensorflow as tf
from tensorflow.random import uniform
from tensorflow import GradientTape
import numpy as np

from fid import fid_function
from loss_functions import generator_loss, discriminator_loss

import numpy as np

from imageio import imwrite

from get_args import get_args
args = get_args()

from tensorflow.keras.optimizers import Adam
generator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
discriminator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
map_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
adain_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)

# Train the model for one epoch.
def train(
		generator, 
		discriminator, 
		dataset_iterator, 
		manager, 
		mapping_net,
		noise_net,
		adain_net,
		num_gen_updates=1,
		num_channels=512
	):
	"""
	Train the model for one epoch. Save a checkpoint every 500 or so batches.

	:param generator: generator model
	:param discriminator: discriminator model
	:param dataset_ierator: iterator over dataset, see preprocess.py for more information
	:param manager: the manager that handles saving checkpoints by calling save()

	:return: The average FID score over the epoch
	"""
	z_dim = args.z_dim
	batch_size = args.batch_size
	sum_fid = 0
	# Loop over our data until we run out
	for iteration, batch in enumerate(dataset_iterator):
		z = uniform((batch_size, z_dim), minval=-1, maxval=1)

		# with GradientTape() as gen_tape, GradientTape() as disc_tape:
		with GradientTape() as tape:
			w = mapping_net(z)
			assert(w.shape == (batch_size, z_dim))

			adain_params = adain_net(w)
			scale = tf.slice(adain_params, [0, 0, 0], (adain_params.shape[0], 1, -1))
			bias = tf.slice(adain_params, [0, 1, 0], (adain_params.shape[0], 1, -1))
			assert(scale.shape == (batch_size, 1, num_channels))
			assert(scale.shape == (batch_size, 1, num_channels))

			# noise_scale = noise_net(uniform((batch_size, 4, 4, z_dim), minval=-1, maxval=1))

			# generated images
			G_sample = generator(scale, bias, z_dim)

			# test discriminator against real images
			logits_real = discriminator(batch, training=True)
			# re-use discriminator weights on new inputs
			logits_fake = discriminator(G_sample, training=True)

			g_loss = generator_loss(logits_fake)
			d_loss = discriminator_loss(logits_real, logits_fake)

		map_grads = tape.gradient(g_loss, mapping_net.trainable_variables) # success measured by same parameters
		map_optimizer.apply_gradients(zip(map_grads, mapping_net.trainable_variables))

		a_grads = tape.gradient(g_loss, adain_net.trainable_variables) # success measured by same parameters
		adain_optimizer.apply_gradients(zip(a_grads, adain_net.trainable_variables))
			
		# optimize the generator and the discriminator
		gen_gradients = tape.gradient(g_loss, generator.trainable_variables)
		generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

		if (iteration % num_gen_updates == 0):
			disc_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
			discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

		# Save
		if iteration % args.save_every == 0:
			manager.save()

		# Calculate inception distance and track the fid in order
		# to return the average
		if iteration % 500 == 0:
			fid_ = fid_function(batch, G_sample)
			print('**** D_LOSS: %g ****' % d_loss)
			print('**** G_LOSS: %g ****' % g_loss)
			print('**** INCEPTION DISTANCE: %g ****' % fid_)
			sum_fid += fid_
	return sum_fid / (iteration // 500)


# Test the model by generating some samples.
def test(generator, batch_size, z_dim, out_dir):
	"""
	Test the model.

	:param generator: generator model

	:return: None
	"""
	img = np.array(generator(uniform(batch_size, z_dim), minval=-1, maxval=1), training=False)

	### Below, we've already provided code to save these generated images to files on disk
	# Rescale the image from (-1, 1) to (0, 255)
	img = ((img / 2) - 0.5) * 255
	# Convert to uint8
	img = img.astype(np.uint8)
	# Save images to disk
	for i in range(0, batch_size):
		img_i = img[i]
		s = out_dir+'/'+str(i)+'.png'
		imwrite(s, img_i)
