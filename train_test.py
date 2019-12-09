import tensorflow as tf
from tensorflow.random import uniform
from tensorflow import GradientTape
import numpy as np

from fid import fid_function
from loss_functions import generator_loss, discriminator_loss

import numpy as np

from imageio import imwrite

# Train the model for one epoch.
def train(
	generator, 
	discriminator, 
	mapping_net,
	dataset_iterator, 
	manager, 
	batch_size, 
	z_dim, 
	generator_optimizer, 
	discriminator_optimizer
	):
	"""
	Train the model for one epoch. Save a checkpoint every 500 or so batches.

	:param generator: generator model
	:param discriminator: discriminator model
	:param dataset_ierator: iterator over dataset, see preprocess.py for more information
	:param manager: the manager that handles saving checkpoints by calling save()

	:return: The average FID score over the epoch
	"""
	sum_fid = 0
	# Loop over our data until we run out
	for iteration, batch in enumerate(dataset_iterator):
		z = uniform((batch_size, z_dim), minval=-1, maxval=1)

		with GradientTape() as gen_tape, GradientTape() as disc_tape:
			style_space = mapping_net(z)
			# generated images
			G_sample = generator(z, training=True)

			# test discriminator against real images
			logits_real = discriminator(batch, training=True)
			# re-use discriminator weights on new inputs
			logits_fake = discriminator(G_sample, training=True)

			g_loss = generator_loss(logits_fake)
			d_loss = discriminator_loss(logits_real, logits_fake)
			
		# optimize the generator and the discriminator
		gen_gradients = gen_tape.gradient(g_loss, generator.trainable_variables)
		generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

		disc_gradients = disc_tape.gradient(d_loss, discriminator.trainable_variables)
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
