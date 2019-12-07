import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

from preprocess import load_image_batch
from get_args import get_args
from train_test import train, test
from models import make_generator, make_discriminator

import os

from tensorflow.keras.optimizers import Adam

args = get_args()
generator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)
discriminator_optimizer = Adam(learning_rate=args.learn_rate, beta_1=args.beta1)

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
	# Load a batch of images (to feed to the discriminator)
	dataset_iterator = load_image_batch(args.img_dir, batch_size=args.batch_size, n_threads=args.num_data_threads)

	# Initialize generator and discriminator models
	generator = make_generator(args.num_channels, args.z_dim)
	discriminator = make_discriminator(args.num_channels)

	# For saving/loading models
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer,)
	manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
	# Ensure the output directory exists
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	if args.restore_checkpoint or args.mode == 'test':
		# restores the latest checkpoint using from the manager
		checkpoint.restore(manager.latest_checkpoint) 

	try:
		# Specify an invalid GPU device
		with tf.device('/device:' + args.device):
			if args.mode == 'train':
				for epoch in range(0, args.num_epochs):
					print('========================== EPOCH %d  ==========================' % epoch)
					avg_fid = train(generator, discriminator, dataset_iterator, manager, args.batch_size, args.z_dim, generator_optimizer, discriminator_optimizer)
					print("Average FID for Epoch: " + str(avg_fid))
					# Save at the end of the epoch, too
					print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
					manager.save()
			if args.mode == 'test':
				test(generator, args.batch_size, args.z_dim, args.out_dir)
	except RuntimeError as e:
		print(e)

if __name__ == '__main__':
   main()


