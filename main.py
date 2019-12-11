# StyleGAN: https://arxiv.org/pdf/1812.04948.pdf
# GANs: https://arxiv.org/pdf/1406.2661.pdf
# Progressive GANs: https://arxiv.org/pdf/1710.10196.pdf
# ADAin: https://arxiv.org/pdf/1703.06868.pdf

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose

from preprocess import load_image_batch
from get_args import get_args
from train_test import train, test
from models import Generator_Model, Discriminator_Model, make_noise_scale_net, make_affine_transform_net, make_mapping_net

import numpy as np
import os

args = get_args(want_gpu=True)

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		
def iter_len(iterr):
	l = 0
	for i, e in enumerate(iterr):
		print(e)
		l += 1
	return l

def main():
	# Load a batch of images (to feed to the discriminator)
	rock = load_image_batch(args.img_dir + '/rock', batch_size=args.batch_size, n_threads=args.num_data_threads)
	rap = load_image_batch(args.img_dir + '/rap', batch_size=args.batch_size, n_threads=args.num_data_threads)
	jazz = load_image_batch(args.img_dir + '/jazz', batch_size=args.batch_size, n_threads=args.num_data_threads)
	
	rock_len = iter_len(rock)
	rap_len = iter_len(rap)
	jazz_len = iter_len(jazz)

	genre_labels = np.concatenate([np.full(rock_len, 0), np.full(rap_len, 1), np.full(jazz_len, 2)])
	dataset = chain(rock, rap, jazz)
	# dataset_iterator = list(dataset_iterator)
	# np.random.shuffle(dataset_iterator)
	# dataset_iterator = iter(dataset_iterator)

	# Initialize models
	generator = Generator_Model(args.num_channels, args.num_genres, args.img_side_len)
	discriminator = Discriminator_Model(args.num_channels, args.num_genres, args.img_side_len)
	mapping_net = make_mapping_net(args.mapping_dim, args.z_dim)
	noise_net = make_noise_scale_net(args.num_channels)
	adain_net = make_affine_transform_net(args.num_channels, args.z_dim)

	# For saving/loading models
	checkpoint_dir = './checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
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
					avg_fid = train(generator, discriminator, dataset, genre_labels, manager, mapping_net, noise_net, adain_net)
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


