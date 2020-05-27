import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import random

import pprint
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import json
import matplotlib.pyplot as plt

import gan_models

pp = pprint.PrettyPrinter()


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# SpaceGAN-V179
	parser.add_argument('--name', type=str, default='SpaceGAN-V179')
	parser.add_argument('--checkpoint_dir', type=str, default='D:/Models/SpaceGAN')
	parser.add_argument('--num_samples', type=int, default=10)
	parser.add_argument('--num_interpolations', type=int, default=10)
	parser.add_argument('--sample_seed', type=int, default=11)
	parser.add_argument(
		'--sample_path',
		type=str,
		default=None,
		help='File path for sample image output. If [None] then will use [sample_type].png as file name.'
	)
	parser.add_argument(
		'--sample_type',
		type=str,
		default='interpolate',
		help='Options: [interpolate, independent, anchor_interpolate]'
	)

	args = parser.parse_args()
	config_path = os.path.join(args.checkpoint_dir, args.name, f'{args.name}.json')
	print('Loading model config...')
	with open(config_path, 'r') as f:
		gan_kwargs = json.load(f)
	pp.pprint(gan_kwargs)

	seed = args.sample_seed
	random.seed(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)

	gan_type = gan_kwargs['gan_type'].lower()
	if gan_type == 'dcgan':
		model = gan_models.DCGAN(**gan_kwargs)
	elif gan_type == 'wgan':
		model = gan_models.WGAN(**gan_kwargs)
	elif gan_type == 'wgangp':
		model = gan_models.WGANGP(**gan_kwargs)
	elif gan_type == 'infogan':
		model = gan_models.InfoGAN(**gan_kwargs)
	else:
		raise ValueError(f'Unimplemented type of GAN: {gan_type}')

	print('Building model...')
	model.build()

	print('Loading model...')
	model.load()

	print('Generating samples...')
	sample_type = args.sample_type.lower()
	if sample_type == 'interpolate':
		# [num_samples, num_interpolations]
		# images where we sample 2 x num_samples as [0] and [num_interpolations-1]
		# noise vectors and we interpolate between noise vectors for in-between indices.
		# Demonstrates progressive changes between latent noise vectors.
		sample_image = model.sample_interpolate(
			args.num_samples,
			args.num_interpolations
		)
	elif sample_type == 'independent':
		# [num_samples, num_samples]
		# images where we sample num_samples * num_samples and
		# simply reshape to a grid for ease of viewing. Samples are completely
		# independent of each other.
		sample_image = model.sample_independent(
			args.num_samples,
			args.num_samples,
		)
	elif sample_type == 'anchor_interpolate':
		# [num_interpolations, num_interpolations]
		# images where we sample 4 latent vectors for each of the four corners of the grid.
		# We interpolate between each of these corners to get edges of grid, and then we interpolate between edges.
		# Because of non-linear interpolation we must select direction, which we choose as up-to-down for non-edge
		# interpolations.
		sample_image = model.sample_anchor_interpolate(
			args.num_interpolations
		)
	else:
		raise ValueError(f'Unimplemented type of sampling: {sample_type}')

	sample_path = args.sample_path
	if sample_path is None:
		sample_path = f'{sample_type}.png'

	print(f'Saving {sample_type} sample to file: {sample_path}')
	plt.imsave(sample_path, sample_image, format='png')

	print('Closing model...')
	model.close()
