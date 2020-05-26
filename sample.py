import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import random

import pprint
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import json

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

	args = parser.parse_args()
	config_path = os.path.join(args.checkpoint_dir, args.name, f'{args.name}.json')
	print('Loading model config...')
	with open(config_path, 'r') as f:
		gan_kwargs = json.load(f)
	pp.pprint(gan_kwargs)

	seed = gan_kwargs['seed']
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
		raise Exception('Unimplemented type of GAN: {}'.format(gan_type))

	print('Building model...')
	model.build()

	print('Loading model...')
	model.load()

	print('Generating samples...')
	model.sample(args.num_samples, args.num_interpolations)

	print('Closing model...')
	model.close()
