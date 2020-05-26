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
	# 1e-5
	parser.add_argument('--generator_lr', type=float, default=1e-5)
	# 1e-4
	parser.add_argument('--discriminator_lr', type=float, default=1e-4)
	# 0.5
	parser.add_argument('--beta1', type=float, default=0.5)
	# 0.999
	parser.add_argument('--beta2', type=float, default=0.999)
	# True
	parser.add_argument('--use_batch_norm', type=str2bool, nargs='?', const=True, default=True)
	# 0.9
	parser.add_argument('--bn_momentum', type=float, default=0.9)
	parser.add_argument('--generator_base_size', type=int, default=64)
	parser.add_argument('--discriminator_base_size', type=int, default=64)
	parser.add_argument('--label_smoothing_factor', type=float, default=1.0)
	parser.add_argument('--discriminator_noise_factor', type=float, default=0.0)
	parser.add_argument('--discriminator_noise_decay_zero_steps', type=int, default=100000)
	# dcgan
	parser.add_argument('--gan_type', type=str, default='dcgan')
	parser.add_argument('--epochs', type=int, default=300)
	# 512 x 512
	parser.add_argument('--crop_dataset', type=int, nargs=2, default=(512, 512))
	# 128 x 128
	parser.add_argument('--scale_dataset', type=int, nargs=2, default=(128, 128))
	parser.add_argument('--channels', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--generator_steps', type=int, default=1)
	parser.add_argument('--discriminator_steps', type=int, default=1)
	parser.add_argument('--max_grad_norm', type=float, default=100.0)
	parser.add_argument('--epsilon', type=float, default=1e-8)
	parser.add_argument('--regularizer_scale', type=float, default=0.0)

	parser.add_argument('--balance_loss', type=str2bool, nargs='?', const=True, default=False)
	parser.add_argument('--generator_loss_ratio', type=float, default=2.0)
	parser.add_argument('--discriminator_loss_ratio', type=float, default=1.0)
	parser.add_argument('--random_size', type=int, default=128)
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--checkpoint_dir', type=str, default='D:/Models/SpaceGAN')
	parser.add_argument('--data_dir', type=str, default='C:/Users/maxwe/Data')
	parser.add_argument('--files_list', type=str, default='D:/Data/nasa/observations_factors.txt')
	parser.add_argument('--log_dir', type=str, default='D:/Logs/SpaceGAN/logs')
	parser.add_argument('--log_steps', type=int, default=5)
	parser.add_argument('--generate_steps', type=int, default=50)
	parser.add_argument('--save_steps', type=int, default=1000)
	parser.add_argument(
		'--load', type=str2bool, nargs='?', const=True, default=True, help='Load model from checkpoint directory.'
	)

	args = parser.parse_args()
	config_path = os.path.join(args.checkpoint_dir, args.name, f'{args.name}.json')
	load = args.load or os.path.exists(config_path)
	if load:
		print('Loading model config...')
		with open(config_path, 'r') as f:
			gan_kwargs = json.load(f)
	else:
		gan_kwargs = vars(args)
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

	if load:
		print('Loading model from checkpoint...')
		try:
			model.load()
		except FileNotFoundError:
			print('Unable to find checkpoint, initializing...')
			model.init()
	else:
		print('Initializing model...')
		with open(config_path, 'w') as f:
			json.dump(
				gan_kwargs,
				f,
				indent=2
			)
		model.init()

	print('Training model...')
	model.train()

	print('Closing model...')
	model.close()
