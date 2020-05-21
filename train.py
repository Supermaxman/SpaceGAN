import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import argparse
import random

import pprint
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np

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
	parser.add_argument('--name', type=str, default='SpaceGAN-V60')
	parser.add_argument('--gan_type', type=str, default='dcwgangp')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--scale_dataset', type=int, nargs=2, default=(512, 512))
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--generator_lr', type=float, default=5e-5)
	parser.add_argument('--discriminator_lr', type=float, default=5e-5)
	parser.add_argument('--generator_steps', type=int, default=1)
	parser.add_argument('--discriminator_steps', type=int, default=5)
	parser.add_argument('--max_grad_norm', type=float, default=100.0)

	parser.add_argument('--random_size', type=int, default=128)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--checkpoint_dir', type=str, default='D:/Models/SpaceGAN')
	parser.add_argument('--data_dir', type=str, default='C:/Users/maxwe/Data')
	parser.add_argument('--files_list', type=str, default='D:/Data/nasa/observations_factors.txt')
	parser.add_argument('--log_dir', type=str, default='D:/Logs/SpaceGAN/logs')
	parser.add_argument('--log_steps', type=int, default=10)
	parser.add_argument('--generate_steps', type=int, default=100)
	parser.add_argument('--save_steps', type=int, default=1000)
	parser.add_argument('--load', type=str2bool, nargs='?',
											const=True, default=False,
											help='Load model from checkpoint directory.')

	args = parser.parse_args()
	pp.pprint(args)

	seed = args.seed
	random.seed(seed)
	np.random.seed(seed)
	tf.set_random_seed(seed)

	gan_kwargs = {
		'name': args.name,
		'epochs': args.epochs,
		'generator_lr': args.generator_lr,
		'discriminator_lr': args.discriminator_lr,
		'generator_steps': args.generator_steps,
		'discriminator_steps': args.discriminator_steps,
		'scale_dataset': args.scale_dataset,
		'batch_size': args.batch_size,
		'max_grad_norm': args.max_grad_norm,
		'random_size': args.random_size,
		'checkpoint_dir': args.checkpoint_dir,
		'data_dir': args.data_dir,
		'files_list': args.files_list,
		'log_dir': args.log_dir,
		'log_steps': args.log_steps,
		'generate_steps': args.generate_steps,
		'save_steps': args.save_steps
	}

	gan_type = args.gan_type.lower()
	if gan_type == 'dcgan':
		model = gan_models.DCGAN(**gan_kwargs)
	elif gan_type == 'dcwgangp':
		model = gan_models.DCWGANGP(**gan_kwargs)
	else:
		raise Exception('Unimplemented type of GAN: {}'.format(args.gan_type))

	print('Building model...')
	model.build()

	if args.load:
		print('Loading model from checkpoint...')
		model.load()
	else:
		print('Initializing model...')
		model.init()

	print('Training model...')
	model.train()

	print('Closing model...')
	model.close()
