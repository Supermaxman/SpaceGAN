import os
import abc

import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

from categorical_grid_plots import CategoricalPlotter

from tqdm import tqdm
import data_utils
import noise_utils

import custom_layers


class AbstractGAN(abc.ABC):

	def __init__(
			self,
			name,
			epochs,
			scale_dataset,
			batch_size,
			generator_lr,
			discriminator_lr,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps):

		self.name = name
		self.epochs = epochs
		self.scale_dataset = scale_dataset
		self.batch_size = batch_size
		self.generator_lr = generator_lr
		self.discriminator_lr = discriminator_lr
		self.generator_steps = generator_steps
		self.discriminator_steps = discriminator_steps
		self.random_size = random_size
		self.z_size = random_size

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.checkpoint_dir = os.path.join(checkpoint_dir, self.name)

		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		self.log_dir = os.path.join(log_dir, self.name)

		self.log_steps = log_steps
		self.generate_steps = generate_steps
		self.save_steps = save_steps

		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.name)

		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		self.data_dir = data_dir
		self.files_list = files_list

		self.image_paths = data_utils.load_path_factors(self.files_list)

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		self.random_size = random_size

		self.sample_noise = noise_utils.create_noise(
			self.random_size
		)

		# self.kernel_initializer = tf.initializers.orthogonal(gain=0.1)
		self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
		# self.kernel_initializer = tf.initializers.he_normal()
		# self.kernel_initializer = None
		# self.kernel_initializer = tf.glorot_normal_initializer()
		# self.kernel_initializer = tf.glorot_uniform_initializer()
		# self.resize_method = tf.image.ResizeMethod.BILINEAR
		# self.resize_method = tf.image.ResizeMethod.BICUBIC
		self.resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR

		self.epsilon = 1e-8
		self.max_grad_norm = max_grad_norm
		# self.regularizer_scale = 0.0
		self.regularizer_scale = 0.0

		self.balance_loss = False
		# if generator loss * ratio < discriminator loss then don't train generator
		self.g_loss_ratio = 1.0
		# if discriminator loss * ratio < generator loss then don't train discriminator
		self.d_loss_ratio = 1.0

	def build_inputs(self):
		with tf.variable_scope('inputs'):

			print('Loading dataset...')
			self.dataset = data_utils.create_dataset(
				self.data_dir,
				self.batch_size,
				self.scale_dataset
			)
			self.dataset_size = sum(self.image_paths)
			self.channels = 3

			print('Dataset: {}'.format('SPACE'))
			print(' - size: {}'.format(self.dataset_size))
			print(' - channels: {}'.format(self.channels))

			self.data_iterator = self.dataset.make_one_shot_iterator()
			self.data_init_op = None

			self.real_images = self.data_iterator.get_next(
				name='real_images'
			)

			self.zc_vectors = tf.placeholder(
				tf.float32,
				[None, self.z_size],
				name='zc_vectors'
			)

			self.is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

			self.global_step = tf.Variable(0, trainable=False, name='global_step')

			# self.generator_regularizer = None
			# self.discriminator_regularizer = None
			# self.mutual_info_regularizer = None
			if self.regularizer_scale > 0.0:
				self.generator_regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_scale)
				self.discriminator_regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_scale)
				self.mutual_info_regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_scale)
			else:
				self.generator_regularizer = None
				self.discriminator_regularizer = None
				self.mutual_info_regularizer = None

	def build_generator(self):
		with tf.variable_scope('generator'):
			def layer(inputs, filters, kernel_size, strides=(2, 2),
								padding='same', use_bn=True, use_reg=True,
								activation=tf.nn.leaky_relu):
				if use_reg:
					kernel_regularizer = self.generator_regularizer
				else:
					kernel_regularizer = None

				# TODO add better upscaling method
				out = tf.layers.conv2d_transpose(
					inputs, filters, kernel_size,
					strides=strides, padding=padding,
					kernel_regularizer=kernel_regularizer,
					kernel_initializer=self.kernel_initializer,
					use_bias=not use_bn
				)
				# TODO more efficient upconv2d probably
				# out = custom_layers.upconv2d(
				# 	inputs, filters, kernel_size,
				# 	up_strides=strides,
				# 	conv_strides=(1, 1),
				# 	padding=padding,
				# 	resize_method=self.resize_method,
				# 	kernel_regularizer=kernel_regularizer,
				# 	kernel_initializer=self.kernel_initializer
				# )

				if use_bn:
					# out = tf.layers.batch_normalization(
					# 	out,
					# 	training=self.is_training,
					# 	momentum=0.9,
					# 	center=True,
					# 	scale=True
					# )
					out = tf.contrib.layers.layer_norm(
						out
					)
				if activation is not None:
					out = activation(out)
				return out

			G = self.zc_vectors
			G = tf.layers.dense(
				G,
				units=4*4*32,
				activation=None
			)
			G = tf.reshape(G, shape=[-1, 4, 4, 32])
			G = layer(G, 256, 4, strides=2)
			G = layer(G, 128, 4, strides=2)
			G = layer(G, 64, 4, strides=2)
			G = layer(G, 64, 4, strides=2)
			G = layer(G, 64, 4, strides=2)
			G = layer(G, 64, 4, strides=2)
			G = layer(G, self.channels, 4, strides=2, use_bn=False, use_reg=False, activation=tf.nn.tanh)
			self.fake_images = tf.identity(G, name='fake_images')

	def build_discriminator(self):
			def apply_discriminator(x):
				with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
					activation = tf.nn.leaky_relu
					x = tf.layers.conv2d(
						x,
						filters=64,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x = tf.layers.conv2d(
						x,
						filters=64,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x = tf.layers.conv2d(
						x,
						filters=64,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x = tf.layers.conv2d(
						x,
						filters=64,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x = tf.layers.conv2d(
						x,
						filters=128,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x = tf.layers.conv2d(
						x,
						filters=256,
						kernel_size=4,
						strides=2,
						padding='same',
						kernel_regularizer=self.discriminator_regularizer,
						kernel_initializer=self.kernel_initializer,
						use_bias=False
					)
					x = tf.contrib.layers.layer_norm(
						x
					)
					x = activation(x)
					x_shape = x.get_shape().as_list()

					# x = tf.reshape(x, [-1, np.prod(x_shape[1:])])
					x = tf.layers.average_pooling2d(x, pool_size=(x_shape[1], x_shape[2]), strides=1)
					x = tf.reshape(x, [-1, x_shape[3]])
					F = x
					x = tf.layers.dense(
						x,
						units=1,
						kernel_initializer=self.kernel_initializer
					)
					C = x
					D = tf.nn.sigmoid(x)
					return F, C, D
			self.apply_discriminator = apply_discriminator
			self.real_features, self.real_score, self.real_prob = apply_discriminator(self.real_images)
			self.fake_features, self.fake_score, self.fake_prob = apply_discriminator(self.fake_images)

	def build_train(self):
		# generator wants to maximize fake classification
		self.generator_loss = self.build_generator_loss()

		# discriminator wants to maximize real classification and minimize fake classification
		self.discriminator_loss = self.build_discriminator_loss()

		with tf.variable_scope('train'):

			self.generator_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES,
				scope='generator')
			self.discriminator_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES,
				scope='discriminator')

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			#
			generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_lr, beta1=0.0, beta2=0.9)
			discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_lr, beta1=0.0, beta2=0.9)
			# generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.generator_lr)
			# discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.discriminator_lr)

			self.increment_global_step = tf.assign_add(self.global_step, 1)

			self.generator_regularizer_loss = tf.losses.get_regularization_loss(
				scope='generator',
				name='generator_regularizer_loss')
			self.discriminator_regularizer_loss = tf.losses.get_regularization_loss(
				scope='discriminator',
				name='discriminator_regularizer_loss')

			self.total_generator_loss = self.generator_loss + self.generator_regularizer_loss
			self.total_discriminator_loss = self.discriminator_loss + self.discriminator_regularizer_loss

			with tf.control_dependencies(update_ops):
				def clip_train_op(optimizer, loss, params, name):
					with tf.variable_scope(name):
						grads_and_vars = optimizer.compute_gradients(loss, params)
						actual_grads_and_vars = [(gv[0], gv[1]) for gv in grads_and_vars if gv[0] is not None]
						# max_grad = tf.reduce_max([tf.reduce_max(tf.abs(gv[0])) for gv in actual_grads_and_vars])
						# max_grad_norm = tf.reduce_max([tf.norm(gv[0]) for gv in actual_grads_and_vars])
						# clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in
						# 													actual_grads_and_vars]
						clipped_grads_and_vars = actual_grads_and_vars
						optim = optimizer.apply_gradients(clipped_grads_and_vars, name=name)
						max_grad = tf.constant(0)
						max_grad_norm = tf.constant(0)
					return optim, max_grad, max_grad_norm

				self.train_generator, self.generator_max_grad, self.generator_max_grad_norm = clip_train_op(
					generator_optimizer,
					self.total_generator_loss,
					self.generator_params,
					name='train_generator')

				self.train_discriminator, self.discriminator_max_grad, self.discriminator_max_grad_norm = clip_train_op(
					discriminator_optimizer,
					self.total_discriminator_loss,
					self.discriminator_params,
					name='train_discriminator')

	def build_summaries(self):
		summaries = []
		with tf.variable_scope('loss_summaries'):
			summaries.append(
				tf.summary.scalar(
					name='generator_loss',
					tensor=self.generator_loss)
			)
			summaries.append(
				tf.summary.scalar(
					name='discriminator_loss',
					tensor=self.discriminator_loss)
			)
		with tf.variable_scope('grad_summaries'):
			summaries.append(
				tf.summary.scalar(
					name='generator_max_grad',
					tensor=self.generator_max_grad)
			)
			summaries.append(
				tf.summary.scalar(
					name='generator_max_grad_norm',
					tensor=self.generator_max_grad_norm)
			)
			summaries.append(
				tf.summary.scalar(
					'discriminator_max_grad',
					tensor=self.discriminator_max_grad)
			)
			summaries.append(
				tf.summary.scalar(
					'discriminator_max_grad_norm',
					tensor=self.discriminator_max_grad_norm)
			)

		with tf.variable_scope('discriminator_summaries'):
			summaries.append(
				tf.summary.scalar(
					name='real_score',
					tensor=tf.reduce_mean(self.real_score))
			)
			summaries.append(
				tf.summary.scalar(
					name='real_prob',
					tensor=tf.reduce_mean(self.real_prob))
			)
			summaries.append(
				tf.summary.histogram(
					name='real_score_hist',
					values=self.real_score)
			)
			summaries.append(
				tf.summary.histogram(
					name='real_prob_hist',
					values=self.real_prob)
			)

		with tf.variable_scope('generator_summaries'):
			summaries.append(
				tf.summary.scalar(
					name='fake_score',
					tensor=tf.reduce_mean(self.fake_score))
			)
			summaries.append(
				tf.summary.scalar(
					name='fake_prob',
					tensor=tf.reduce_mean(self.fake_prob))
			)
			summaries.append(
				tf.summary.histogram(
					name='fake_score_hist',
					values=self.fake_score)
			)
			summaries.append(
				tf.summary.histogram(
					name='fake_prob_hist',
					values=self.fake_prob)
			)

		with tf.variable_scope('regularizer_summaries'):
			summaries.append(
				tf.summary.scalar(
					name='generator_regularizer_loss',
					tensor=self.generator_regularizer_loss)
			)
			summaries.append(
				tf.summary.scalar(
					name='discriminator_regularizer_loss',
					tensor=self.discriminator_regularizer_loss)
			)

		with tf.variable_scope('summaries'):
			self.summaries = tf.summary.merge(
				inputs=summaries,
				name='summaries'
			)

	def build(self):
		self.build_inputs()
		self.build_generator()
		self.build_discriminator()
		self.build_train()
		self.build_final()
		self.build_summaries()
		self.saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)

	def init(self):
		self.start_session()
		self.sess.run(tf.global_variables_initializer())

	@abc.abstractmethod
	def build_generator_loss(self):
		pass

	@abc.abstractmethod
	def build_discriminator_loss(self):
		pass

	def train_step(self, prev_step, prev_d_loss, prev_g_loss):
		is_log_step = (prev_step + 1) % self.log_steps == 0

		train_g = True
		train_d = True
		if self.balance_loss:
			if prev_g_loss * self.g_loss_ratio < prev_d_loss:
				train_g = False
			elif prev_d_loss * self.d_loss_ratio < prev_g_loss:
				train_d = False

		discriminator_fetches = [
															self.train_discriminator,
															self.discriminator_loss] + ([self.summaries] if is_log_step else [])

		generator_fetches = [
													self.train_generator,
													self.generator_loss] + ([self.summaries] if is_log_step else [])

		if train_d:
			for d_idx in range(self.discriminator_steps):
				noise = self.sample_noise(self.batch_size)
				d_results = self.sess.run(
					discriminator_fetches,
					feed_dict={
						self.zc_vectors: noise,
						self.is_training: True
					}
				)
				_, d_loss = d_results[:2]
				summary = d_results[-1] if is_log_step else None
		else:
			d_loss = prev_d_loss

		if train_g:
			for g_idx in range(self.generator_steps):
				noise = self.sample_noise(self.batch_size)
				g_results = self.sess.run(
					generator_fetches,
					feed_dict={
						self.zc_vectors: noise,
						self.is_training: True
					}
				)
				_, g_loss = g_results[:2]
				summary = g_results[-1] if is_log_step else None
		else:
			g_loss = prev_g_loss

		step = self.sess.run(self.increment_global_step)

		return d_loss, g_loss, step, summary

	def train_epoch(self):
		avg_d_loss = 0.0
		avg_g_loss = 0.0

		epoch_size = self.dataset_size // self.batch_size
		step = self.sess.run(self.global_step)

		d_loss, g_loss = 0.0, 0.0

		for b_idx in tqdm(range(epoch_size), total=epoch_size):

			d_loss, g_loss, step, g_summary = self.train_step(step, d_loss, g_loss)

			if g_summary is not None:
				self.summary_writer.add_summary(g_summary, step)

			if step % self.generate_steps == 0:
				self.plotter.generate_images(self.sess, iteration=step)

			if step % self.save_steps == 0:
				self.saver.save(self.sess, self.checkpoint_path, global_step=step, write_meta_graph=False)

			avg_d_loss += d_loss
			avg_g_loss += g_loss

		avg_d_loss = avg_d_loss / epoch_size
		avg_g_loss = avg_g_loss / epoch_size

		return avg_d_loss, avg_g_loss, step

	def train(self):
		self.count_trainable_variables()
		self.summary_writer = tf.summary.FileWriter(
			self.log_dir,
			self.sess.graph)

		# TODO also output an image log file
		self.plotter = CategoricalPlotter(
			random_size=self.random_size,
			journalist=self.summary_writer,
			log_dir=self.log_dir,
			generate=lambda sess, x: sess.run(
				self.fake_images,
				{self.zc_vectors: x}
			),
			row_size=10
		)
		if self.data_init_op is not None:
			self.sess.run(self.data_init_op)

		# step = self.sess.run(self.global_step)
		# epoch_size = self.dataset_size // self.batch_size
		# TODO calculate remaining epochs
		for epoch in range(self.epochs):
			avg_d_loss, avg_g_loss, step = self.train_epoch()

			state = {
				'Epoch': epoch,
				'D Loss': '{:.5f}'.format(avg_d_loss),
				'G Loss': '{:.5f}'.format(avg_g_loss)
			}

			print(state)

	def count_trainable_variables(self):
		total_variables = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape().as_list()
			variable_parameters = 1
			for dim_val in shape:
				variable_parameters *= dim_val
			print('{} | {} -> {}'.format(variable.name, shape, variable_parameters))
			total_variables += variable_parameters
		print('Total number of trainable variables: {}'.format(total_variables))

	def load(self):
		self.start_session()
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
		else:
			raise Exception('No checkpoint found!')

	def start_session(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

	def close(self):
		# TODO maybe make this within a context manager?
		self.sess.close()
		tf.reset_default_graph()

	def build_final(self):
		pass


class DCGAN(AbstractGAN):
	def __init__(
			self,
			name,
			epochs,
			scale_dataset,
			batch_size,
			generator_lr,
			discriminator_lr,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps):
		super().__init__(
			name,
			epochs,
			scale_dataset,
			batch_size,
			generator_lr,
			discriminator_lr,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps)

	def build_generator_loss(self):
		with tf.variable_scope('generator_loss'):
			generator_loss = tf.reduce_mean(
				-tf.log(
					self.fake_prob + self.epsilon
				)
			)
		return generator_loss

	def build_discriminator_loss(self):
		with tf.variable_scope('discriminator_loss'):
			discriminator_loss = tf.reduce_mean(
				- tf.log(
					self.real_prob + self.epsilon
				)
				- tf.log(
					1 - self.fake_prob + self.epsilon
				)
			)
		return discriminator_loss


class DCWGANGP(AbstractGAN):
	def __init__(
			self,
			name,
			epochs,
			scale_dataset,
			batch_size,
			generator_lr,
			discriminator_lr,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps):
		super().__init__(
			name,
			epochs,
			scale_dataset,
			batch_size,
			generator_lr,
			discriminator_lr,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps)

	def build_generator_loss(self):
		with tf.variable_scope('generator_loss'):
			generator_loss = -tf.reduce_mean(
				self.fake_score
			)
		return generator_loss

	def build_discriminator_loss(self):
		with tf.variable_scope('discriminator_loss'):
			real_loss = -tf.reduce_mean(
				self.real_score
			)
			fake_loss = tf.reduce_mean(
				self.fake_score
			)
			discriminator_loss = real_loss + fake_loss

		alpha = tf.random_uniform(
			shape=[tf.shape(self.real_score)[0], 1, 1, 1],
			minval=0.,
			maxval=1.
		)
		differences = self.fake_images - self.real_images
		interpolates = self.real_images + (alpha * differences)
		gradients = tf.gradients(self.apply_discriminator(interpolates), [interpolates])
		gradients = gradients[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
		gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
		discriminator_loss += 10.0 * gradient_penalty

		return discriminator_loss

