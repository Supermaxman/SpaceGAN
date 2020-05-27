import os
import abc

import numpy as np
import tensorflow as tf

from categorical_grid_plots import CategoricalPlotter, create_image_strip

from tqdm import tqdm
import data_utils
import noise_utils
import interpolate

import custom_layers


class AbstractGAN(abc.ABC):

	def __init__(
			self,
			name,
			epochs,
			scale_dataset,
			crop_dataset,
			channels,
			batch_size,
			generator_lr,
			discriminator_lr,
			beta1,
			beta2,
			bn_momentum,
			generator_base_size,
			discriminator_base_size,
			label_smoothing_factor,
			discriminator_noise_factor,
			discriminator_noise_decay_zero_steps,
			use_batch_norm,
			balance_loss,
			generator_loss_ratio,
			discriminator_loss_ratio,
			generator_steps,
			discriminator_steps,
			max_grad_norm,
			epsilon,
			regularizer_scale,
			random_size,
			checkpoint_dir,
			data_dir,
			files_list,
			log_dir,
			log_steps,
			generate_steps,
			save_steps,
			*args,
			**kwargs
	):

		self.name = name
		self.epochs = epochs
		self.scale_dataset = scale_dataset
		self.crop_dataset = crop_dataset
		self.batch_size = batch_size
		self.generator_lr = generator_lr
		self.discriminator_lr = discriminator_lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.bn_momentum = bn_momentum
		self.generator_steps = generator_steps
		self.discriminator_steps = discriminator_steps
		self.random_size = random_size
		self.z_size = random_size
		self.use_batch_norm = use_batch_norm
		self.generator_base_size = generator_base_size
		self.discriminator_base_size = discriminator_base_size

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

		self.dataset_size = sum(self.image_paths)
		self.channels = channels

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		self.random_size = random_size

		self.sample_noise = noise_utils.create_noise(
			self.random_size
		)

		self.label_smoothing_factor = label_smoothing_factor

		self.discriminator_noise_factor = discriminator_noise_factor
		self.discriminator_noise_decay_zero_steps = discriminator_noise_decay_zero_steps
		self.kernel_initializer = None
		# TODO add kernel_initializer support everywhere
		assert self.kernel_initializer is None
		# self.kernel_initializer = tf.initializers.orthogonal(gain=0.1)
		# self.kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
		# self.kernel_initializer = tf.initializers.he_normal()

		self.epsilon = epsilon
		self.max_grad_norm = max_grad_norm
		self.regularizer_scale = regularizer_scale

		self.balance_loss = balance_loss
		# if generator loss * ratio < discriminator loss then don't train generator
		self.generator_loss_ratio = generator_loss_ratio
		# if discriminator loss * ratio < generator loss then don't train discriminator
		self.discriminator_loss_ratio = discriminator_loss_ratio

	def build_inputs(self):
		with tf.variable_scope('inputs'):

			print('Loading dataset...')
			self.dataset = data_utils.create_dataset(
				self.data_dir,
				self.batch_size,
				self.scale_dataset,
				self.crop_dataset
			)

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
			activation = tf.nn.relu
			layer = tf.layers.conv2d_transpose
			dense_layer = tf.layers.dense
			g_out = self.zc_vectors
			base_size = self.generator_base_size
			noise_dense_input_shape = [-1, 4, 4, 2*2*2*2*base_size]
			g_out = dense_layer(
				g_out,
				units=np.prod(noise_dense_input_shape[1:])
			)

			g_out = tf.reshape(g_out, shape=noise_dense_input_shape)

			def block(x, filters):
				with tf.variable_scope(name_or_scope=None, default_name='block'):
					x = layer(
						x,
						filters=filters,
						kernel_size=4,
						strides=2,
						padding='same'
					)
					x = activation(x)
					if self.use_batch_norm:
						x = tf.layers.batch_normalization(x, training=self.is_training, momentum=self.bn_momentum)
					return x

			g_out = block(g_out, 2*2*2*2*base_size)
			g_out = block(g_out, 2*2*2*base_size)
			g_out = block(g_out, 2*2*base_size)
			g_out = block(g_out, 2*base_size)
			g_out = block(g_out, base_size)

			g_out = tf.layers.conv2d(g_out, filters=3, kernel_size=3, strides=1, padding='same', activation=tf.nn.tanh)

			self.fake_images = tf.identity(g_out, name='fake_images')

	def build_discriminator(self):
			def apply_discriminator(d_out):
				with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
					base_size = self.discriminator_base_size
					activation = tf.nn.leaky_relu
					layer = custom_layers.spectral_conv2d
					# layer = tf.layers.conv2d
					dense_layer = custom_layers.spectral_dense
					# dense_layer = tf.layers.dense

					def block(x, filters):
						with tf.variable_scope(name_or_scope=None, default_name='block'):
							x = layer(
								x,
								filters=filters,
								kernel_size=3,
								strides=1,
								padding='same'
							)
							x = activation(x)
							x = layer(
								x,
								filters=filters,
								kernel_size=4,
								strides=2,
								padding='same'
							)
							x = activation(x)
							return x

					d_out = block(d_out, base_size)
					d_out = block(d_out, 2*base_size)
					d_out = block(d_out, 2*2*base_size)
					d_out = block(d_out, 2*2*2*base_size)
					d_out = block(d_out, 2*2*2*2*base_size)

					d_shape = d_out.get_shape().as_list()

					d_out = tf.reshape(d_out, [-1, np.prod(d_shape[1:])])
					feature_out = d_out
					d_out = dense_layer(
						d_out,
						units=1
					)
					score_out = d_out
					d_out = tf.nn.sigmoid(d_out)
					return feature_out, score_out, d_out

			self.apply_discriminator = apply_discriminator

			real_inputs = self.real_images
			fake_inputs = self.fake_images

			if self.discriminator_noise_factor > 0.0:
				decay_factor = tf.maximum(
					1.0 - (tf.cast(self.global_step, tf.float32) / self.discriminator_noise_decay_zero_steps),
					0.0
				)
				decayed_noise_factor = self.discriminator_noise_factor * decay_factor

				real_noise = tf.random.normal(shape=tf.shape(self.real_images), mean=0, stddev=decayed_noise_factor)
				fake_noise = tf.random.normal(shape=tf.shape(self.fake_images), mean=0, stddev=decayed_noise_factor)
				real_inputs += real_noise
				fake_noise += fake_noise

			self.real_features, self.real_score, self.real_prob = apply_discriminator(real_inputs)
			self.fake_features, self.fake_score, self.fake_prob = apply_discriminator(fake_inputs)

	def build_train(self):
		# generator wants to maximize fake classification
		self.generator_loss = self.build_generator_loss()

		# discriminator wants to maximize real classification and minimize fake classification
		self.discriminator_loss = self.build_discriminator_loss()

		with tf.variable_scope('train'):

			self.all_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES,
				scope='all'
			)
			self.generator_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES,
				scope='generator'
			) + self.all_params
			self.discriminator_params = tf.get_collection(
				tf.GraphKeys.TRAINABLE_VARIABLES,
				scope='discriminator'
			) + self.all_params

			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			generator_optimizer = tf.train.AdamOptimizer(
				learning_rate=self.generator_lr,
				beta1=self.beta1,
				beta2=self.beta2
			)
			discriminator_optimizer = tf.train.AdamOptimizer(
				learning_rate=self.discriminator_lr,
				beta1=self.beta1,
				beta2=self.beta2
			)

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
						max_grad = tf.reduce_max([tf.reduce_max(tf.abs(gv[0])) for gv in actual_grads_and_vars])
						max_grad_norm = tf.reduce_max([tf.norm(gv[0]) for gv in actual_grads_and_vars])

						if self.max_grad_norm > 0.0:
							clipped_grads_and_vars = [
								(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in actual_grads_and_vars
							]
						else:
							clipped_grads_and_vars = actual_grads_and_vars

						train_op = optimizer.apply_gradients(clipped_grads_and_vars, name=name)
					return train_op, max_grad, max_grad_norm

				self.train_generator, self.generator_max_grad, self.generator_max_grad_norm = clip_train_op(
					generator_optimizer,
					self.total_generator_loss,
					self.generator_params,
					name='train_generator'
				)

				self.train_discriminator, self.discriminator_max_grad, self.discriminator_max_grad_norm = clip_train_op(
					discriminator_optimizer,
					self.total_discriminator_loss,
					self.discriminator_params,
					name='train_discriminator'
				)

	def build_summaries(self):
		with tf.variable_scope('loss_summaries'):
			tf.summary.scalar(
				name='generator_loss',
				tensor=self.generator_loss
			)

			tf.summary.scalar(
				name='discriminator_loss',
				tensor=self.discriminator_loss)
		if self.max_grad_norm > 0.0:
			with tf.variable_scope('grad_summaries'):
				tf.summary.scalar(
					name='generator_max_grad',
					tensor=self.generator_max_grad
				)

				tf.summary.scalar(
					name='generator_max_grad_norm',
					tensor=self.generator_max_grad_norm
				)
				tf.summary.scalar(
					'discriminator_max_grad',
					tensor=self.discriminator_max_grad
				)

				tf.summary.scalar(
					'discriminator_max_grad_norm',
					tensor=self.discriminator_max_grad_norm
				)

		with tf.variable_scope('discriminator_summaries'):
			tf.summary.scalar(
				name='real_score',
				tensor=tf.reduce_mean(self.real_score)
			)
			tf.summary.scalar(
				name='real_prob',
				tensor=tf.reduce_mean(self.real_prob)
			)

		with tf.variable_scope('generator_summaries'):
			tf.summary.scalar(
				name='fake_score',
				tensor=tf.reduce_mean(self.fake_score)
			)

			tf.summary.scalar(
				name='fake_prob',
				tensor=tf.reduce_mean(self.fake_prob)
			)

		if self.regularizer_scale > 0.0:
			with tf.variable_scope('regularizer_summaries'):
				tf.summary.scalar(
					name='generator_regularizer_loss',
					tensor=self.generator_regularizer_loss
				)

				tf.summary.scalar(
					name='discriminator_regularizer_loss',
					tensor=self.discriminator_regularizer_loss
				)

		with tf.variable_scope('summaries'):
			self.summaries = tf.summary.merge_all(name='summaries')

	def build(self):
		self.build_inputs()
		self.build_generator()
		self.build_discriminator()
		self.build_train()
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
			if prev_g_loss * self.generator_loss_ratio < prev_d_loss:
				train_g = False
			elif prev_d_loss * self.discriminator_loss_ratio < prev_g_loss:
				train_d = False

		discriminator_fetches = [
															self.train_discriminator,
															self.discriminator_loss] + ([self.summaries] if is_log_step and not train_g else [])

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
				summary = d_results[-1] if is_log_step and not train_g else None
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
				[self.fake_images, self.real_images],
				{self.zc_vectors: x}
			),
			row_size=10
		)
		if self.data_init_op is not None:
			self.sess.run(self.data_init_op)

		for epoch in range(self.epochs):
			avg_d_loss, avg_g_loss, step = self.train_epoch()

			state = {
				'Epoch': epoch,
				'D Loss': '{:.5f}'.format(avg_d_loss),
				'G Loss': '{:.5f}'.format(avg_g_loss)
			}

			print(state)

	@staticmethod
	def count_trainable_variables():
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
			raise FileNotFoundError('No checkpoint found!')

	def start_session(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)

	def close(self):
		self.sess.close()
		tf.reset_default_graph()

	def sample_interpolate(self, num_samples, num_interpolations=5):
		samples_noise = np.reshape(
			self.sample_noise(2*num_samples),
			[num_samples, 2, self.random_size]
		)
		assert num_interpolations >= 3

		interpolate_noise = np.zeros(
			shape=[num_samples, num_interpolations, self.random_size]
		)
		# TODO can probably vectorize this to be faster
		for i in range(num_samples):
			start_noise = samples_noise[i, 0]
			end_noise = samples_noise[i, 1]
			for j in range(num_interpolations):
				interp_val = j / (num_interpolations - 1)
				i_j_noise = interpolate.slerp_gaussian(interp_val, start_noise, end_noise)
				interpolate_noise[i, j] = i_j_noise

		interpolate_noise = np.reshape(
			interpolate_noise,
			[num_samples * num_interpolations, self.random_size]
		)

		sample_images = self.sess.run(
			self.fake_images,
			{self.zc_vectors: interpolate_noise}
		)
		_, h, w, c = sample_images.shape

		sample_images = np.reshape(
			sample_images,
			[num_samples, num_interpolations, h, w, c]
		)
		images = []
		for i in range(num_samples):
			image = create_image_strip(
				sample_images[i],
				gutter=1
			)
			images.append(image)

		sample_image = np.vstack(images)
		return sample_image

	def sample_independent(self, num_row_samples, num_column_samples):

		samples_noise = self.sample_noise(num_row_samples * num_column_samples)

		sample_images = self.sess.run(
			self.fake_images,
			{self.zc_vectors: samples_noise}
		)
		_, h, w, c = sample_images.shape

		sample_images = np.reshape(
			sample_images,
			[num_row_samples, num_column_samples, h, w, c]
		)
		images = []
		for i in range(num_row_samples):
			image = create_image_strip(
				sample_images[i],
				gutter=1
			)
			images.append(image)

		sample_image = np.vstack(images)
		return sample_image

	def sample_anchor_interpolate(self, num_interpolations=10):
		samples_noise = self.sample_noise(4)

		assert num_interpolations >= 3

		interpolate_noise = np.zeros(
			shape=[num_interpolations, num_interpolations, self.random_size]
		)
		# covers the following first
		# ul - - - ur
		#
		#
		#
		# bl - - - br

		for j in range(num_interpolations):
			j_interp_val = j / (num_interpolations - 1)
			ul_ur_j_noise = interpolate.slerp_gaussian(j_interp_val, samples_noise[0], samples_noise[1])
			interpolate_noise[0, j] = ul_ur_j_noise
			bl_br_j_noise = interpolate.slerp_gaussian(j_interp_val, samples_noise[2], samples_noise[3])
			interpolate_noise[-1, j] = bl_br_j_noise

			# then computes interpolations vertically:
			# ul - - - ur
			# |  | | | |
			# |  | | | |
			# |  | | | |
			# bl - - - br
			for i in range(num_interpolations):
				i_interp_val = i / (num_interpolations - 1)
				i_j_noise = interpolate.slerp_gaussian(i_interp_val, interpolate_noise[0, j], interpolate_noise[-1, j])
				interpolate_noise[i, j] = i_j_noise

		interpolate_noise = np.reshape(
			interpolate_noise,
			[num_interpolations * num_interpolations, self.random_size]
		)

		sample_images = self.sess.run(
			self.fake_images,
			{self.zc_vectors: interpolate_noise}
		)
		_, h, w, c = sample_images.shape

		sample_images = np.reshape(
			sample_images,
			[num_interpolations, num_interpolations, h, w, c]
		)
		images = []
		for i in range(num_interpolations):
			image = create_image_strip(
				sample_images[i],
				gutter=0
			)
			images.append(image)

		sample_image = np.vstack(images)
		return sample_image


class DCGAN(AbstractGAN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def build_generator_loss(self):
		with tf.variable_scope('generator_loss'):
			generator_loss = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(self.fake_score),
					logits=self.fake_score
				)
			)
		return generator_loss

	def build_discriminator_loss(self):
		with tf.variable_scope('discriminator_loss'):
			discriminator_loss = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.ones_like(self.real_score) * self.label_smoothing_factor,
					logits=self.real_score
				)
				+
				tf.nn.sigmoid_cross_entropy_with_logits(
					labels=tf.zeros_like(self.fake_score),
					logits=self.fake_score
				)
			)
		return discriminator_loss


class WGAN(AbstractGAN):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

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

		return discriminator_loss


class WGANGP(WGAN):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def build_discriminator_loss(self):
		discriminator_loss = super().build_discriminator_loss()

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
		# LAMBDA = 10 usually, but for larger images this makes the loss massive
		discriminator_loss += 1.0 * gradient_penalty

		return discriminator_loss


class InfoGAN(DCGAN):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.zc_loss_factor = 0.1
		self.num_continuous = 2
		assert self.num_continuous <= self.random_size

	def build_generator_loss(self):
		generator_loss = super().build_generator_loss()

		with tf.variable_scope('all', reuse=tf.AUTO_REUSE):
			# first num_continuous random variables are special InfoGan noise variables
			self.c_vectors = self.zc_vectors[:, :self.num_continuous]
			c_prediction = tf.layers.dense(
				self.fake_features,
				units=self.num_continuous,
				name='c_prediction'
			)
			# assumes 1 stddev, set from noise_utils
			std_config = tf.ones_like(c_prediction)
			diff = (self.c_vectors - c_prediction) / (std_config + self.epsilon)
			zc_loss = - 0.5 * np.log(2 * np.pi) - tf.log(std_config + self.epsilon) - 0.5 * tf.square(diff)
			zc_loss = -zc_loss
			zc_loss = tf.reduce_sum(
				zc_loss,
				axis=-1,
			)
			self.zc_loss = tf.reduce_mean(zc_loss, axis=-1)
		tf.summary.scalar(
			name='generator_zc_loss',
			tensor=self.zc_loss
		)
		generator_loss += self.zc_loss_factor * self.zc_loss
		return generator_loss

	def build_discriminator_loss(self):
		discriminator_loss = super().build_discriminator_loss()
		discriminator_loss += self.zc_loss_factor * self.zc_loss
		return discriminator_loss
