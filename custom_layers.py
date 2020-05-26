import tensorflow as tf


# TODO
# class BnWeightNormConv2D(WeightNormConv2D):
#    pass

# based on https://arxiv.org/pdf/1807.03247.pdf
# adds x, y coordinates normalized to [-1, 1] as additional
# features for a given 2D input
class CoordConvTransform2D(tf.layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs):
		_, w, h, channel = inputs.get_shape()
		bs = tf.shape(inputs)[0]

		# Get indices
		indices = tf.where(tf.ones(tf.stack([bs, w, h])))
		indices = tf.cast(indices, tf.float32)
		canvas = tf.reshape(indices, tf.stack([bs, w, h, 3]))[..., 1:]
		# Normalize the canvas
		w_max = w
		h_max = h
		if w > 1:
			w_max = w - 1
		if h > 1:
			h_max = h - 1
		canvas = canvas / tf.cast(tf.reshape(tf.stack([w_max, h_max]), [1, 1, 1, 2]), tf.float32)
		canvas = (canvas * 2) - 1

		# Concatenate channel-wise
		outputs = tf.concat([inputs, canvas], axis=-1)
		return outputs


class WeightNormDense(tf.layers.Dense):
	def __init__(
			self,
			units,
			activation=None,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=tf.zeros_initializer(),
			g_initializer=tf.ones_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			g_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			g_constraint=None,
			trainable=True,
			name=None,
			**kwargs):
		super().__init__(
			units,
			activation,
			use_bias,
			kernel_initializer,
			bias_initializer,
			kernel_regularizer,
			bias_regularizer,
			activity_regularizer,
			kernel_constraint,
			bias_constraint,
			trainable,
			name,
			**kwargs)

		self.g_initializer = g_initializer
		self.g_regularizer = g_regularizer
		self.g_constraint = g_constraint

	def build(self, input_shape):
		self.g = self.add_variable(
			name='g',
			shape=(self.units,),
			initializer=self.g_initializer,
			regularizer=self.g_regularizer,
			constraint=self.g_constraint,
			trainable=True,
			dtype=self.dtype)
		super().build(input_shape)
		self.kernel = tf.nn.l2_normalize(self.kernel, axis=0) * self.g


class WeightNormConv2D(tf.layers.Conv2D):
	def __init__(
			self,
			filters,
			kernel_size,
			strides=(1, 1),
			padding='same',
			data_format='channels_last',
			dilation_rate=(1, 1),
			activation=None,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=tf.zeros_initializer(),
			g_initializer=tf.ones_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			g_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			g_constraint=None,
			trainable=True,
			name=None,
			**kwargs):
		super().__init__(
			filters,
			kernel_size,
			strides,
			padding,
			data_format,
			dilation_rate,
			activation,
			use_bias,
			kernel_initializer,
			bias_initializer,
			kernel_regularizer,
			bias_regularizer,
			activity_regularizer,
			kernel_constraint,
			bias_constraint,
			trainable,
			name,
			**kwargs)

		self.g_initializer = g_initializer
		self.g_regularizer = g_regularizer
		self.g_constraint = g_constraint

	def build(self, input_shape):
		self.g = self.add_variable(
			name='g',
			shape=(self.filters,),
			initializer=self.g_initializer,
			regularizer=self.g_regularizer,
			constraint=self.g_constraint,
			trainable=True,
			dtype=self.dtype)
		super().build(input_shape)
		self.kernel = tf.nn.l2_normalize(self.kernel, axis=[0, 1, 2]) * self.g


class Upconv2D(tf.layers.Conv2D):
	def __init__(
			self,
			filters,
			kernel_size,
			resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			up_strides=(2, 2),
			conv_strides=(1, 1),
			padding='same',
			data_format='channels_last',
			dilation_rate=(1, 1),
			activation=None,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=tf.zeros_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			trainable=True,
			name=None,
			**kwargs):
		if not isinstance(conv_strides, tuple):
			conv_strides = (conv_strides, conv_strides)
		if not isinstance(up_strides, tuple):
			up_strides = (up_strides, up_strides)

		super().__init__(
			filters,
			kernel_size,
			conv_strides,
			padding,
			data_format,
			dilation_rate,
			activation,
			use_bias,
			kernel_initializer,
			bias_initializer,
			kernel_regularizer,
			bias_regularizer,
			activity_regularizer,
			kernel_constraint,
			bias_constraint,
			trainable,
			name,
			**kwargs)

		self.resize_method = resize_method
		self.up_strides = up_strides

	def build(self, input_shape):
		self.new_h = input_shape[1] * (self.up_strides[0] // self.strides[0])
		self.new_w = input_shape[2] * (self.up_strides[1] // self.strides[1])
		input_shape = tf.TensorShape([input_shape[0], self.new_h, self.new_w, input_shape[3]])

		super().build(input_shape)

	def call(self, inputs):
		# factor = self.up_strides[0]
		# output = inputs
		# output = tf.concat([output]*(factor**2), axis=1)
		# output = tf.transpose(output, [0, 2, 3, 1])
		# output = tf.depth_to_space(output, factor)
		# output = tf.transpose(output, [0, 3, 1, 2])
		# output = super().call(output)
		# return output
		#
		conv_in = inputs
		if self.new_h.value > 1 or self.new_w.value > 1:
			conv_in = tf.image.resize_images(
				conv_in, [self.new_h, self.new_w], method=self.resize_method, )
		conv_out = super().call(conv_in)
		return conv_out


class SpectralUpconv2D(tf.layers.Conv2D):
	def __init__(
			self,
			filters,
			kernel_size,
			resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			up_strides=(2, 2),
			conv_strides=(1, 1),
			padding='same',
			data_format='channels_last',
			dilation_rate=(1, 1),
			activation=None,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=tf.zeros_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			trainable=True,
			name=None,
			**kwargs):
		if not isinstance(conv_strides, tuple):
			conv_strides = (conv_strides, conv_strides)
		if not isinstance(up_strides, tuple):
			up_strides = (up_strides, up_strides)

		super().__init__(
			filters,
			kernel_size,
			conv_strides,
			padding,
			data_format,
			dilation_rate,
			activation,
			use_bias,
			kernel_initializer,
			bias_initializer,
			kernel_regularizer,
			bias_regularizer,
			activity_regularizer,
			kernel_constraint,
			bias_constraint,
			trainable,
			name,
			**kwargs)

		self.resize_method = resize_method
		self.up_strides = up_strides

	def build(self, input_shape):
		self.new_h = input_shape[1] * (self.up_strides[0] // self.strides[0])
		self.new_w = input_shape[2] * (self.up_strides[1] // self.strides[1])
		input_shape = tf.TensorShape([input_shape[0], self.new_h, self.new_w, input_shape[3]])

		super().build(input_shape)

	def call(self, inputs):
		# factor = self.up_strides[0]
		# output = inputs
		# output = tf.concat([output]*(factor**2), axis=1)
		# output = tf.transpose(output, [0, 2, 3, 1])
		# output = tf.depth_to_space(output, factor)
		# output = tf.transpose(output, [0, 3, 1, 2])
		# output = super().call(output)
		# return output
		#
		conv_in = inputs
		if self.new_h.value > 1 or self.new_w.value > 1:
			conv_in = tf.image.resize_images(
				conv_in, [self.new_h, self.new_w], method=self.resize_method)
		conv_out = super().call(conv_in)
		return conv_out


class WeightNormUpconv2D(WeightNormConv2D):
	def __init__(
			self,
			filters,
			kernel_size,
			resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
			up_strides=(2, 2),
			conv_strides=(1, 1),
			padding='valid',
			data_format='channels_last',
			dilation_rate=(1, 1),
			activation=None,
			use_bias=True,
			kernel_initializer=None,
			bias_initializer=tf.zeros_initializer(),
			g_initializer=tf.ones_initializer(),
			kernel_regularizer=None,
			bias_regularizer=None,
			g_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			g_constraint=None,
			trainable=True,
			name=None,
			**kwargs):
		super().__init__(
			filters,
			kernel_size,
			conv_strides,
			padding,
			data_format,
			dilation_rate,
			activation,
			use_bias,
			kernel_initializer,
			bias_initializer,
			g_initializer,
			kernel_regularizer,
			bias_regularizer,
			g_regularizer,
			activity_regularizer,
			kernel_constraint,
			bias_constraint,
			g_constraint,
			trainable,
			name,
			**kwargs)

		self.resize_method = resize_method
		self.up_strides = up_strides

	def build(self, input_shape):
		self.new_h = input_shape[1] * (self.up_strides[0] * self.strides[0])
		self.new_w = input_shape[2] * (self.up_strides[1] * self.strides[1])
		input_shape = tf.TensorShape([input_shape[0], self.new_h, self.new_w, input_shape[3]])

		super().build(input_shape)

	def call(self, inputs):
		conv_in = inputs
		if self.new_h.value > 1 or self.new_w.value > 1:
			conv_in = tf.image.resize_images(
				conv_in, [self.new_h, self.new_w], method=self.resize_method)
		conv_out = super().call(conv_in)
		return conv_out


def weight_norm_dense(
		inputs,
		units,
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		g_initializer=tf.ones_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		g_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		g_constraint=None,
		trainable=True,
		name=None,
		**kwargs):
	weight_norm_dense_layer = WeightNormDense(
		units,
		activation,
		use_bias,
		kernel_initializer,
		bias_initializer,
		g_initializer,
		kernel_regularizer,
		bias_regularizer,
		g_regularizer,
		activity_regularizer,
		kernel_constraint,
		bias_constraint,
		g_constraint,
		trainable,
		name,
		**kwargs)

	out = weight_norm_dense_layer.apply(inputs)
	return out


def weight_norm_conv2d(
		inputs,
		filters,
		kernel_size,
		strides=(1, 1),
		padding='same',
		data_format='channels_last',
		dilation_rate=(1, 1),
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		g_initializer=tf.ones_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		g_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		g_constraint=None,
		trainable=True,
		name=None,
		**kwargs):
	weight_norm_conv2d_layer = WeightNormConv2D(
		filters,
		kernel_size,
		strides,
		padding,
		data_format,
		dilation_rate,
		activation,
		use_bias,
		kernel_initializer,
		bias_initializer,
		g_initializer,
		kernel_regularizer,
		bias_regularizer,
		g_regularizer,
		activity_regularizer,
		kernel_constraint,
		bias_constraint,
		g_constraint,
		trainable,
		name,
		**kwargs)

	out = weight_norm_conv2d_layer.apply(inputs)
	return out


def weight_norm_upconv2d(
		inputs,
		filters,
		kernel_size,
		resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
		up_strides=(2, 2),
		conv_strides=(1, 1),
		padding='valid',
		data_format='channels_last',
		dilation_rate=(1, 1),
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		g_initializer=tf.ones_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		g_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		g_constraint=None,
		trainable=True,
		name=None,
		**kwargs):
	upconv2d_layer = WeightNormUpconv2D(
		filters,
		kernel_size,
		resize_method,
		up_strides,
		conv_strides,
		padding,
		data_format,
		dilation_rate,
		activation,
		use_bias,
		kernel_initializer,
		bias_initializer,
		g_initializer,
		kernel_regularizer,
		bias_regularizer,
		g_regularizer,
		activity_regularizer,
		kernel_constraint,
		bias_constraint,
		g_constraint,
		trainable,
		name,
		**kwargs)

	out = upconv2d_layer.apply(inputs)
	return out


def upconv2d(
		inputs,
		filters,
		kernel_size,
		resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
		up_strides=(2, 2),
		conv_strides=(1, 1),
		padding='same',
		data_format='channels_last',
		dilation_rate=(1, 1),
		activation=None,
		use_bias=True,
		kernel_initializer=None,
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		kernel_constraint=None,
		bias_constraint=None,
		trainable=True,
		name=None,
		**kwargs):
	upconv2d_layer = Upconv2D(
		filters,
		kernel_size,
		resize_method,
		up_strides,
		conv_strides,
		padding,
		data_format,
		dilation_rate,
		activation,
		use_bias,
		kernel_initializer,
		bias_initializer,
		kernel_regularizer,
		bias_regularizer,
		activity_regularizer,
		kernel_constraint,
		bias_constraint,
		trainable,
		name,
		**kwargs)

	out = upconv2d_layer.apply(inputs)
	return out


def coord_conv_transform2d(inputs):
	layer = CoordConvTransform2D()
	outputs = layer.apply(inputs)
	return outputs


def spectral_norm(w, iteration=1):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u
	v_hat = None
	for i in range(iteration):
		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = tf.nn.l2_normalize(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = tf.nn.l2_normalize(u_)

	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = w / sigma
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm


def spectral_conv2d(inputs, filters, kernel_size, strides=2, padding='same',
										kernel_regularizer=None, kernel_initializer=None, use_bias=True, activation=None):
	with tf.variable_scope(name_or_scope=None, default_name='spectral_conv2d'):
		w = tf.get_variable(
			"kernel", shape=[kernel_size, kernel_size, inputs.get_shape()[-1], filters], regularizer=kernel_regularizer,
			initializer=kernel_initializer)
		x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w), strides=[1, strides, strides, 1], padding=padding.upper())
		if use_bias:
			b = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
			x = tf.nn.bias_add(x, b)
		if activation is not None:
			x = activation(x)
		return x


def spectral_dense(inputs, units):
	with tf.variable_scope(name_or_scope=None, default_name='spectral_dense'):
		w = tf.get_variable(
			"weights", shape=[inputs.get_shape()[-1], units]
		)
		w = spectral_norm(w)
		x = tf.matmul(inputs, w)
		b = tf.get_variable(
			"bias", shape=[units]
		)
		x = tf.nn.bias_add(x, b)
		return x


def spectral_conv2d_transpose(inputs, filters, kernel_size, strides=2, padding='same',
										kernel_regularizer=None, kernel_initializer=None, use_bias=True, activation=None):
	with tf.variable_scope(name_or_scope=None, default_name='spectral_conv2d_transpose'):
		x_shape = inputs.get_shape().as_list()
		w = tf.get_variable(
			"kernel", shape=[kernel_size, kernel_size, filters, inputs.get_shape()[-1]],
			regularizer=kernel_regularizer,
			initializer=kernel_initializer)
		output_shape = tf.stack([tf.shape(inputs)[0], x_shape[1] * strides, x_shape[2] * strides, filters])

		x = tf.nn.conv2d_transpose(
			inputs, filter=spectral_norm(w),
			output_shape=output_shape,
			strides=[1, strides, strides, 1],
			padding=padding.upper()
		)
		if use_bias:
			b = tf.get_variable("bias", [filters], initializer=tf.constant_initializer(0.0))
			x = tf.nn.bias_add(x, b)
		if activation is not None:
			x = activation(x)
		return x


def res_block(
		inputs, filters=128, kernel_size=3,  sampling=None, is_training=None,
		batch_normalization=True, activation=tf.nn.relu, trainable_sortcut=True):
	assert not batch_normalization or is_training is not None
	with tf.variable_scope(name_or_scope=None, default_name='res_block'):
		x = inputs

		if sampling == 'up':
			x = tf.image.resize_images(
				x,
				[x.get_shape()[1] * 2, x.get_shape()[2] * 2],
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
		if batch_normalization:
			x = tf.layers.batch_normalization(x, training=is_training, momentum=0.9)
		x = activation(x)
		x = spectral_conv2d(x, filters=filters, kernel_size=kernel_size, strides=1, padding='same')

		if sampling == 'down':
			x = tf.layers.average_pooling2d(
				x,
				pool_size=2,
				strides=2
			)
		if trainable_sortcut:
			inputs = spectral_conv2d(inputs, filters=filters, kernel_size=1, strides=1, padding='same')

		if sampling == 'up':
			inputs = tf.image.resize_images(
				inputs,
				[inputs.get_shape()[1] * 2, inputs.get_shape()[2] * 2],
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
		elif sampling == 'down':
			inputs = tf.layers.average_pooling2d(
				inputs,
				pool_size=2,
				strides=2
			)
		x = inputs + x
		return x


def res_block_original(
		inputs, filters=256, kernel_size=3,  sampling=None, is_training=None,
		batch_normalization=True, activation=tf.nn.relu, trainable_sortcut=True):
	assert not batch_normalization or is_training is not None
	with tf.variable_scope(name_or_scope=None, default_name='res_block'):
		x = inputs
		if batch_normalization:
			x = tf.layers.batch_normalization(x, training=is_training, momentum=0.9)
		x = activation(x)
		x = spectral_conv2d(x, filters=filters, kernel_size=kernel_size, strides=1, padding='same')

		if sampling == 'up':
			x = tf.image.resize_images(
				x,
				[x.get_shape()[1] * 2, x.get_shape()[2] * 2],
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
		if batch_normalization:
			x = tf.layers.batch_normalization(x, training=is_training, momentum=0.9)
		x = activation(x)
		x = spectral_conv2d(x, filters=filters, kernel_size=kernel_size, strides=1, padding='same')

		if sampling == 'down':
			x = tf.layers.average_pooling2d(
				x,
				pool_size=2,
				strides=2
			)
		if trainable_sortcut:
			inputs = spectral_conv2d(inputs, filters=filters, kernel_size=1, strides=1, padding='same')

		if sampling == 'up':
			inputs = tf.image.resize_images(
				inputs,
				[inputs.get_shape()[1] * 2, inputs.get_shape()[2] * 2],
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
		elif sampling == 'down':
			inputs = tf.layers.average_pooling2d(
				inputs,
				pool_size=2,
				strides=2
			)
		x = inputs + x
		return x

