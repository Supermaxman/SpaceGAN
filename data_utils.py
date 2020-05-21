import numpy as np
import tensorflow as tf
import os
import random


def create_dataset(
		data_dir,
		batch_size,
		scale=(300, 300),
		random_flip = False,
		random_brightness=False,
		random_contrast=False):
	# random.shuffle(image_paths)
	# image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
	# dataset = tf.data.Dataset.from_tensor_slices((image_paths,))
	# TODO add shards for even faster reading
	num_interleave = 6
	file_paths = [os.path.join(data_dir, f'data-{i}.tfrecords') for i in range(num_interleave)]
	dataset = tf.data.TFRecordDataset(
		filenames=file_paths,
		num_parallel_reads=num_interleave
	)
	features = {
		'filename': tf.FixedLenFeature([], dtype=tf.string),
		'bytes': tf.FixedLenFeature([], dtype=tf.string),
		'duplicates': tf.FixedLenFeature([], dtype=tf.int64),
		'y': tf.FixedLenFeature([], dtype=tf.int64),
		'x': tf.FixedLenFeature([], dtype=tf.int64),
	}

	def crop_image(example):
		crop_shape = [scale[0], scale[1], 3]
		image = tf.image.random_crop(example['image'], size=crop_shape)
		image = tf.reshape(image, shape=crop_shape)
		# TODO random_flip, random_brightness, random_contrast
		# scale image dtype
		# [0, 255] -> [0, 1]
		image = tf.image.convert_image_dtype(image, tf.float32)
		# normalize image
		# [0, 1] -> [-0.5, 0.5] -> [-1, 1]
		image = 2 * (image - 0.5)
		return image

	def decode_image(example_proto):
		example = tf.parse_single_example(example_proto, features)
		image = tf.image.decode_image(example['bytes'], channels=3)
		del example['bytes']
		example['image'] = image
		return example

	# def duplicate_example(example):
	# 	count = example['duplicates']
	# 	return tf.data.Dataset.from_tensors(example).repeat(count)

	dataset = dataset.map(
		map_func=decode_image,
		num_parallel_calls=2
	)

	# dataset = dataset.flat_map(
	# 	map_func=duplicate_example
	# )

	dataset = dataset.shuffle(
		buffer_size=1000,
		reshuffle_each_iteration=True
	)

	dataset = dataset.repeat()

	dataset = dataset.map(
		map_func=crop_image,
		num_parallel_calls=2
	)

	dataset = dataset.batch(
		batch_size=batch_size
	)

	dataset = dataset.prefetch(
			buffer_size=1
	)

	return dataset


def load_path_factors(files_list):
	files = []
	with open(files_list, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			if line:
				files.append(int(line[3]))
	return files
