import tensorflow as tf
import os


def create_dataset(
		data_dir,
		batch_size,
		scale=(128, 128),
		crop=(512, 512)
):
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

	def crop_image(example_proto):
		example = tf.parse_single_example(example_proto, features)
		image = tf.image.decode_image(example['bytes'], channels=3)
		image = tf.reshape(image, shape=[example['y'], example['x'], 3])
		crop_shape = [crop[0], crop[1], 3]
		image = tf.image.random_crop(image, size=crop_shape)
		if crop[0] != scale[0] or crop[1] != scale[1]:
			image = tf.image.resize(image, size=scale, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		image = tf.image.random_flip_left_right(image)
		image = tf.image.random_flip_up_down(image)
		num_rotate = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
		image = tf.image.rot90(image, k=num_rotate)
		# scale image dtype
		# [0, 255] -> [0, 1]
		image = tf.image.convert_image_dtype(image, tf.float32)
		# normalize image
		# [0, 1] -> [-0.5, 0.5] -> [-1, 1]
		image = 2.0 * (image - 0.5)
		return image

	dataset = dataset.shuffle(
		buffer_size=1000,
		reshuffle_each_iteration=True
	)

	dataset = dataset.repeat()

	dataset = dataset.map(
		map_func=crop_image,
		num_parallel_calls=6
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
