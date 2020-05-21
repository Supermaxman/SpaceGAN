
import os
from tqdm import tqdm
import tensorflow as tf
import shutil
import numpy as np
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e20
import random
import io
from multiprocessing import Pool


def _int64_feature(value):
	if not isinstance(value, list):
		value = [value]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


image_folder = 'D:/Data/nasa/observations'
image_list = 'D:/Data/nasa/observations_factors.txt'
output_folder = 'C:/Users/maxwe/Data'
num_interleave = 6


def save_images(args):
	i, files, dup_lookup = args
	output_file = os.path.join(output_folder, f'data-{i}.tfrecords')
	print(f'{output_file}: {len(files)}')
	with tf.io.TFRecordWriter(output_file) as writer:
		if i == 0:
			file_iter = tqdm(files, total=len(files))
		else:
			file_iter = files
		for file_name in file_iter:
			file_path = os.path.join(image_folder, os.path.splitext(os.path.basename(file_name))[0]) + '.tif'
			image = Image.open(file_path)
			image.thumbnail(image.size)
			bytes = io.BytesIO()
			image.save(bytes, "PNG")
			bytes = bytes.getvalue()
			y, x, d = dup_lookup[file_name]
			example = tf.train.Example(
				features=tf.train.Features(
					feature={
						'filename': _bytes_feature(tf.compat.as_bytes(os.path.splitext(os.path.basename(file_name))[0])),
						'bytes': _bytes_feature(tf.compat.as_bytes(bytes)),
						'duplicates': _int64_feature(d),
						'y': _int64_feature(y),
						'x': _int64_feature(x)
					}
				)
			)
			writer.write(example.SerializeToString())


if __name__ == '__main__':
	random.seed(0)
	files = []
	for i in range(num_interleave):
		files.append([])
	dup_lookup = {}
	lines = []
	with open(image_list, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			if line:
				lines.append(line)
	random.shuffle(lines)
	i = 0
	for line in lines:
		files[i].append(line[0])
		i += 1
		i %= num_interleave
		dup_lookup[line[0]] = (int(line[1]), int(line[2]), int(line[3]))

	print('Converting png files to tfrecord...')
	with Pool(processes=num_interleave) as p:
		p.map(save_images, [(i, files[i], dup_lookup) for i in range(num_interleave)])



