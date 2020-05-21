# from https://github.com/JonathanRaiman/tensorflow-infogan/blob/master/infogan/categorical_grid_plots.py

import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from PIL import Image
import os

from noise_utils import (
	create_noise
)


def create_image_strip(images, zoom=1, gutter=5):
	num_images, image_height, image_width, channels = images.shape

	if channels == 1:
		images = images.reshape(num_images, image_height, image_width)

	# add a gutter between images
	effective_collage_width = num_images * (image_width + gutter) - gutter
	effective_collage_height = image_height + gutter

	# use white as background
	start_color = (255, 255, 255)

	collage = Image.new('RGB', (effective_collage_width, effective_collage_height), start_color)
	offset = 0
	for image_idx in range(num_images):
		to_paste = Image.fromarray(
			(((images[image_idx] * 0.5) + 0.5) * 255).astype(np.uint8)
		)
		collage.paste(
			to_paste,
			box=(offset, 0, offset + image_width, image_height)
		)
		offset += image_width + gutter

	if zoom != 1:
		collage = collage.resize(
			(
				int(collage.size[0] * zoom),
				int(collage.size[1] * zoom)
			),
			Image.NEAREST
		)
	return np.array(collage)


class CategoricalPlotter(object):
	def __init__(self,
							 journalist,
							 log_dir,
							 random_size,
							 generate,
							 row_size=10,
							 zoom=1,
							 gutter=3):
		self._journalist = journalist
		self._gutter = gutter
		self.random_size = random_size
		self._generate = generate
		self._row_size = row_size
		self._zoom = zoom
		self._log_dir = log_dir
		if not os.path.exists(self._log_dir):
			os.makedirs(self._log_dir)
		self._z_log_dir = os.path.join(self._log_dir, 'samples')
		if not os.path.exists(self._z_log_dir):
			os.makedirs(self._z_log_dir)
		self.zc_vector = create_noise(self.random_size)(row_size)
		self._placeholders = {}
		self._image_summaries = {}

	def _get_image_summary(self, images):
		img_summaries = []
		for image, name in images:
			with io.BytesIO() as buffer:
				plt.imsave(buffer, image, format='png')
				img_sum = tf.Summary.Image(
					encoded_image_string=buffer.getvalue(),
					height=image.shape[0],
					width=image.shape[1])
			smm = tf.Summary.Value(tag=name, image=img_sum)
			img_summaries.append(smm)

		summary = tf.Summary(value=img_summaries)
		return summary

	def _save_images(self, images, folder, iteration=None):
		if iteration is None:
			filename = 'images.png'
		else:
			filename = 'images-{}.png'.format(iteration)
		filepath = os.path.join(folder, filename)

		image = np.vstack([image for image, _ in images])

		plt.imsave(filepath, image, format='png')

	def _add_image_summary(self, images, iteration=None):

		summary = self._get_image_summary(images)

		self._journalist.add_summary(summary, iteration)
		self._journalist.flush()

	def _create_images(self, session):
		images = []
		name = 'z_samples'
		images.append(
			(
				create_image_strip(
					self._generate(session, self.zc_vector),
					zoom=self._zoom, gutter=self._gutter
				),
				name
			)
		)
		return images

	def save_variations(self, session, iteration=None):
		images = self._create_images(
			session
		)
		self._save_images(images, self._z_log_dir, iteration)
		self._add_image_summary(images, iteration=iteration)

	def generate_images(self, session, iteration=None):
		self.save_variations(
			session, iteration=iteration
		)
