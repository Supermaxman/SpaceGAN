import numpy as np


def create_noise(random_size):
	def sample(batch_size):
		return np.random.standard_normal(size=(batch_size, random_size))
	return sample

