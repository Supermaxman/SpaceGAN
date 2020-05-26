import numpy as np


def create_noise(random_size):
	def sample(batch_size):
		return np.random.standard_normal(size=(batch_size, random_size))
		# return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, random_size))
	return sample

