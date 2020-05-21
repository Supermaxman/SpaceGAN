import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from psd_tools import PSDImage
import shutil
from tqdm import tqdm
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e20


def compute_factor(sizes, min_size):
	size_factors = sizes // min_size
	return size_factors


def crop(im, height, width):
	imgwidth, imgheight = im.size
	rows = np.int(imgheight/height)
	cols = np.int(imgwidth/width)
	for i in range(rows):
		for j in range(cols):
			# print (i,j)
			box = (j*width, i*height, (j+1)*width, (i+1)*height)
			yield im.crop(box)


def recursive_split(image_folder, image_file):
	file_path = os.path.join(image_folder, image_file)
	file_size = os.stat(file_path).st_size
	# if image is larger than 10 MB then we split image up:
	if file_size > 1e+7:
		im = Image.open(file_path)
		print(f'Splitting {file_path} ({file_size / 1000000:.2f} MB)')
		imgwidth, imgheight = im.size
		height = np.int(imgheight / 2)
		width = np.int(imgwidth / 2)
		for k, piece in enumerate(crop(im, height, width)):
			# print k
			# print piece
			img = Image.new('RGB', (width, height), 255)
			# print img
			img.paste(piece)
			k_filename = image_file.replace('.tif', '') + f'_{k + 1}.tif'
			k_image_file = os.path.join(image_folder, k_filename)
			img.save(k_image_file, 'TIFF')
			recursive_split(image_folder, k_image_file)
		os.remove(file_path)


if __name__ == '__main__':
	image_folder = 'D:/Data/nasa/observations'
	output_image_folder = 'D:/Data/nasa/data'
	info_path = 'D:/Data/nasa/observations_info.txt'
	factor_path = 'D:/Data/nasa/observations_factors.txt'
	file_list_path = 'D:/Data/nasa/observations_files.txt'
	min_x = 512
	min_y = 512
	min_prob_unique = 0.5
	with open(info_path, 'w') as f:
		files = list(os.listdir(image_folder))
		for file_name in tqdm(files):
			file_path = os.path.join(image_folder, file_name)
			image = PIL.Image.open(file_path)
			x = image.width
			y = image.height
			c = len(image.getbands())
			# file_name, y, x, c
			f.write(f'{file_name},{y},{x},{c},{image.mode}\n')

	image_info = {}
	y_list = []
	x_list = []
	with open(info_path, 'r') as f:
		for line in f:
			file_name, y, x, c, mode = line.strip().split(',')
			file_path = os.path.join(image_folder, file_name)
			y = int(y)
			x = int(x)
			c = int(c)
			if y < min_y:
				print(f'Deleting {file_name} due to y size ({y})')
				os.remove(file_path)
				continue
			if x < min_x:
				print(f'Deleting {file_name} due to x size ({x})')
				os.remove(file_path)
				continue
			if mode == 'L':
				print(f'Deleting {file_name} due to black-and-white')
				os.remove(file_path)
				continue
			if mode == 'RGBX' or mode == 'RGBA':
				print(f'Converting {file_name} ({mode}) to RGB')
				image = Image.open(file_path)
				image.load()
				new_image = Image.new("RGB", image.size, (255, 255, 255))
				new_image.paste(image, mask=image.split()[3])
				new_image.save(os.path.join(image_folder, file_name), 'TIFF')
			image_info[file_name] = (y, x)

		print('Splitting up large images...')
		for image_file in image_info.keys():
			recursive_split(image_folder, image_file)
		# exit()
		factors = []
		data_size = 0
		weighted_data_size = 0
		file_list = []
		with open(factor_path, 'w') as f:
			for image_file, (y, x), in image_info.items():
				prob_unique = 1.0
				num_samples = 0
				# TODO figure out non-iterative or better equation
				# current calculation assumes we select min_x * min_y pixels uniformly across
				# x * y which is not strictly true since we are selecting a rectangle.
				# formally product_x=0^y(1-x*(min_x * min_y)/(x * y)) < 0.5
				# where we solve for y. Iterative approach seems simplest, but maybe closed form?
				# https://www.wolframalpha.com/input/?i=%28product_x%3D0%5Ey%281-x300%5E2%2F1000%5E2%29%29+%3C+0.5
				# uses gamma function so probably not...
				# we could compute a more accurate probability using sampling, or there may be a better
				# formula for this type of problem of overlapping rectangles.
				while True:
					prob_unique = prob_unique * (1.0 - num_samples * (min_x * min_y)/(x * y))
					if prob_unique < min_prob_unique:
						break
					num_samples += 1
				factor = num_samples
				data_size += 1
				weighted_data_size += factor
				factors.append(factor)
				f.write(f'{image_file},{y},{x},{factor}\n')
				for i in range(factor):
					file_list.append(image_file.replace('.tif', '.jpeg'))
		with open(file_list_path, 'w') as f:
			for file_name in file_list:
				f.write(f'{file_name}\n')
		print(np.min(factors))
		print(np.max(factors))
		print(np.mean(factors))
		print(np.median(factors))
		print(np.percentile(factors, 95))
		print(f'{data_size} images with {weighted_data_size} weighted images')


