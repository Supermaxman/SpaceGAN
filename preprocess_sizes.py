
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e20


if __name__ == '__main__':
	image_folder = 'D:/Data/nasa/observations'
	output_path = 'D:/Data/nasa/observations_info.txt'
	with open(output_path, 'w') as f:
		files = list(os.listdir(image_folder))
		for file_name in tqdm(files):
			file_path = os.path.join(image_folder, file_name)
			image = PIL.Image.open(file_path)
			x = image.width
			y = image.height
			c = len(image.getbands())
			# file_name, y, x, c
			f.write(f'{file_name},{y},{x},{c},{image.mode}\n')


