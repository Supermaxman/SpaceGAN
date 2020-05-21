import os
from tqdm import tqdm
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1e20
import shutil
import numpy as np


if __name__ == '__main__':
	image_folder = 'D:/Data/nasa/observations'
	output_image_folder = 'D:/Data/nasa/data'
	image_list = 'D:/Data/nasa/observations_files.txt'
	files = set()
	with open(image_list, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				files.add(line)
	print('Converting tif files to png')
	for file_name in tqdm(files, total=len(files)):
		file_path = os.path.join(image_folder, file_name.replace('.jpeg', '.tif'))
		output_file_path = os.path.join(output_image_folder, file_name.replace('.jpeg', '.png'))
		if not os.path.exists(output_file_path):
			image = Image.open(file_path)
			image.thumbnail(image.size)
			image.save(output_file_path, "PNG")
