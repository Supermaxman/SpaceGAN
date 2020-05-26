
import os
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)


if __name__ == '__main__':
	image_folder = 'D:/Data/nasa/observations'
	statistics_path = 'D:/Data/nasa/observations_stats.txt'
	removed_files = []

	stats = {}
	rgb = []
	files = []
	with open(statistics_path, 'r') as f:
		for line in f:
			line = line.strip()
			if line:
				row = line.split(',')
				file_name = row[0]
				file_rgb = list(map(float, row[1:]))
				f_rgb = np.array(file_rgb, dtype=np.float32)
				files.append(file_name)
				rgb.append(f_rgb)

	rgb = np.array(rgb)
	files = np.array(files)

	rgb_mean = np.mean(rgb, axis=0)
	rgb_std = np.std(rgb, axis=0)

	rgb_color = ['r', 'g', 'b']

	removed_files = []
	# for color_data, color_code in zip(rgb.T, rgb_color):
	# 	color_mean = np.mean(color_data)
	# 	color_max = np.percentile(color_data, 98)
	# 	sns.distplot(color_data, kde=False, color=color_code)
	# 	plt.axvline(color_mean, label='mean', color=color_code, alpha=0.6)
	# 	plt.axvline(color_max, label=f'98%', color=color_code, alpha=0.5)
	# 	plt.legend()
	# 	plt.show()
	#
	# 	unusual_mask = color_data > color_max
	# 	unusual_files = [(f_name, color_code) for f_name in files[unusual_mask]]
	# 	removed_files.extend(unusual_files)


	total_data = np.mean(rgb, axis=1)
	total_mean = np.mean(total_data, axis=0)
	color_min = np.percentile(total_data, 1)
	sns.distplot(total_data, kde=False, color='orange')
	plt.axvline(total_mean, label='mean', color='orange', alpha=0.6)
	plt.axvline(color_min, label=f'99%', color='orange', alpha=0.5)
	plt.legend()
	plt.show()

	unusual_mask = total_data < color_min
	unusual_files = [(f_name, 'b') for f_name in files[unusual_mask]]
	removed_files.extend(unusual_files)

	r_set = set()
	for file_name, color_code in tqdm(removed_files):
		file_path = os.path.join(image_folder, file_name)
		if os.path.exists(file_path):
			r_set.add(file_name)
			print(f'Deleting {file_name} due to {color_code} statistics...')
			# os.remove(file_path)

	print(len(r_set))
	print(len(files))
