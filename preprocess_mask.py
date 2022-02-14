import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
import os
import numpy as np

import numpy as np
import cv2
from scipy.ndimage import label
#%%


#%% scratch
mask = np.load(r'G:\OF\mask\57F1-6-2-20-OF1009animal_mask.npy')
frame = np.load(r'G:\OF\img\57F1-6-2-20-OF1009.npy')
plt.imshow(mask)
plt.show()
labeled_array, num_features = label(mask)

denoised = labeled_array==1
B = np.argwhere(denoised)
(ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
y_mid = int((ystart + ystop)/2)
x_mid = int((xstart + xstop)/2)
y_start = y_mid-100
y_end = y_mid +100
if y_mid - 100 < 0:
	y_start = 0
	y_end = 200
if y_mid + 100 > 1080:
	y_end = 1080
	y_start = 880
Atrim = denoised[y_start:y_end, x_mid-100:x_mid+100]
Amask = np.repeat(Atrim[:, :, np.newaxis], 3, axis=2)

Cframe = frame[y_start:y_end, x_mid - 100:x_mid + 100,:] ** Amask
plt.imshow(Cframe)
plt.show()


# find bounding box:
#https://stackoverflow.com/questions/4808221/is-there-a-bounding-box-function-slice-with-non-zero-values-for-a-ndarray-in


#%%


import csv


data = []


bad_ones = ['79F0-7-1-20-OF_5363', '79F0-7-1-20-OF_5362'
			'57F1-6-2-20-OF_108', '57F1-6-2-20-OF_109', '57F1-6-2-20-OF_130', '57F1-6-2-20-OF_142', '57F1-6-2-20-OF_143', '57F1-6-2-20-OF_154', '57F1-6-2-20-OF_163','57F1-6-2-20-OF_165'
			'57F1-6-2-20-OF_484', '57F1-6-2-20-OF_485', '57F1-6-2-20-OF_506', '57F1-6-2-20-OF_788', '57F1-6-2-20-OF_816']

offset = 150
for file in os.listdir(r'G:\OF\mask'):
	mask = np.load(r'G:\OF\mask\\' + file)

	file_name = file[:-4]
	flow = np.load(r'G:\OF\flow\\' + file_name[:-11] + 'flow.npy')
	frame = np.load(r'G:\OF\img\\' + file_name[:-11] + '.npy')
	print(np.shape(frame))
	if file_name not in bad_ones:

		labeled_array, num_features = label(mask)
		denoised = labeled_array==1
		B = np.argwhere(denoised)
		(ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
		y_mid = int((ystart + ystop) / 2)
		x_mid = int((xstart + xstop) / 2)
		y_start = y_mid - offset
		y_end = y_mid + offset
		if y_mid - offset < 0:
			y_start = 0
			y_end = offset*2
		if y_mid + offset > 1080:
			y_end = 1080
			y_start = 1080 - offset*2
		global_location = [file_name[:-11], ystart, ystop, xstart, xstop, y_mid, x_mid]
		data.append(global_location)
		Atrim = denoised[y_start:y_end, x_mid - offset:x_mid + offset]
		Amask = np.repeat(Atrim[:, :, np.newaxis], 3, axis=2)

		Btrim = flow[y_start:y_end, x_mid - offset:x_mid + offset,:] ** Amask
		Cframe = frame[y_start:y_end, x_mid - offset:x_mid + offset,:] ** Amask


		mypath_flow = 'G:\OF\\flow_denoised\\' + file_name + '_dn_flow.npy'
		mypath_npy = 'G:\OF\mask_denoised\\' + file_name + '_dn_mask.npy'
		mypath_npy_img = 'G:\OF\denoised_png\\' + file_name + '_dn.png'

		ax0 = plt.subplot(1, 3, 1)
		ax0.set_title("frame")
		ax0.imshow(Cframe)


		ax = plt.subplot(1, 3, 2)
		ax.set_title("animal mask")
		ax.imshow(Atrim)

		ax2 = plt.subplot(1, 3, 3)
		ax2.set_title("flow")
		ax2.imshow(Btrim)
		plt.tight_layout()

		plt.savefig(mypath_npy_img)
		np.save(mypath_npy, Atrim)
		np.save(mypath_flow, Btrim)
		print("save", file_name)

header = ['name', 'ystart', 'ystop', 'xstart', 'xstop', 'y_mid', 'x_mid']
# opening the csv file in 'a+' mode
file = open('G:\OF\mouse_global_loc.csv', 'a+', newline='')

# writing the data into the file
with file:
	write = csv.writer(file)
	write.writerow(header)
	write.writerows(data)