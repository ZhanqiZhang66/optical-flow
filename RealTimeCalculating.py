import cv2 as cv
import numpy as np

video_idx = r"D:\OF videos for Victoria\79F0-7-1-20-OF"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
#stablized_dlc_data = video_idx + "DLC_resnet50_CP_dystoniaJan22shuffle1_976600.csv"

import csv
from itertools import zip_longest
import numpy as np
# Optical Flow
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture(stablized_video_name)  # stablized_video Name

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only

# need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

codec = 'MJPG'
fps =  33
fourcc = cv.VideoWriter_fourcc(*codec)
print("frame_width", cap.get(cv.CAP_PROP_FRAME_WIDTH))  # float
print("frame_height", cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # float
out = cv.VideoWriter(stablized_of_video_name, fourcc, fps, (1920, 1080), True)  # stablized_of_video_name

#%%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import csv
import random
import time

length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

count = 0
iCountMovement= np.zeros(length)
frames = np.zeros([length])
# all_angles, magnitudes = [], []
iThreshold = 0
# fieldnames = ["frames", "number of Movements"]
# output_data_name = video_idx + "movement_threshold.csv"
# with open(output_data_name, 'w', newline='') as csv_file:
#     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     csv_writer.writeheader()
mag = np.zeros(length)
ang = np.zeros(length)
Thresholds = np.zeros(length)
#%%
while (cap.isOpened()):
    # with open(output_data_name, 'a') as csv_file:
    #     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        ret, frame = cap.read()
        if ret == False:
            break
        #cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # magnitudes.append(magnitude)
        blue_box = magnitude[99:1001, 567:1473]  # the blue area
        flattened = magnitude.flatten()
        flattened_bluebox = blue_box.flatten()
        descending_mouse_mag = (np.sort(flattened_bluebox, axis=0)[::-1])
        mouse_box = descending_mouse_mag[:40860]
        background_area = (magnitude[:, 0:560]).flatten()
        background_area = np.append(background_area, (magnitude[:, 1489:1920]).flatten())
        iThreshold = np.mean(background_area)
        iThresholdstd = np.std(background_area)
        Thresholds[count] = iThreshold # 3 * iThresholdstd
        # if count == 0:
        #     background_area = (magnitude[:, 0:560]).flatten()
        #     background_area = np.append(background_area, (magnitude[:, 1489:1920]).flatten())
        #     iThreshold = np.mean(background_area)
        #     iThresholdstd = np.std(background_area)
        #iThreshold = np.max(descending_mouse_mag[40860:])
        # iCountMovement[count] = len([i for i in flattened if i > iThreshold + 3* iThresholdstd])
        iCountMovement[count] = len([i for i in flattened_bluebox if i > iThreshold + 3 * iThresholdstd])
        iMovement = np.mean(mouse_box)
        frames[count] = count
        mag[count] = iCountMovement[count] * iMovement
        # iCountMovement_plusSTD[count] = len([i for i in flattened if i > iThreshold + iThresholdstd])
        # iCountMovement_minusSTD[count] = len([i for i in flattened if i > iThreshold - iThresholdstd])
        # info = {
        #     "frames": count,
        #     "mag": iCountMovement[count] * iMovement
        # }
        # csv_writer.writerow(info)


    # angles = angle * 180 / np.pi
    # all_angles.append(angles)
    # # Sets image hue according to the optical flow direction
    # mask[..., 0] = angle * 180 / np.pi / 2  # angles angles # angles #
    # # Sets image value according to the optical flow magnitude (normalized)
    # mask[..., 2] = cv.normalize(magnitude, None, 0, 100, cv.NORM_MINMAX)  # #magnitude #
    # # Converts HSV to RGB (BGR) color representation

        count += 1
        break

        # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
    # out.write(rgb)
    # Updates previous frame
        prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
# The following frees up resources and closes all windows

cap.release()
cv.destroyAllWindows()
#%%
#
from pandas import DataFrame
df = DataFrame({'frames': frames, 'mag': mag, 'threshold': Thresholds})
xlsx_name = video_idx + "movement_threshold.xlsx"
df.to_excel(xlsx_name, sheet_name='sheet1', index=False)
