import cv2 as cv
import numpy as np

video_idx = r"D:\79F0-7-1-20-OF_trim30"
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
iCountMovement = 0
frames = []
all_angles, magnitudes = [], []
descending_mouse_mag, except5percent_mouse_mag = [], []
fieldnames = ["frames", "number of Movements"]
output_data_name = video_idx + "movement_threshold.csv"
with open(output_data_name, 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

while (cap.isOpened()):
    with open(output_data_name, 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        ret, frame = cap.read()
        if ret == False:
            break
        # Opens a new window and displays the input frame
        cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(magnitude)
        iMouse_mag = magnitude[99:1001, 567:1473]
        flattened = iMouse_mag.flatten()
        descending_mouse_mag = (np.sort(flattened, axis=0)[::-1])

        iThreshold = np.mean(descending_mouse_mag[40860:])
        iCountMovement = len([i for i in flattened if i > iThreshold])
        info = {
            "frames": count,
            "number of Movements": iCountMovement,
        }
        csv_writer.writerow(info)


    # angles = angle * 180 / np.pi
    # all_angles.append(angles)
    # # Sets image hue according to the optical flow direction
    # mask[..., 0] = angle * 180 / np.pi / 2  # angles angles # angles #
    # # Sets image value according to the optical flow magnitude (normalized)
    # mask[..., 2] = cv.normalize(magnitude, None, 0, 100, cv.NORM_MINMAX)  # #magnitude #
    # # Converts HSV to RGB (BGR) color representation

        count += 1

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
# all_data = [magnitudes,all_angles]
# export_data = zip_longest(*all_data, fillvalue = '')
#
# output_data_name = video_idx + "numbers.csv"
# with open(output_data_name, 'w', encoding="ISO-8859-1", newline='') as myfile:
#     wr = csv.writer(myfile)
#     wr.writerow(("magnitudes", "angles"))
#     wr.writerows(export_data)
# myfile.close()
#%%
mouse_mag = [magnitudes[i][99:1001, 567:1473] for i in np.arange(len(magnitudes))]
flattened = [subarr.flatten() for subarr in mouse_mag]
# flattened.sort(axis= 1)[::-1]
descending_mouse_mag , except5percent_mouse_mag = [], []
for count , arr in enumerate(flattened):
    descending_mouse_mag.append(np.sort(arr,axis=0)[::-1])

threshold = [np.mean(x[40860:]) for x in descending_mouse_mag]
threshold = np.mean(threshold)
#%%
import matplotlib.pyplot as plt
countMovement = np.zeros(len(magnitudes))
for iframe in np.arange(len(magnitudes)):
    countMovement[iframe] = len([i for i in flattened[iframe] if i > threshold])
    #%%
plt.plot(np.arange(len(magnitudes)),countMovement,'ro')
plt.xlabel('frames')
plt.ylabel('Number of Movement greater than threshold')
plt.show()
# %%
# now count = maxcount
Lknee_mag, Rknee_mag, Lankle_mag, Rankle_mag, Ltoe_mag, Rtoe_mag = [], [], [], [], [], []
Lknee_angle, Rknee_angle, Lankle_angle, Rankle_angle, Ltoe_angle, Rtoe_angle = [], [], [], [], [], []

# (-1,2) is  3*3 = 9 pixels
# (-2,3) is  5*5 = 25 pixels
# (-3,4) is  7*7 = 49 pixels

for i in range(-3, 4):
    for j in range(-3, 4):
        Lknee_mag.append([magnitude[Lknee[count][0] + i][Lknee[count][1] + j]])
        Rknee_mag.append(magnitude[Rknee[count][0] + i][Rknee[count][1] + j])
        Lankle_mag.append(magnitude[Lankle[count][0] + i][Lankle[count][1] + j])
        Rankle_mag.append(magnitude[Rankle[count][0] + i][Rankle[count][1] + j])
        Ltoe_mag.append(magnitude[Ltoe[count][0] + i][Ltoe[count][1] + j])
        Rtoe_mag.append(magnitude[Rtoe[count][0] + i][Rtoe[count][1] + j])

        Lknee_angle.append(angles[Lknee[count][0] + i][Lknee[count][1] + j])
        Rknee_angle.append(angles[Rknee[count][0] + i][Rknee[count][1] + j])
        Lankle_angle.append(angles[Lankle[count][0] + i][Lankle[count][1] + j])
        Rankle_angle.append(angles[Rankle[count][0] + i][Rankle[count][1] + j])
        Ltoe_angle.append(angles[Ltoe[count][0] + i][Ltoe[count][1] + j])
        Rtoe_angle.append(angles[Rtoe[count][0] + i][Rtoe[count][1] + j])
Lknee_mag9.append(Lknee_mag)
Rknee_mag9.append(Rknee_mag)
Lankle_mag9.append(Lankle_mag)
Rankle_mag9.append(Rankle_mag)
Ltoe_mag9.append(Ltoe_mag)
Rtoe_mag9.append(Rtoe_mag)

Lknee_angle9.append(Lknee_angle)
Rknee_angle9.append(Rknee_angle)
Lankle_angle9.append(Lankle_angle)
Rankle_angle9.append(Rankle_angle)
Ltoe_angle9.append(Ltoe_angle)
Rtoe_angle9.append(Rtoe_angle)

all_data = [Lknee_mag9, Rknee_mag9, Lankle_mag9, Rankle_mag9, Ltoe_mag9, Rtoe_mag9,
            Lknee_angle9, Rknee_angle9, Lankle_angle9, Rankle_angle9, Ltoe_angle9, Rtoe_angle9]
export_data = zip_longest(*all_data, fillvalue='')

output_data_name = video_idx + "numbers.csv"
with open(output_data_name, 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(
    # wr.writerow(("Lknee_mag9", "Rknee_mag9", "Lankle_mag9", "Rankle_mag9", "Ltoe_mag9",
    #              "Rtoe_mag9", "Lknee_angle9", "Rknee_angle9", "Lankle_angle9", "Rankle_angle9", "Ltoe_angle9",
    #              "Rtoe_angle9"))
    wr.writerows(export_data)
myfile.close()

# %%
import matplotlib.pyplot as plt
import cv2
def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
for i in np.arange(1,count + 1):
    # img = cv2.cvtColor(frames[1][162-50: 162+51, 324-50:324+51,:], cv2.COLOR_BGR2RGB);
    img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB);
    # plt.imshow(img); plt.show()
    w, h = img.shape[1], img.shape[0]
    y, x = np.mgrid[0:h, 0:w]

    # tmp = np.reshape(magnitudes[0][162-50: 162+51, 324-50:324+51],(101,101))
    tmp = magnitudes[i-1]
    plt.imshow(tmp)
    # plt.show()
    #Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.jet)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)

    cb = ax.contourf(x, y, tmp, 15, cmap=mycmap)

    plt.colorbar(cb)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    figurename  = video_idx + '_frame' + str(i) +'_heatmap.png'
    fig1.savefig(figurename)
# %%
import matplotlib.pyplot as plt
import cv2

tmp = np.reshape(Rankle_angle9[0], (51, 51))
plt.imshow(tmp)
plt.show()

tmp = np.reshape(Rankle_mag9[0], (51, 51))
plt.imshow(tmp)
plt.show()
# %%
tmp = np.reshape(Rknee_mag9[0], (51, 51))
plt.imshow(tmp)
plt.show()

tmp = np.reshape(Rknee_angle9[0], (51, 51))
plt.imshow(tmp)
plt.show()

# %%
# Stop here
#################################################
# below plotting legs
#################################################
import math


def interpolate_pixels_along_line(x0, y0, x1, y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels


Lleg = interpolate_pixels_along_line(Lknee[243][0], Lknee[243][1], Lankle[243][0], Lankle[243][1])
Lfoot = interpolate_pixels_along_line(Lankle[243][0], Lankle[243][1], Ltoe[243][0], Ltoe[243][1])

Rleg = interpolate_pixels_along_line(Rknee[243][0], Rknee[243][1], Rankle[243][0], Rankle[243][1])
Rfoot = interpolate_pixels_along_line(Rankle[243][0], Rankle[243][1], Rtoe[243][0], Rtoe[243][1])
# %%
import numpy as np
import matplotlib.pyplot as plt

plt.imshow(magnitude)
plt.plot([i[0] for i in Lleg], [j[1] for j in Lleg], 'ro')
plt.plot([i[0] for i in Rleg], [j[1] for j in Rleg], 'bo')
plt.show()
plt.imshow(Lknee_mag)
plt.show()
# %%
#######################################################################
# Below are scratch
import csv
import numpy as np

with open('2DLC_resnet50_CP_dystoniaJan22shuffle1_976600.csv', newline='') as f:  # stablized_dlc_data
    reader = csv.reader(f)
    data = list(reader)
rows = data[3:]
for val, row in enumerate(rows):
    for val1, col in enumerate(row):
        rows[val][val1] = int(float(col))
print(rows)
# %%
Lknee = [x[1:3] for x in data[3:]]
Rknee = [x[4:6] for x in data[3:]]
Lankle = [x[7:9] for x in data[3:]]
Rankle = [x[10:12] for x in data[3:]]
Ltoe = [x[13:15] for x in data[3:]]
Rtoe = [x[16:18] for x in data[3:]]

# %%
import cv2

vidcap = cv2.VideoCapture(stablized_of_video_name)  # stablized_of_video_name
success, image = vidcap.read()
count = 0

Lknee_mag9, Rknee_mag9, Lankle_mag9, Rankle_mag9, Ltoe_mag9, Rtoe_mag9 = [], [], [], [], [], []
Lknee_angle9, Rknee_angle9, Lankle_angle9, Rankle_angle9, Ltoe_angle9, Rtoe_angle9 = [], [], [], [], [], []
while success:
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    magnitude = hsv[..., 2]
    angles = hsv[..., 0]
    Lknee_mag, Rknee_mag, Lankle_mag, Rankle_mag, Ltoe_mag, Rtoe_mag = [], [], [], [], [], []
    Lknee_angle, Rknee_angle, Lankle_angle, Rankle_angle, Ltoe_angle, Rtoe_angle = [], [], [], [], [], []
    # (-1,2) is the 9 pixels
    for i in range(-1, 2):
        for j in range(-1, 2):
            Lknee_mag.append([magnitude[Lknee[count][0] + i][Lknee[count][1] + j]])
            Rknee_mag.append(magnitude[Rknee[count][0] + i][Rknee[count][1] + j])
            Lankle_mag.append(magnitude[Lankle[count][0] + i][Lankle[count][1] + j])
            Rankle_mag.append(magnitude[Rankle[count][0] + i][Rankle[count][1] + j])
            Ltoe_mag.append(magnitude[Ltoe[count][0] + i][Ltoe[count][1] + j])
            Rtoe_mag.append(magnitude[Rtoe[count][0] + i][Rtoe[count][1] + j])

            Lknee_angle.append(angles[Lknee[count][0] + i][Lknee[count][1] + j])
            Rknee_angle.append(angles[Rknee[count][0] + i][Rknee[count][1] + j])
            Lankle_angle.append(angles[Lankle[count][0] + i][Lankle[count][1] + j])
            Rankle_angle.append(angles[Rankle[count][0] + i][Rankle[count][1] + j])
            Ltoe_angle.append(angles[Ltoe[count][0] + i][Ltoe[count][1] + j])
            Rtoe_angle.append(angles[Rtoe[count][0] + i][Rtoe[count][1] + j])
    Lknee_mag9.append(Lknee_mag)
    Rknee_mag9.append(Rknee_mag)
    Lankle_mag9.append(Lankle_mag)
    Rankle_mag9.append(Rankle_mag)
    Ltoe_mag9.append(Ltoe_mag)
    Rtoe_mag9.append(Rtoe_mag)

    Lknee_angle9.append(Lknee_angle)
    Rknee_angle9.append(Rknee_angle)
    Lankle_angle9.append(Lankle_angle)
    Rankle_angle9.append(Rankle_angle)
    Ltoe_angle9.append(Ltoe_angle)
    Rtoe_angle9.append(Rtoe_angle)
    # Lknee_mag.append(magnitude[Lknee[count][0]][Lknee[count][1]])
    # Rknee_mag.append(magnitude[Rknee[count][0]][Rknee[count][1]])
    # Lankle_mag.append(magnitude[Lankle[count][0]][Lankle[count][1]])
    # Rankle_mag.append(magnitude[Rankle[count][0]][Rankle[count][1]])
    # Ltoe_mag.append(magnitude[Ltoe[count][0]][Ltoe[count][1]])
    # Rtoe_mag.append(magnitude[Rtoe[count][0]][Rtoe[count][1]])
    #
    # Lknee_angle.append(angles[Lknee[count][0]][Lknee[count][1]])
    # Rknee_angle.append(angles[Rknee[count][0]][Rknee[count][1]])
    # Lankle_angle.append(angles[Lankle[count][0]][Lankle[count][1]])
    # Rankle_angle.append(angles[Rankle[count][0]][Rankle[count][1]])
    # Ltoe_angle.append(angles[Ltoe[count][0]][Ltoe[count][1]])
    # Rtoe_angle.append(angles[Rtoe[count][0]][Rtoe[count][1]])
    success, image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1

# now count = 150
Lknee_mag, Rknee_mag, Lankle_mag, Rankle_mag, Ltoe_mag, Rtoe_mag = [], [], [], [], [], []
Lknee_angle, Rknee_angle, Lankle_angle, Rankle_angle, Ltoe_angle, Rtoe_angle = [], [], [], [], [], []
for i in range(-1, 2):
    for j in range(-1, 2):
        Lknee_mag.append([magnitude[Lknee[count][0] + i][Lknee[count][1] + j]])
        Rknee_mag.append(magnitude[Rknee[count][0] + i][Rknee[count][1] + j])
        Lankle_mag.append(magnitude[Lankle[count][0] + i][Lankle[count][1] + j])
        Rankle_mag.append(magnitude[Rankle[count][0] + i][Rankle[count][1] + j])
        Ltoe_mag.append(magnitude[Ltoe[count][0] + i][Ltoe[count][1] + j])
        Rtoe_mag.append(magnitude[Rtoe[count][0] + i][Rtoe[count][1] + j])

        Lknee_angle.append(angles[Lknee[count][0] + i][Lknee[count][1] + j])
        Rknee_angle.append(angles[Rknee[count][0] + i][Rknee[count][1] + j])
        Lankle_angle.append(angles[Lankle[count][0] + i][Lankle[count][1] + j])
        Rankle_angle.append(angles[Rankle[count][0] + i][Rankle[count][1] + j])
        Ltoe_angle.append(angles[Ltoe[count][0] + i][Ltoe[count][1] + j])
        Rtoe_angle.append(angles[Rtoe[count][0] + i][Rtoe[count][1] + j])
Lknee_mag9.append(Lknee_mag)
Rknee_mag9.append(Rknee_mag)
Lankle_mag9.append(Lankle_mag)
Rankle_mag9.append(Rankle_mag)
Ltoe_mag9.append(Ltoe_mag)
Rtoe_mag9.append(Rtoe_mag)

Lknee_angle9.append(Lknee_angle)
Rknee_angle9.append(Rknee_angle)
Lankle_angle9.append(Lankle_angle)
Rankle_angle9.append(Rankle_angle)
Ltoe_angle9.append(Ltoe_angle)
Rtoe_angle9.append(Rtoe_angle)

