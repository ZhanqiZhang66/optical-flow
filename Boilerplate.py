import kmeans1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pandas import *
import numpy as np
from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
from matplotlib.pyplot import plot
#%%
video_idx = r"D:\79F0-7-1-20-OF"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
output_data_name = video_idx + "movement_threshold (1).csv"
hand_label_csv = video_idx +  "_reFormatted.xls"
N = 20

# xls = ExcelFile(hand_label_csv)
# df = xls.parse(xls.sheet_names[0])
# my_dict = df.to_dict()
data = pd.read_csv(output_data_name)
#%%
x = data['frames']
y1 = data['percentage']
#%% get ride of 0.2 Hz noise
y2 = list(y1)
x1 = list(x)
for idx, y in enumerate(y1):
    if x[idx] % 30 == 26:
        y2.remove(y)
        x1.remove(x[idx])
#%%
plot(x,y1)
plt.show()
#%%
plot(x1,y2)
plt.show()
#%%
from scipy.signal import argrelextrema
y2 = np.array(y2)
x1 = np.array(x1)
mi, ma = argrelextrema(y2, np.less)[0], argrelextrema(y2, np.greater)[0]
print("Minima:", x1[mi])
print("Maxima:", x1[ma])
#%%
plot(x1[:mi[0]+1], y2[:mi[0]+1], 'r',
     x1[mi[0]:mi[1]+1], y2[mi[0]:mi[1]+1], 'g',
     x1[mi[1]:], y2[mi[1]:], 'b',
     x1[ma], y2[ma], 'go',
     x1[mi], y2[mi], 'ro')

plt.show()