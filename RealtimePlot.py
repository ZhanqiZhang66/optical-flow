import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pandas import *
import numpy as np
#%%
from pandas import ExcelFile

video_idx = r"D:\output_data_sheet\79F0-7-1-20-OF"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
output_data_name = video_idx + "movement_threshold_updateBG.xlsx"#"movement_threshold (1).csv"
hand_label_csv = video_idx +  "_reFormatted1.xlsx"
N = 20

xls = ExcelFile(hand_label_csv)
df = xls.parse(xls.sheet_names[0])
my_dict = df.to_dict()
#%%
def animate(i):
    if output_data_name[-3:] == "csv":
        data = pd.read_csv(output_data_name)
    if output_data_name[-3:] == "lsx":
        data = pd.read_excel(output_data_name, sheet_name="sheet1")

    x = data['frames']
    y1 = data['mag']

    iCountMovementAvg = np.convolve(y1, np.ones((N,)) / N, mode='valid')
    y2 = np.zeros(y1.shape)
    offset = y1.shape[0] - iCountMovementAvg.shape[0]
    y2[offset:iCountMovementAvg.shape[0] + offset] = iCountMovementAvg
    plt.cla()

    plt.plot(x, y1, label='Raw data')
    plt.plot(x, y2, label='Moving average')
    #plt.fill_between(x, y2 - 1/2 *y1Error, y2 + 1/2 *y1Error, color="#3F5D7D")
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # axes = plt.gca()
    # #axes.set_ylim([300000, 700000])
#%%
with plt.style.context('ggplot'):

    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.draw()
    plt.show()

data = pd.read_csv(output_data_name)

df0 = data.query('frames >= 0 & frames < 80')
df1 = data.query('frames >= 80 & frames < 155')
df2 = data.query('frames >= 155 & frames < 197')
df3 = data.query('frames >= 197 & frames < 413')
df4 = data.query('frames >= 413 & frames < 865')
df5 = data.query('frames >= 865 & frames < 910')


# set x-axis label and specific size
plt.xlabel('frames')
# set y-axis label and specific size
plt.ylabel('mag')

plt.scatter(df0.frames, df0.mag, color="pink", label='Diagonal')
plt.scatter(df1.frames, df1.mag, color="plum", label='Turning/Roaming')
plt.scatter(df2.frames, df2.mag, color="purple", label='Vertical')
plt.scatter(df3.frames, df3.mag, color="orchid", label='Stop/Turns in each direction')
plt.scatter(df4.frames, df4.mag, color="salmon", label='Grooming')
plt.scatter(df5.frames, df5.mag, color="tomato", label='Turning')

x = data['frames']
y1 = data['mag']

iCountMovementAvg = np.convolve(y1, np.ones((N,)) / N, mode='valid')
y2 = np.zeros(y1.shape)
offset = y1.shape[0] - iCountMovementAvg.shape[0]
y2[offset:iCountMovementAvg.shape[0] + offset] = iCountMovementAvg

plt.plot(x, y2, label='Moving average')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

