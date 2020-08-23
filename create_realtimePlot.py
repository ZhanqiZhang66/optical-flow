import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
#%%
video_idx = r"D:\79F0-7-1-20-OF_TrimRunStopStand"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
output_data_name = video_idx + "movement_threshold.csv"

N = 15
def animate(i):

    data = pd.read_csv(output_data_name)

    x = data['frames']
    y1 = data['number of Movements']
    y1Error = data["number of Movements Error"]
    iCountMovementAvg = np.convolve(y1, np.ones((N,)) / N, mode='valid')
    y2 = np.zeros(y1.shape)
    offset = y1.shape[0] - iCountMovementAvg.shape[0]
    y2[offset:iCountMovementAvg.shape[0] + offset] = iCountMovementAvg
    plt.cla()

    plt.plot(x, y1, label='Raw data')
    plt.plot(x, y2,label='Moving average')
    #plt.fill_between(x, y2 - 1/2 *y1Error, y2 + 1/2 *y1Error, color="#3F5D7D")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    axes = plt.gca()
    #axes.set_ylim([300000, 700000])
with plt.style.context('ggplot'):
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)

    plt.tight_layout()
    plt.show()