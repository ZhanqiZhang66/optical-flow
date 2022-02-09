import kmeans1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from pandas import *
import numpy as np
#%%
video_idx = r"D:\output_data_sheet\79F0-7-1-20-OF"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
output_data_name = video_idx + "movement_threshold_updateBG.xlsx" #"movement_threshold (1).csv"
hand_label_csv = video_idx +  "_reFormatted.xls"
N = 20

# xls = ExcelFile(hand_label_csv)
# df = xls.parse(xls.sheet_names[0])
# my_dict = df.to_dict()
if output_data_name[-3:] == "csv":
    data = pd.read_csv(output_data_name)
if output_data_name[-3:] == "lsx":
    data = pd.read_excel(output_data_name, sheet_name="sheet1")
#%%
x = data['frames']
y1 = data['mag']
plt.title()
plt.scatter(x,y1,s=0.3)
plt.show()
#%%
y2 = list(y1)
for idx, y in enumerate(y1):
    if x[idx] % 30 == 26:
        y2.remove(y)
print(len(y2))
#%%
#x = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
k = 8
colors = ['k','tomato','g','b','pink','plum','purple','orchid']
clusters, centroids = kmeans1d.cluster(y2, k)
#%%
#print(clusters)   # [1, 1, 1, 0, 3, 3, 3, 2, 2, 2]
print(centroids)  # [-50.0, 4.1, 94.0, 200.5]if
for n in range(k):
    # Filter data points to plot each in turn.
    idx = [i for i,x in enumerate(clusters) if x==n]
    ys = [y2[i] for i in idx]
    xs = list(x[idx])

    plt.scatter(xs, ys, color=colors[n], s=0.3)

plt.title("Points by cluster")
plt.show()