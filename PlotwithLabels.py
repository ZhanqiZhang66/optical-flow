import random
from itertools import count
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
video_name = '79F0-7-1-20 OF'
video_idx = r"D:\output_data_sheet\79F0-7-1-20 OF"
stablized_video_name = video_idx + ".mp4"
stablized_of_video_name = video_idx + "_out.avi"
output_data_name = video_idx +  "movement_threshold (1).csv" #"movement_threshold.xlsx"
hand_label_csv = video_idx +  "_reFormatted1.xlsx"
N = 20
# read in hand labels
df = pd.read_excel(hand_label_csv, sheet_name='labels') # can also index sheet by name or fetch all sheets

Diagonal = np.vstack((np.array(df['Diagonal start'].tolist()), np.array(df['Diagonal end'].tolist())))
Vertical = np.vstack((np.array(df['vertical start'].tolist()), np.array(df['vertical end'].tolist())))
Groom = np.vstack((np.array(df['groom start'].tolist()), np.array(df['groom end'].tolist())))
Horizontal = np.vstack((np.array(df['horizontal start'].tolist()), np.array(df['horizontal end'].tolist())))
Rearing = np.vstack((np.array(df['Rearing start'].tolist()), np.array(df['Rearing end'].tolist())))
Roaming = np.vstack((np.array(df['Roaming start'].tolist()), np.array(df['Roaming end'].tolist())))
Stop = np.vstack((np.array(df['stop start'].tolist()), np.array(df['stop end'].tolist())))
Turn = np.vstack((np.array(df['turn start'].tolist()), np.array(df['turn end'].tolist())))

Diagonal = Diagonal[:, ~np.isnan(Diagonal).any(axis=0)]
Vertical = Vertical[:, ~np.isnan(Vertical).any(axis=0)]
Groom = Groom[:, ~np.isnan(Groom).any(axis=0)]
Horizontal = Horizontal[:, ~np.isnan(Horizontal).any(axis=0)]
Rearing = Rearing[:, ~np.isnan(Rearing).any(axis=0)]
Roaming = Roaming[:, ~np.isnan(Roaming).any(axis=0)]
Stop = Stop[:, ~np.isnan(Stop).any(axis=0)]
Turn = Turn[:, ~np.isnan(Turn).any(axis=0)]

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
    plt.plot(x, y2,label='Moving average')
    #plt.fill_between(x, y2 - 1/2 *y1Error, y2 + 1/2 *y1Error, color="#3F5D7D")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    axes = plt.gca()
    #axes.set_ylim([300000, 700000np.arange(3)])
#%%
runing_mean = 1
with plt.style.context('ggplot'):
    #ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    #set x-axis label and specific size
    plt.title(video_name)
    plt.xlabel('frames')
    # set y-axis label and specific size
    plt.ylabel('magnitude')

    if output_data_name[-3:] == "csv":
        data = pd.read_csv(output_data_name)
    if output_data_name[-3:] == "lsx":
        data = pd.read_excel(output_data_name, sheet_name="sheet1")

    data_list =  data.values.tolist()
    data_list = np.array(data_list)
    x = data_list[:,0]
    y1 = data_list[:,1]

    iCountMovementAvg = np.convolve(y1, np.ones((N,)) / N, mode='valid')
    y2 = np.zeros(y1.shape)
    offset = y1.shape[0] - iCountMovementAvg.shape[0]
    y2[offset:iCountMovementAvg.shape[0] + offset] = iCountMovementAvg
    if runing_mean:
        data = pd.DataFrame({'frames': x, 'mag':y2})
        plt.title(video_name + ' moving average')
    else:
        plt.title(video_name)
    #plt.plot(x, y2, label='Moving average')



    for i in np.arange(np.shape(Diagonal)[1]):
        start, end = Diagonal[0,i], Diagonal[1,i]
        df0 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df0.frames, df0.mag, color="pink", s=0.3,label='Diagonal')
        else:
            plt.scatter(df0.frames, df0.mag, color="pink",s=0.3)
    for i in np.arange(np.shape(Vertical)[1]):
        start, end = Vertical[0,i], Vertical[1,i]
        df1 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df1.frames, df1.mag, color="plum",s=0.3,label='Vertical') # plum
        else:
            plt.scatter(df1.frames, df1.mag, color="plum",s=0.3)
    for i in np.arange(np.shape(Groom)[1]):
        start, end = Groom[0, i], Groom[1, i]
        df2 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df2.frames, df2.mag, color="purple",s=0.3 ,label='Groom')
        else:
            plt.scatter(df2.frames, df2.mag, color="purple",s=0.3)
    for i in np.arange(np.shape(Horizontal)[1]):
        start, end = Horizontal[0, i], Horizontal[1, i]
        df3 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df3.frames, df3.mag, color="red",s=0.3, label='Horizontal') # red
        else:
            plt.scatter(df3.frames, df3.mag, color="red",s=0.3)
    for i in np.arange(np.shape(Rearing)[1]):
        start, end = Rearing[0, i], Rearing[1, i]
        df4 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df4.frames, df4.mag, color="black",s=0.3, label='Rearing') #black
        else:
            plt.scatter(df4.frames, df4.mag, color="black",s=0.3)
    for i in np.arange(np.shape(Roaming)[1]):
        start, end = Roaming[0, i], Roaming[1, i]
        df5 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df5.frames, df5.mag, color="limegreen",s=0.3, label='Roaming')
        else:
            plt.scatter(df5.frames, df5.mag, color="limegreen",s=0.3)
    for i in np.arange(np.shape(Stop)[1]):
        start, end = Stop[0, i], Stop[1, i]
        df6 = data.query('frames >= @start & frames <= @end')
        if i == 0:
            plt.scatter(df6.frames, df6.mag, color="orange",s=0.3, label='Stop')
        else:
            plt.scatter(df6.frames, df6.mag, color="orange",s=0.3)
    for i in np.arange(np.shape(Turn)[1]):
        start, end = Turn[0, i], Turn[1, i]
        df7 = data.query('frames >= @start & frames <= @end')
        if i == 0 :
            plt.scatter(df7.frames, df7.mag, color="teal",s=0.3, label='Turn') # teal
        else:
            plt.scatter(df7.frames, df7.mag, color="teal",s=0.3)
    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    #%%
    import os
    import scipy.io

    scipy.io.savemat("D:", mdict={'mag': y2})
    #%% peak fiting
    import matplotlib.pyplot as plt
    from scipy.misc import electrocardiogram
    from scipy.signal import find_peaks
    #step 1: fine local peaks

    peaks, _  = find_peaks(y2, height=0)
    plt.xlim(-5, 20680)
    plt.plot(y2)

    plt.plot(peaks, y2[peaks], "x")

    plt.plot(np.zeros_like(y2), "--", color="gray")
    plt.show()

#     % compute
#     starting and ending
#     points
#     startpoint = zeros(size(pks));
#     endpoint = zeros(size(pks));
#     for ii = 1:length(pks)
#     plot(locs(ii) + peakWidth1(ii) * [-.5 .5], pks(ii) - p(ii) / 2 + [0 0], 'y')
#     sp = find((x < locs(ii)) & (dy_dx <= 0), 1, 'last');
#     if isempty(sp)
#         sp = 1;
#     end
#     startpoint(ii) = sp;
#     ep = find((x > locs(ii)) & (dy_dx >= 0), 1, 'first') - 1;
#     if isempty(ep)
#         ep = length(x);
#     end
#     endpoint(ii) = ep;
#     plot(x(startpoint(ii)), y(startpoint(ii)), 'og')
#     plot(x(endpoint(ii)), y(endpoint(ii)), 'sr')
# end
    # #step 2: for each local peaks, ganssian peak fit
    # def gaussian(x, a, b, c):
    #     return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))
    #
    #
    # #popt, pcov = curve_fit(func, xdata, ydata)