import cv2 as cv
import numpy as np

#%% Optical Flow
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("23-7.mp4")

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
fps = 33
fourcc = cv.VideoWriter_fourcc(*codec)
out = cv.VideoWriter("23-7out.avi", fourcc, fps, (640, 360), True)

while (cap.isOpened()):
    # read in vcap property
    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float
    frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    #print((frame_width,frame_height))
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    angles = angle * 180/ np.pi
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    out.write(rgb)
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
#%%
import cv2
vidcap = cv2.VideoCapture('23-7out.avi')
success,image = vidcap.read()
count = 0
while success:
  #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  magnitude_normalized = hsv[..., 2]
  angles_pi = hsv[..., 0]
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1