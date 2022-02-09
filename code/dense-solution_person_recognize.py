import cv2 as cv
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
from datetime import datetime, time
import numpy as np
import time as time2
import os
#%% Arguments
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
ap.add_argument("--return_counts", type=bool, default=True)
ap.add_argument("--mode", default='client')
ap.add_argument("--port", default=64873)
args = vars(ap.parse_args())


# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
            "csrt": cv.TrackerCSRT_create,
            "kcf": cv.TrackerKCF_create,
            "boosting": cv.TrackerBoosting_create,
            "mil": cv.TrackerMIL_create,
            "tld": cv.TrackerTLD_create,
            "medianflow": cv.TrackerMedianFlow_create,
            "mosse": cv.TrackerMOSSE_create
}
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    #tracker = cv.TrackerGOTURN_create()
# if the video argument is None, then the code will read from webcam (work in progress)
vs = cv.VideoCapture("shibuya.mp4")
#%% Optical Flow
# The video feed is read in as a VideoCapture object
#vs = cv.VideoCapture("23-7.mp4")

# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
# ret, first_frame = vs.read()
# # Converts frame to grayscale because we only
# # need the luminance channel for detecting edges - less computationally expensive
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# # Creates an image filled with zero intensities with the same dimensions as the frame
# mask = np.zeros_like(first_frame)
# # Sets image saturation to maximum
# mask[..., 1] = 255

codec = 'MJPG'
fps = 33
fourcc = cv.VideoWriter_fourcc(*codec)
out = cv.VideoWriter("23-7out.avi", fourcc, fps, (640, 360), True)

firstFrame = None
initBB2 = None
fps = None
differ = None
now = ''
framecounter = 0
trackeron = 0

while (vs.isOpened()):
    # read in vcap property
    frame_width = vs.get(cv.CAP_PROP_FRAME_WIDTH)  # float
    frame_height = vs.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    #print((frame_width,frame_height))
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    # if the frame can not be grabbed, then we have reached the end of the video
    if frame is None:
        break
    # resize the frame to 500
    frame = imutils.resize(frame, width=500)

    framecounter = framecounter + 1
    if framecounter > 1:

        (H, W) = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv.absdiff(firstFrame, gray)
        thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv.dilate(thresh, None, iterations=2)
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] #if imutils.is_cv2() else cnts[1]

        # loop over the contours identified
        contourcount = 0
        for c in cnts:
            contourcount = contourcount + 1

            # if the contour is too small, ignore it
            if cv.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            (x, y, w, h) = cv.boundingRect(c)
            initBB2 = (x, y, w, h)
            prott1 = r'C:\Users\zhanq\Downloads\MobileNetSSD_deploy.prototxt'
            prott2 = r'C:\Users\zhanq\Downloads\mobilenet_iter_73000.caffemodel'
            net = cv.dnn.readNetFromCaffe(prott1, prott2)

            CLASSES = ["person"]
            COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

            trackbox = frame[y:y+h, x:x+w]
            trackbox = cv.resize(trackbox, (224, 224))
            cv.imshow('image',trackbox)
            blob = cv.dnn.blobFromImage(cv.resize(trackbox, (300, 300)),0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                confidence_level = 0.8

                if confidence > confidence_level:
                    # extract the index of the class label from the `detections`, then compute the (x, y)-coordinates of
                    # the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    cv.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv.putText(frame, label, (startX, y),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # Start tracker
            now = datetime.now()
            if differ == None or differ > 9:
                 tracker.init(frame, initBB2)
                 fps = FPS().start()

                # check to see if we are currently tracking an object, if so, ignore other boxes
                # this code is relevant if we want to identify particular persons (section 2 of this tutorial)
    if initBB2 is not None:

        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        differ = 10
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            differ = abs(initBB2[0] - box[0]) + abs(initBB2[1] - box[1])
            i = tracker.update(lastframe)
            if i[0] != True:
                time2.sleep(4000)
        else:
            trackeron = 1

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, H - ((i * 20) + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                        2)

        # draw the text and timestamp on the frame
        now2 = datetime.now()
        time_passed_seconds = str((now2 - now).seconds)
        cv.putText(frame, 'Detecting persons', (10, 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    #
    # # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # # Calculates dense optical flow by Farneback method
    # # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    # flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # # Computes the magnitude and angle of the 2D vectors
    # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # angles = angle * 180/ np.pi
    # # Sets image hue according to the optical flow direction
    # mask[..., 0] = angle * 180 / np.pi / 2
    # # Sets image value according to the optical flow magnitude (normalized)
    # mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # # Converts HSV to RGB (BGR) color representation
    # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # # Opens a new window and displays the output frame
    # cv.imshow("dense optical flow", rgb)
    out.write(frame)
    # # Updates previous frame
    # prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    key = cv.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    if key == ord("d"):
        firstFrame = None
    lastframe = frame

# The following frees up resources and closes all windows
vs.release()
cv.destroyAllWindows()
#%%

vidcap = cv.VideoCapture('23-7out.avi')
success,image = vidcap.read()
count = 0
while success:
  #cv.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
  magnitude_normalized = hsv[..., 2]
  angles_pi = hsv[..., 0]
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1