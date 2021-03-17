import cv2
import os

import numpy as np

videoPath = './data/video.mp4'

cap = cv2.VideoCapture(videoPath)

try:
    if not os.path.exists('./data/frame'):
        os.makedirs('./data/frame')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Saves image of the current frame in jpg file
    name = './data/frame/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()