import cv2
import numpy as np


def main():

    frames = getFrame('./data/video.mp4', 4)

    print(frames.shape)


def getFrame(videoPath, num):

    cap = cv2.VideoCapture(videoPath)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

    frames = np.zeros([0, height, width, 3])
    cnt = 0
    while(cnt < num):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frames = np.concatenate(
            (frames, np.expand_dims(frame, axis=0)), axis=0)
        cnt += 1

    return frames


if __name__ == '__main__':
    main()
