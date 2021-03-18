import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    frames = getFrame('./data/video.mp4', 1)

    for frame in frames:
        img = frame

        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.GaussianBlur(img, (9,15), 1.5)
        showImg(img)
        img = cv2.Canny(img, 100, 200)

        showImg(img)


def showImg(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def getFrame(videoPath, num):

    cap = cv2.VideoCapture(videoPath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    frames = np.zeros([0, height, width, 3], dtype=np.uint8)
    cnt = 0
    while(cnt < num):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # print(frame.shape)
        # plt.imshow(frame)
        # plt.show()
        # exit()
        frames = np.concatenate(
            (frames, np.expand_dims(frame, axis=0)), axis=0)
        cnt += 1

    return frames


if __name__ == '__main__':
    main()
