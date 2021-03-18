import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    frames = getFrame('./data/video.mp4', 1)

    for frame in frames:
        img = frame

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (9, 15), 1.5)
        img = cv2.Canny(img, 100, 200)

        # MASK
        mask_ROI = np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.uint8)
        dots_ROI = np.array([(0, 700), (540, 460), (740, 460), (1280, 700)])
        cv2.drawContours(mask_ROI, [dots_ROI], 0, 255, -1)
        mask_ROI = cv2.bitwise_not(mask_ROI)
        mask_LOGO = np.zeros([frame.shape[0], frame.shape[1], 1], dtype=np.uint8)
        dots_LOGO = np.array(
            [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
        cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 255, -1)
        mask = cv2.bitwise_or(mask_ROI, mask_LOGO)
        mask = cv2.bitwise_not(mask)

        # APPLY MASK
        img = cv2.bitwise_and(img, mask)

        #
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
