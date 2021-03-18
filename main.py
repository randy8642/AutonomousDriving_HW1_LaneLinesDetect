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
        mask_ROI = np.zeros(
            [frame.shape[0], frame.shape[1], 1], dtype=np.uint8)
        dots_ROI = np.array([(0, 700), (540, 460), (740, 460), (1280, 700)])
        cv2.drawContours(mask_ROI, [dots_ROI], 0, 255, -1)
        mask_ROI = cv2.bitwise_not(mask_ROI)
        mask_LOGO = np.zeros(
            [frame.shape[0], frame.shape[1], 1], dtype=np.uint8)
        dots_LOGO = np.array(
            [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
        cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 255, -1)
        mask = cv2.bitwise_or(mask_ROI, mask_LOGO)
        mask = cv2.bitwise_not(mask)

        # APPLY MASK
        img = cv2.bitwise_and(img, mask)

        #
        showImg(img)
        # Detect line
        LeftLane = []
        RightLane = []

        
        lines = cv2.HoughLines(img, 1, np.pi / 180, threshold=150)
  
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                
                m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
                if m < 0 and len(LeftLane) < 1:
                    LeftLane.extend([pt1, pt2])
                elif m > 0 and len(RightLane) < 1:
                    RightLane.extend([pt1, pt2])

        # draw
        cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(cdst, LeftLane[0], LeftLane[1], (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(cdst, RightLane[0], RightLane[1], (255, 0, 0), 3, cv2.LINE_AA)
        plt.imshow(cdst)
        plt.show()
        


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
