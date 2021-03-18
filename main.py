import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (1280, 720))

    frames = getFrame('./data/video.mp4', 100)

    

    for n, rawframe in enumerate(frames):
        n += 1
        if n%20==0:
            print(f'{n}/{frames.shape[0]}')
        img = rawframe
        width, height = rawframe.shape[0], rawframe.shape[1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (9, 9), 1.5)
        img = cv2.Canny(img, 100, 200)

        # MASK
        mask_ROI = np.zeros([width, height, 1], dtype=np.uint8)
        dots_ROI = np.array([(0, 700), (540, 450), (740, 450), (1280, 700)])
        cv2.drawContours(mask_ROI, [dots_ROI], 0, 255, -1)
        mask_ROI = cv2.bitwise_not(mask_ROI)
        mask_LOGO = np.zeros([width, height, 1], dtype=np.uint8)
        dots_LOGO = np.array(
            [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
        cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 255, -1)
        mask = cv2.bitwise_or(mask_ROI, mask_LOGO)
        mask = cv2.bitwise_not(mask)

        # APPLY MASK
        img = cv2.bitwise_and(img, mask)

        # Detect line
        LeftLane = []
        RightLane = []
        lines = cv2.HoughLines(img, 1, np.pi / 180, threshold=50)

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

                # y = ma + b
                m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
                b = pt1[1] - (pt1[0] * m)
                pt1_new = (int((720 - b) // m), 720)
                pt2_new = (int((460 - b) // m), 460)

                if m < 0 and len(LeftLane) < 1:
                    LeftLane.extend([pt1_new, pt2_new])

                elif m > 0 and len(RightLane) < 1:
                    RightLane.extend([pt1_new, pt2_new])

        # draw
        # cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(LeftLane) > 0:
            cv2.line(rawframe, LeftLane[0], LeftLane[1], (0, 0, 255), 3, cv2.LINE_AA)
        if len(RightLane) > 0:
            cv2.line(rawframe, RightLane[0], RightLane[1], (255, 0, 0), 3, cv2.LINE_AA)

        out.write(rawframe)


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
