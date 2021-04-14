import cv2
import numpy as np
import matplotlib.pyplot as plt

# PARAM
SRC_PATH = './data/challenge.mp4'
OUT_PATH = './challenge.mp4'
NUM = 1000

# get frame
cap = cv2.VideoCapture(SRC_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(
    *'mp4v'), 24, (width, height))

LeftLane = []
RightLane = []


for n in range(NUM):
    n += 1
    if n % 20 == 0:
        print(f'{n}/{NUM}')

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    img = frame

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (9, 9), 0.1)
    img = cv2.Canny(img, 100, 200)

    # MASK
    mask_ROI = np.zeros([height, width, 1], dtype=np.uint8)
    dots_ROI = np.array([(0, 700), (500, 530), (780, 530), (1280, 700)])
    cv2.drawContours(mask_ROI, [dots_ROI], 0, 255, -1)
    mask_ROI = cv2.bitwise_not(mask_ROI)
    mask_LOGO = np.zeros([height, width, 1], dtype=np.uint8)
    dots_LOGO = np.array(
        [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
    cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 255, -1)
    mask = cv2.bitwise_or(mask_ROI, mask_LOGO)
    mask = cv2.bitwise_not(mask)

    # APPLY MASK
    img = cv2.bitwise_and(img, mask)

    # Detect line
    LeftUpdate = False
    RightUpdate = False
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

            if m < 0.4 and m > -0.2:
                continue

            if m < 0:
                if not LeftUpdate:
                    LeftLane = [pt1_new, pt2_new]
                    LeftUpdate = True
                    continue
                if pt1_new[0] > LeftLane[0][0]:
                    LeftLane = [pt1_new, pt2_new]

            elif m > 0:
                if not RightUpdate:
                    RightLane = [pt1_new, pt2_new]
                    RightUpdate = True
                    continue
                if pt1_new[0] < RightLane[0][0]:
                    RightLane = [pt1_new, pt2_new]

    # DRAW LINE
    weight_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if len(LeftLane) > 0:
        cv2.line(weight_img, LeftLane[0], LeftLane[1],
                 (0, 0, 255), 8, cv2.LINE_AA)
    if len(RightLane) > 0:
        cv2.line(weight_img, RightLane[0],
                 RightLane[1], (0, 0, 255), 8, cv2.LINE_AA)

   
    # ADD WEIGHT
    outframe = cv2.addWeighted(frame, 1, weight_img, 0.5, 0)
   
    # OUTPUT
    out.write(outframe)
