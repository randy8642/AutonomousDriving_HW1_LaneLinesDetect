import cv2
import numpy as np
import matplotlib.pyplot as plt

# PARAM
SRC_PATH = './data/challenge.mp4'
OUT_PATH = './output.mp4'


# get frame
cap = cv2.VideoCapture(SRC_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(
    *'mp4v'), 24, (width, height))

tmpout = cv2.VideoWriter('./tmp.mp4', cv2.VideoWriter_fourcc(
    *'mp4v'), 24, (height, width))

n = 0
while True:
    n += 1
    if n % 20 == 0:
        print(f'{n}/???', end='\r')

    ret, frame = cap.read()

    if frame is None:
        break
    
    img = frame

    # --------------------------------------------------------------------------------------------- #

    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2HLS)[:,:,2]

    # 套用 Sobel x 和 Sobel y
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)

    # 縮放置 0 - 255
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobelx)
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    sy_binary = np.zeros_like(scaled_sobely)

    
    # 選取範圍
    sx_binary[(scaled_sobelx >= 20) & (scaled_sobelx <= 130)] = 1
    sy_binary[(scaled_sobely >= 50) & (scaled_sobely <= 150)] = 1

    # 邊緣膨脹
    kernel = np.ones((4, 4), np.uint8)
    sx_binary = cv2.bitwise_not(
        cv2.erode(cv2.bitwise_not(sx_binary), kernel, iterations=2))
    sy_binary = cv2.bitwise_not(
        cv2.erode(cv2.bitwise_not(sy_binary), kernel, iterations=2))

    img_combine = np.logical_and(sx_binary, sy_binary).astype(np.uint8)

    # --------------------------------------------------------------------------------------------- #

    src = np.float32([
        (100, 650),
        (600, 440),
        (730, 440),
        (1230, 650)
    ])

    dst = np.float32([
        [0, img.shape[1]],
        [0, 0],
        [img.shape[0], 0],
        [img.shape[0], img.shape[1]]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    img_birdeye = cv2.warpPerspective(
        src=img_combine, M=M, dsize=(img.shape[0], img.shape[1]))

    # cv2.drawContours(img,[src.astype(np.int32)],0,(0,255,0),2)
    # cv2.imshow('',img)
    # cv2.waitKey()
    # exit()

    # --------------------------------------------------------------------------------------------- #

    binary_warped = img_birdeye

    # 垂直方向疊加
    histogram = np.sum(binary_warped, axis=0)

    areaBoundary = \
        0, int(histogram.shape[0]//4 * 2), histogram.shape[0]

    laneBase = \
        [np.argmax(histogram[areaBoundary[0]:areaBoundary[1]]) + areaBoundary[0],
         np.argmax(histogram[areaBoundary[1]:areaBoundary[2]]) + areaBoundary[1], ]

    # ---------------------------------------------------------------------------------- #

    nwindows = 9
    margin = 50
    minpixel = 10

    window_height = np.int32(binary_warped.shape[0]//nwindows)

    laneLine_y = np.linspace(
        0, binary_warped.shape[0]-1, binary_warped.shape[0])
    laneLine_x = np.ones([4, binary_warped.shape[0]]) * -100

    for n_lane in range(len(laneBase)):

        x_point = []
        y_point = []

        # Up
        laneCurrent = laneBase[n_lane]
        for n_window in range(nwindows//2+1):

            x_range = laneCurrent - margin if laneCurrent - margin >= 0 else 0, laneCurrent + \
                margin if laneCurrent + \
                margin < binary_warped.shape[1] else binary_warped.shape[1] - 1

            win_y_low = binary_warped.shape[0] - (n_window+1)*window_height
            win_y_high = binary_warped.shape[0] - n_window*window_height

            window = binary_warped[win_y_low:win_y_high, x_range[0]:x_range[1]]
            y_nonzero, x_nonzero = np.nonzero(window)
            x_nonzero += x_range[0]
            y_nonzero += win_y_low

            if np.count_nonzero(window) > minpixel:
                x_point.extend(x_nonzero)
                y_point.extend(y_nonzero)

                laneCurrent = np.mean(x_nonzero, axis=0, dtype=np.int32)

        # DOWN
        laneCurrent = laneBase[n_lane]
        for n_window in range(nwindows//2):

            x_range = laneCurrent - margin if laneCurrent - margin >= 0 else 0, laneCurrent + \
                margin if laneCurrent + \
                margin < binary_warped.shape[1] else binary_warped.shape[1] - 1

            win_y_low = n_window*window_height
            win_y_high = (n_window+1)*window_height

            window = binary_warped[win_y_low:win_y_high, x_range[0]:x_range[1]]
            y_nonzero, x_nonzero = np.nonzero(window)
            x_nonzero += x_range[0]
            y_nonzero += win_y_low

            if np.count_nonzero(window) > minpixel:
                x_point.extend(x_nonzero)
                y_point.extend(y_nonzero)

                laneCurrent = np.mean(x_nonzero, axis=0, dtype=np.int32)
        # print(y_nonzero)
        # tmpimg = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
        # for xx in range(len(y_point)):

        #     tmpimg = cv2.circle(
        #         tmpimg, (x_point[xx], y_point[xx]), 1, (0, 255, 0), 1)

        # cv2.imshow('', tmpimg)
        # cv2.waitKey()
        if len(y_point) > 0:

            fit = np.polyfit(y_point, x_point, 2)
            laneLine_x[n_lane, :] = fit[0] * \
                laneLine_y**2 + fit[1]*laneLine_y + fit[2]

    # --------------------------------------------------------------------------------------------- #

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    margin = 10

    for line_x in laneLine_x:
        # print(np.abs(line_x[-1]-line_x[0]))
        if np.abs(line_x[-1]-line_x[0]) > 100:
            continue
        if np.abs(line_x[-1] - line_x[len(line_x)//2]) > 100:
            continue
        if np.abs(line_x[0] - line_x[len(line_x)//2]) > 100:
            continue

        lineWindow1 = np.expand_dims(
            np.vstack([line_x - margin, laneLine_y]).T, axis=0)
        lineWindow2 = np.expand_dims(
            np.flipud(np.vstack([line_x + margin, laneLine_y]).T), axis=0)
        linePts = np.hstack((lineWindow1, lineWindow2))

        cv2.fillPoly(window_img, np.int32([linePts]), (0, 0, 100))

    # --------------------------------------------------------------------------------------------- #

    # OUTPUT
    weight = cv2.warpPerspective(
        window_img, M_inv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, weight, 0.9, 0)

    o = cv2.cvtColor(img_birdeye*255, cv2.COLOR_GRAY2BGR)
    r = cv2.addWeighted(o, 0.8, window_img, 1, 0)
    tmpout.write(r)

    out.write(result)

    # cv2.imshow('',r )
    # cv2.waitKey()
