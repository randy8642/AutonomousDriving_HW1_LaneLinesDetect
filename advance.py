import cv2
import numpy as np
import matplotlib.pyplot as plt

# PARAM
SRC_PATH = './data/video.mp4'
OUT_PATH = './output.avi'
NUM = 480


# get frame
cap = cv2.VideoCapture(SRC_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (height, width))

for n in range(NUM):
    n += 1
    if n % 20 == 0:
        print(f'{n}/{NUM}', end='\r')

    # Capture frame-by-frame
    ret, frame = cap.read()

    img = frame

    # MASK LOGO
    mask_LOGO = np.ones([height, width, 1], dtype=np.uint8)*255
    dots_LOGO = np.array(
        [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
    cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 0, -1)
    img = cv2.bitwise_or(img, img, mask=mask_LOGO)
    
    # --------------------------------------------------------------------------------------------- #

    # Source points taken from images with straight lane lines, these are to become parallel after the warp transform
    src = np.float32([
        (-580, 720),  # bottom-left corner
        (454, 460),  # top-left corner
        (765, 460),  # top-right corner
        (1900, 720)  # bottom-right corner
    ])

    # Destination points are to be parallel, taking into account the image size
    dst = np.float32([
        [0, img.shape[1]],             # bottom-left corner
        [0, 0],                       # top-left corner
        [img.shape[0], 0],           # top-right corner
        [img.shape[0], img.shape[1]]  # bottom-right corner
    ])

    # Calculate the transformation matrix and it's inverse transformation
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    img_birdeye = cv2.warpPerspective(
        src=img, M=M, dsize=(img.shape[0], img.shape[1]))

    # --------------------------------------------------------------------------------------------- #
   
    # Transform image to gray scale
    gray_img = cv2.cvtColor(img_birdeye, cv2.COLOR_BGR2GRAY)

    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)

    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)

    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1


    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1

    
    # --------------------------------------------------------------------------------------------- #
    # Convert image to HLS
    hls = cv2.cvtColor(img_birdeye, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 90) & (S <= 255)] = 1
 
    

    hue_binary = np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 10) & (H <= 25)] = 1
 
    
    img_combine = sx_binary + white_binary + sat_binary + hue_binary
  
    img_combine = cv2.cvtColor(img_combine*255,cv2.COLOR_GRAY2BGR)

    # cv2.imshow('',img_combine)
    # cv2.waitKey()
    # exit()

    # OUTPUT
    out.write(img_combine)
