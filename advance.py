import cv2
import numpy as np
import matplotlib.pyplot as plt

# PARAM
SRC_PATH = './data/video.mp4'
OUT_PATH = './output.mp4'
NUM = 7200


# get frame
cap = cv2.VideoCapture(SRC_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

for n in range(NUM):
    n += 1
    if n % 20 == 0:
        print(f'{n}/{NUM}', end='\r')

    # Capture frame-by-frame
    ret, frame = cap.read()
    if(n < 4100):
        continue
    img = frame

    # --------------------------------------------------------------------------------------------- #

    # Transform image to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # H = hls[:, :, 0]
    # S = hls[:, :, 2]
    # sat_binary = np.zeros_like(S)
    # # Detect pixels that have a high saturation value
    # sat_binary[(S > 90) & (S <= 255)] = 1

    # hue_binary = np.zeros_like(H)
    # # Detect pixels that are yellow using the hue component
    # hue_binary[(H > 10) & (H <= 25)] = 1

    img_combine = np.logical_or(sx_binary, white_binary).astype(np.uint8)

    # --------------------------------------------------------------------------------------------- #
    # MASK
    mask_ROI = np.zeros([height, width, 1], dtype=np.uint8)
    dots_ROI = np.array([(100, 700), (450, 250), (830, 250), (1180, 700)])
    cv2.drawContours(mask_ROI, [dots_ROI], 0, 255, -1)
    mask_ROI = cv2.bitwise_not(mask_ROI)
    mask_LOGO = np.zeros([height, width, 1], dtype=np.uint8)
    dots_LOGO = np.array(
        [(1040, 660), (1260, 660), (1260, 715), (1040, 715)])
    cv2.drawContours(mask_LOGO, [dots_LOGO], 0, 255, -1)
    mask = cv2.bitwise_or(mask_ROI, mask_LOGO)
    mask = cv2.bitwise_not(mask)
    
    img_combine = cv2.bitwise_or(img_combine, img_combine, mask=mask)

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
        src=img_combine, M=M, dsize=(img.shape[0], img.shape[1]))

    # --------------------------------------------------------------------------------------------- #

    binary_warped = img_birdeye

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # --------------------------------------------------------------------------------------------- #
    ### Fit a second order polynomial to each with np.polyfit() ###
    print(left_lane_inds)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # --------------------------------------------------------------------------------------------- #
    

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
        
    margin = 7
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 0, 100))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 100))
    

    # --------------------------------------------------------------------------------------------- #
    # img_birdeye = cv2.cvtColor(img_birdeye*255, cv2.COLOR_GRAY2BGR)
    # cv2.imshow('', result)
    # cv2.waitKey()
    # exit()

    # OUTPUT
    weight = cv2.warpPerspective(window_img, M_inv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, weight, 0.9, 0)

    # cv2.imshow('', result)
    # cv2.waitKey()
    # exit()

    
    out.write(result)
