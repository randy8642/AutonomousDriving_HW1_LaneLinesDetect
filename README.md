# 自動駕駛實務 作業1 道路邊線檢測 AutonomousDriving_HW1_LaneLinesDetect

NCKU Practices of Autonomous Driving course homework

## 目標

車道邊線偵測

## 影片素材

1. [Udacity Test Video - solidWhiteRight](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidWhiteRight.mp4)
2. [Udacity Test Video - solidYellowLeft](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidYellowLeft.mp4)
3. [Udacity Test Video - challenge](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/challenge.mp4)
4. [國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube](https://www.youtube.com/watch?v=0crwED4yhBA)

## 成果

**各影片做法相同但參數不同**

1. Udacity Test Video - solidWhiteRight
完整影片：[solidWhiteRight_result](/output/solidWhiteRight_result.mp4)
![solidWhiteRight_sample](/img/solidWhiteRight_sample.gif)

2. Udacity Test Video - solidYellowLeft
完整影片：[solidYellowLeft_result](/output/solidYellowLeft_result.mp4)
![](/img/solidYellowLeft_sample.gif)
3. Udacity Test Video - challenge
完整影片：[challenge_result](/output/challenge_result.mp4)
![](/img/challenge_sample.gif)
4. 國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube
完整影片：[NCKU_AutonomousDriving_HW1_LaneLinesDetect - Youtube](https://www.youtube.com/watch?v=5E8ZZ89CN6o)
![](/img/Taiwan_freeway_sample.gif)


## 作法

參考：\
[OanaGaskey / Advanced-Lane-Detection - github](https://github.com/OanaGaskey/Advanced-Lane-Detection)\
[udacity / CarND-LaneLines-P1 - github](https://github.com/udacity/CarND-LaneLines-P1)

### Step 0 讀取影片

```python
    '''
    ## 可設定參數 ##
    SRC_PATH    要處理的影片檔位置
    '''
    cap = cv2.VideoCapture(SRC_PATH)

    # 取得影格(呼叫1次取1張)
    ret, frame = cap.read()
    img = frame
```

### Step 1 特徵擷取

1. 將圖片經過 Sobel x 和 Sobel y 轉換
```python
    # 將圖片轉換為灰階
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    # 執行 Sobel x & Sobel y
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)

    # 將結果縮放至 0-255
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) 
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # 遮罩
    sx_binary = np.zeros_like(scaled_sobelx)
    sy_binary = np.zeros_like(scaled_sobely)
    sx_binary[(scaled_sobelx >= 20) & (scaled_sobelx <= 130)] = 1
    sy_binary[(scaled_sobely >= 50) & (scaled_sobely <= 150)] = 1
```

![](/img/sobel_result.png)

2. 邊緣侵蝕
```python
    # 設定 kernal size
    kernel = np.ones((3, 3), np.uint8)

    # 將 binary mask 0/1轉換
    sx_binary = cv2.bitwise_not(sx_binary)
    sy_binary = cv2.bitwise_not(sy_binary)

    # 侵蝕
    sx_binary = cv2.erode(sx_binary, kernel, iterations=2)
    sy_binary = cv2.erode(sy_binary, kernel, iterations=2)

    # 將 binary mask 0/1轉換回來
    sx_binary = cv2.bitwise_not(sx_binary)
    sy_binary = cv2.bitwise_not(sy_binary)
```
![](/img/sobelERODE_result.png)

3. 進行 AND 運算

```python
    img_combine = np.logical_and(sx_binary, sy_binary).astype(np.uint8)
```
![](/img/sobelAND_result.png)


### Step 2 視角轉換
* 將影像轉換為鳥瞰視角
```python
    '''
    ## 可設定參數 ##
    src     轉換區域
    '''
    # 原圖片轉換範圍
    src = np.float32([
        (-1000, 720),   # 左下角
        (405, 460),     # 左上角
        (830, 460),     # 右上角
        (2600, 720)     # 右下角
    ])

    # 轉換後大小
    dst_height = img.shape[1]
    dst_width = img.shape[0]
    dst = np.float32([
        [0, dst_height],        # 左下角
        [0, 0],                 # 左上角
        [dst_width, 0],         # 右上角
        [dst_width, dst_height] # 右下角
    ])

    # 計算轉換與反轉換係數
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # 執行轉換
    img_birdeye = cv2.warpPerspective(src=img_combine, M=M, dsize=(dst_width, dst_height))
```
![](/img/birdeye_area.jpg)
![](/img/birdeye_result.jpg)

### Step 3 偵側車道
* 對圖片做垂直方向的加總以判別車道線位置\
    共4個判斷區間
```python
    binary_warped = img_birdeye

    # 垂直方向疊加
    histogram = np.sum(binary_warped, axis=0)

```
![](/img/histogram.jpg)

### Step 4 車道追蹤
1. 將上一步驟的峰值位置當作車道線起點
```python
    # 取各區段的峰值
    areaBoundary = [
        0, 
        int(histogram.shape[0]//4), 
        int(histogram.shape[0]//4 * 2), 
        int(histogram.shape[0]//4 * 3), 
        histogram.shape[0]
        ]
    
    laneBase = [
        np.argmax(histogram[areaBoundary[0]:areaBoundary[1]]) + reaBoundary[0],
        np.argmax(histogram[areaBoundary[1]:areaBoundary[2]]) + areaBoundary[1],
        np.argmax(histogram[areaBoundary[2]:areaBoundary[3]]) + areaBoundary[2],
        np.argmax(histogram[areaBoundary[3]:areaBoundary[4]]) + areaBoundary[3]
        ]
```
2. 由下而上，將畫面切割為數個水平區塊
3. 透過前一區塊中的**非零像素平均**調整下一區塊左右位置以追蹤車道彎曲
```python
    '''
    ## 可設定參數 ##
    nwindows    垂直方向切割區塊數    
    margin      區塊由中心左右延伸的寬度
    minpixel    是否儲存像素點的閥值(儲存的點會用於擬合曲線)
    '''
    
    window_height = int(binary_warped.shape[0]//nwindows)

    x_point = []
    y_point = []

    laneCurrent = laneBase[n_lane]
    for n_window in range(nwindows):

        # 計算 區塊 左右範圍
        x_range = [laneCurrent - margin , laneCurrent + margin]
        if x_range[0] < 0:
            x_range[0] = 0
        if x_range[1] >= binary_warped.shape[1]:
            x_range[1] = binary_warped.shape[1] - 1

        # 計算 區塊 上下範圍
        win_y_low = binary_warped.shape[0] - (n_window+1)*window_height
        win_y_high = binary_warped.shape[0] - n_window*window_height

        # 擷取 區塊
        window = binary_warped[win_y_low:win_y_high, x_range[0]:x_range[1]]

        # 取得區塊中非零像速點座標
        y_nonzero, x_nonzero = np.nonzero(window)
        x_nonzero += x_range[0]
        y_nonzero += win_y_low

        # 若區塊中非零像速點數量多於閥值
        # 則儲存這些像素點
        # 並取其水平方向平均，更新下一區塊的水平中心位置
        if np.count_nonzero(window) > minpixel:
            x_point.extend(x_nonzero)
            y_point.extend(y_nonzero)

            laneCurrent = np.mean(x_nonzero, axis=0, dtype=np.int32)
```
4. 每一判斷區間執行一次

![](/img/laneDetect_window.jpg)

### Step 5 擬合曲線
1. 將上一步驟所取得的像素點做**二次曲線**擬合
```python
    # 儲存線段空間
    laneLine_y = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    laneLine_x = np.ones([4, binary_warped.shape[0]]) * np.inf

    # 擬合二次曲線
    fit = np.polyfit(y_point, x_point, 2)

    # 轉換為點
    laneLine_x[n_lane, :] = fit[0] * laneLine_y**2 + fit[1] * laneLine_y + fit[2]
```
2. 透過**彎曲程度**判斷擬合是否正確
```python
    '''
    ## 可設定參數 ##
    threshold   可容許最大差距
    '''
    threshold = 60

    # 影像上下兩點的水平距離
    if np.abs(line_x[-1]-line_x[0]) > threshold:
        continue
    # 影像中間點與上下兩點的水平距離
    if np.abs(line_x[-1] - line_x[len(line_x)//2]) > threshold:
        continue
    if np.abs(line_x[0] - line_x[len(line_x)//2]) > threshold:
        continue
```
3. 繪製車道線
```python
    '''
    ## 可設定參數 ##
    margin      繪製的車道線左右寬度
    '''
    
    # 產生輸出圖層
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # 線段左邊界
    lineWindow1 = np.expand_dims(np.vstack([line_x - margin, laneLine_y]).T, axis=0)
    # 線段右邊界
    lineWindow2 = np.expand_dims(np.flipud(np.vstack([line_x + margin, laneLine_y]).T), axis=0)
    linePts = np.hstack((lineWindow1, lineWindow2))

    # 使用 openCV 填上曲線間區域
    cv2.fillPoly(window_img, np.int32([linePts]), (0, 0, 100))
```
![](/img/laneDetect_fitpoly.jpg)

### Step 6 影像疊合

1. 將繪製的車道線圖層轉換回原始視角
```python
    weight = cv2.warpPerspective(window_img, M_inv, (img.shape[1], img.shape[0]))
```
2. 將車道線圖層疊加到原始影像上
```python
    result = cv2.addWeighted(img, 1, weight, 0.9, 0)
```
![](/img/laneDetect_ori_result.jpg)


