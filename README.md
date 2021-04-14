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
![](/img/Taiwan_freeway_sample.gif)


## 作法

參考：\
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



