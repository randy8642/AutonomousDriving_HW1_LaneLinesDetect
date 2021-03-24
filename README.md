# 自動駕駛實務 作業1 道路邊線檢測 AutonomousDriving_HW1_LaneLinesDetect
NCKU Practices of Autonomous Driving course homework

## 目標
車道邊線偵測

## 影片素材
1. [Udacity Test Video - solidWhiteRight](https://github.com/udacity/CarND-LaneLines-P1/blob/master/test_videos/solidWhiteRight.mp4)
2. [國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube](https://www.youtube.com/watch?v=0crwED4yhBA)

## 成果
**各影片做法相同但參數不同**
1. Udacity Test Video - solidWhiteRight
![](/img/udacity_test_solidWhiteRight.gif)
完整連結：[udacity_test_solidWhiteRight](/img/udacity_test_solidWhiteRight.mp4)

2. 國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube
![](/img/sample.gif)
完整連結：[NCKU_AutonomousDriving_HW1_LaneLinesDetect - Youtube](https://www.youtube.com/watch?v=5E8ZZ89CN6o)



## 作法

參考：\
[OanaGaskey / Advanced-Lane-Detection - github](https://github.com/OanaGaskey/Advanced-Lane-Detection)\
[udacity / CarND-LaneLines-P1 - github](https://github.com/udacity/CarND-LaneLines-P1)

### Step 1 特徵擷取
1. 將圖片經過 Sobel x 和 Sobel y 轉換
![](/img/sobel_result.png)
2. 邊緣膨脹
![](/img/sobelERODE_result.png)
3. 進行 AND 運算
![](/img/sobelAND_result.png)


### Step 2 視角轉換
* 將影像轉換為鳥瞰視角
![](/img/birdeye_area.jpg)
![](/img/birdeye_result.jpg)

### Step 3 偵側車道
* 對圖片做垂直方向的加總以判別車道線位置\
    共4個判斷區間
![](/img/histogram.jpeg)

### Step 4 車道追蹤
1. 將上一步驟的峰值位置當作車道線起點
2. 由下而上，將畫面切割為數個水平區塊
3. 透過前一區塊中的**非零像素平均**調整下一區塊左右位置以追蹤車道彎曲
4. 每一判斷區間執行一次

    ![](/img/laneDetect_window.jpeg)

### Step 5 擬合曲線
1. 將上一步驟所取得的像素點做**二次曲線**擬合
2. 透過**彎曲程度**判斷擬合是否正確

    ![](/img/laneDetect_fitpoly.jpeg)

### Step 6 影像疊合
1. 將繪製的車道線圖層轉換回原始視角
2. 將車道線圖層疊加到原始影像上
![](/img/laneDetect_ori_result.jpg)

## 心得
>車道偵測最困難的地方是特徵擷取部分，
>需要套用和嘗試各種邊緣擷取及色彩的閥值，
>才能得到將車道特徵最大化的圖
