# AutonomousDriving_HW1_LaneLinesDetect
NCKU Practices of Autonomous Driving course homework

## 目標
車道邊線偵測

## 影片素材
[國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1 - Youtube](https://www.youtube.com/watch?v=0crwED4yhBA)

## 成果
![](/img/sample.gif)
完整連結：[NCKU_AutonomousDriving_HW1_LaneLinesDetect - Youtube](https://www.youtube.com/watch?v=5E8ZZ89CN6o)



## 作法

參考：[OanaGaskey / Advanced-Lane-Detection - github](https://github.com/OanaGaskey/Advanced-Lane-Detection)

**Step 1**
1. 將圖片經過 Sobel x 和 Sobel y 轉換
![](/img/sobel_result.png)
2. 邊緣膨脹
![](/img/sobelERODE_result.png)
3. 進行 AND 運算
![](/img/sobelAND_result.png)


**Step 2**
* 將影像轉換為鳥瞰視角
![](/img/birdeye_area.jpg)
![](/img/birdeye_result.jpg)

**Step 3**
* 對圖片做垂直方向的加總以判別車道線位置
![](/img/histogram.jpeg)

**Step 4**
* 4
