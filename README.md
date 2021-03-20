# AutonomousDriving_HW1_LaneLinesDetect
NCKU Practices of Autonomous Driving course homework

## 目標
車道邊線偵測

## 影片素材
[國道一號 中山高速公路 北向 高雄-基隆 374K-0K 全程 路程景National Highway No. 1](https://www.youtube.com/watch?v=0crwED4yhBA)

## 作法

參考：[OanaGaskey / Advanced-Lane-Detection - github](https://github.com/OanaGaskey/Advanced-Lane-Detection)

**Step 1**
* 將原始影像轉換為魚眼視角(須包含欲辨識的車道線)
![](/img/Figure_1.png)

**Step 2**
* 將圖片做轉換並疊加
![](/img/Figure_2.png)

**Step 3**
* 對圖片做垂直方向的加總以判別車道線位置