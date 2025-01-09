# README: YOLO Segmentation Visualization Notebook

## 概述

這個 Notebook 演示了如何使用 [YOLOv11](https://github.com/ultralytics/ultralytics) 模型進行圖像分割，並通過 Matplotlib 對結果進行可視化。具體包括以下步驟：
1. 加載 YOLO 模型和推理。
2. 使用 OpenCV 處理圖片。
3. 提取分割多邊形並進行可視化。

---

## 文件結構和主要代碼邏輯

### 1. 必要的依賴項
Notebook 中使用了以下主要的 Python 庫：
- `cv2`： 用於處理圖像。
- `numpy`： 用於數據處理。
- `matplotlib.pyplot`： 用於圖像的可視化。
- `ultralytics.YOLO`： YOLO 模型的推理和結果處理。

安裝依賴項的命令：
```bash
pip install opencv-python numpy matplotlib ultralytics
```

### 2. 載入 YOLO 模型
以下代碼載入 YOLO 模型的權重，並進行推理：
```python
from ultralytics import YOLO

model = YOLO(r"runs\\segment\\train\\weights\\best.pt")  # 替換為你的模型路徑
image_path = r"test\\S__102228039.jpg"  # 替換為你的圖像路徑
results = model.predict(source=image_path)
result = results[0]
```

- `best.pt` 是訓練好的模型權重。
- `image_path` 是輸入圖片路徑。
- 推理結果存儲在 `results` 中。

### 3. 圖像處理與可視化
原圖使用 OpenCV 讀取，並用 Matplotlib 顯示：
```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(img)
```

提取 YOLO 結果中的分割多邊形，並繪製邊界線：
```python
polygons = result.masks.xy  # 取得多邊形座標
for poly in polygons:
    x = poly[:, 0]  # x 座標
    y = poly[:, 1]  # y 座標
    plt.plot(x, y, color='red', linewidth=2)  # 畫出多邊形邊界
plt.title("Segmentation Polygons")
plt.axis('off')
plt.show()
```

---

## 如何使用

1. 確保安裝了必要的依賴項。
2. 替換 `best.pt` 為你的模型權重路徑。
3. 替換 `image_path` 為你的輸入圖片路徑。
4. 運行 Notebook，查看分割結果的可視化。

---

## 輸出結果
運行 Notebook 後，輸出將顯示：
1. 原始圖像。
2. 標註了分割多邊形的圖像，可選擇顯示輪廓或填充區域。

