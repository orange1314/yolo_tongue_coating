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

資料集

[舌苔數據集-中醫圖像識別資源](https://gitcode.com/open-source-toolkit/7542e/blob/main/Tongue%20coating%20classification%20%E5%A2%9E%E5%BC%BA.zip)
---

### **舌苔數據集 - 中醫圖像識別資源**

此數據集包含不同舌苔種類的資料夾，每個資料夾中包含圖片及對應的標籤文件，用於圖像分割任務。標籤文件採用 JSON 格式，主要結構如下：

1. **`shapes`**
   - 定義標記的形狀（多邊形）。
   - **`label`**：標籤名稱，例如 `"black tongue coating"`，表示“黑舌苔”區域。
   - **`points`**：多邊形的頂點座標列表，用於描述標記區域的形狀。

2. **`imagePath`**
   - 圖像文件名稱，例如 `"black tongue coating_1.jpg"`。

3. **`imageData`**
   - 圖像的 base64 編碼，用於內嵌圖像數據。

4. **`imageHeight` 和 `imageWidth`**
   - 圖像尺寸：512x512 像素。

5. **`lineColor` 和 `fillColor`**
   - 線條顏色和填充顏色，用於標記區域的可視化。

此標籤文件適用於圖像標註工具（如 LabelMe），主要用於訓練深度學習模型進行圖像分割任務。

---

### **數據集內容**
- **黑舌苔（Black tongue coating）**：420 張照片  
- **地圖舌（Map tongue coating）**：80 張照片  
- **紫舌苔（Purple tongue coating）**：350 張照片  
- **紅舌黃苔厚膩苔（Red tongue yellow fur thick greasy fur）**：770 張照片  
- **紅舌厚膩苔（The red tongue is thick and greasy）**：550 張照片  
- **白舌厚膩苔（The white tongue is thick and greasy）**：300 張照片  

此數據集可用於中醫圖像識別及舌苔分類的研究與應用，特別適合深度學習模型的訓練與測試。


[TongeImageDataset](https://github.com/BioHit/TongeImageDataset/tree/master)
---
包含300張
包含原圖以及mask圖片

[Oral Cancer (Lips and Tongue) images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images/data)
---
包含87張癌症舌頭資料以及44張飛癌症舌頭資料


[tooth-marked-tongue](https://www.kaggle.com/datasets/clearhanhui/biyesheji?utm_source=chatgpt.com)
---
包含546張齒痕黑白舌頭以及704張正常舌頭