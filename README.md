
---

# YOLO Segmentation Visualization Notebook

## 概述
本 Notebook 演示了如何使用 [YOLOv11](https://github.com/ultralytics/ultralytics) 模型進行圖像分割，並通過 Matplotlib 完成可視化。主要流程包括：
1. 加載 YOLO 模型並進行推理。
2. 使用 OpenCV 讀取並處理圖片。
3. 提取 YOLO 結果中的分割多邊形，並使用 Matplotlib 對結果進行可視化。

---

## 文件結構與主要代碼邏輯

### 1. 必要依賴項
- **OpenCV (`cv2`)**：圖像讀取與處理。  
- **NumPy (`numpy`)**：數據運算與處理。  
- **Matplotlib (`matplotlib.pyplot`)**：圖像可視化。  
- **Ultralytics (`ultralytics`)**：YOLO 模型推理與結果處理。

安裝命令：
```bash
pip install opencv-python numpy matplotlib ultralytics
```

### 2. 加載 YOLO 模型
以下範例載入 YOLO 模型權重並對圖像進行推理：
```python
from ultralytics import YOLO

model = YOLO(r"runs\\segment\\train\\weights\\best.pt")  # 替換為你的權重路徑
image_path = r"test\\S__102228039.jpg"                   # 替換為你的圖像路徑
results = model.predict(source=image_path)
result = results[0]
```
- `best.pt`：訓練好的模型權重。
- `image_path`：輸入圖片路徑。
- 推理結果存在 `results` 中，`result = results[0]` 取得第一張圖像的推理結果。


---

## 模型效能報告

### 1. 數據統計
- 處理圖片數量：`300`
- 平均 IoU（Intersection over Union）：`0.9547`
- 平均 Dice 系數：`0.9766`

### 2. 分類報告

| 類別            | Precision | Recall | F1-Score | Support      |
|-----------------|-----------|--------|----------|--------------|
| **Background**  | 0.99      | 0.99   | 0.99     | 103,634,498  |
| **Foreground**  | 0.98      | 0.98   | 0.98     | 29,075,902   |
| **Accuracy**    |           |        | 0.99     | 132,710,400  |
| **Macro Avg**   | 0.99      | 0.99   | 0.99     | 132,710,400  |
| **Weighted Avg**| 0.99      | 0.99   | 0.99     | 132,710,400  |

### 3. 指標總結
- **精確率（Precision）**：區分前景與背景的準確度高達 98%-99%。  
- **召回率（Recall）**：對目標區域的覆蓋率為 98%-99%。  
- **F1 分數（F1-Score）**：綜合評估達 98%-99%。  
- **整體準確率（Accuracy）**：所有像素的準確率為 `99%`。

模型在分割任務中表現優異，適用於有明確分割需求的多種應用場景。

---

## 如何使用

1. **確保已安裝必要依賴項**  
   ```bash
   pip install opencv-python numpy matplotlib ultralytics
   ```
2. **替換為個人檔案路徑**  
   - 模型權重：`best.pt`  
   - 圖像：`image_path`
3. **運行 Notebook**  
   - 觀察輸出結果，包括原始圖像與疊加分割多邊形的圖像。

---

## 輸出結果

運行 Notebook 後將得到兩張圖：
1. **原始圖像**。  
2. **疊加分割多邊形的圖像**（可選填充或顯示輪廓）。

---

## 數據集資源

以下為多個可用於舌苔或口腔圖像分割與識別的數據集，方便進行模型訓練與測試。

### 1. [舌苔數據集 - 中醫圖像識別資源](https://gitcode.com/open-source-toolkit/7542e/blob/main/Tongue%20coating%20classification%20%E5%A2%9E%E5%BC%BA.zip)
- **標籤格式**：JSON（包含分割多邊形的頂點座標）。  
- **圖像尺寸**：通常為 512×512。  
- **主要標籤**：不同舌苔類型（如黑苔、地圖舌、紫苔、紅舌黃苔、白苔等）。  

### 2. [TongeImageDataset](https://github.com/BioHit/TongeImageDataset/tree/master)
- **規模**：300 張  
- **包含內容**：原圖與對應的 mask 圖片

### 3. [Oral Cancer (Lips and Tongue) images](https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images/data)
- **規模**：87 張口腔（癌症舌頭）+ 44 張非癌症舌頭  
- **用例**：針對口腔癌圖像識別的研究

### 4. [tooth-marked-tongue](https://www.kaggle.com/datasets/clearhanhui/biyesheji?utm_source=chatgpt.com)
- **規模**：546 張齒痕黑白舌頭 + 704 張正常舌頭  

---
以上數據集適用於中醫舌苔圖像識別、口腔癌檢測等應用領域，方便利用深度學習技術進行分割、分類與診斷研究。