# 線性回歸 CRISP-DM 演示專案

這是一個使用 Python 和 Streamlit 建立的互動式教學應用程式，旨在演示如何按照 **CRISP-DM** (跨行業資料探勘標準流程) 實作線性回歸 (Linear Regression) 模型。
live demo: https://wilson052755.github.io/DIC7-ML/

## 專案功能

- **1. 業務理解 (Business Understanding)**：定義預測目標。
- **2. 資料理解 (Data Understanding)**：根據側邊欄參數生成合成資料，包含雜訊控制。
- **3. 資料準備 (Data Preparation)**：自動進行資料分割與標準化縮放。
- **4. 建立模型 (Modeling)**：使用 scikit-learn 訓練線性回歸模型。
- **5. 模型評估 (Evaluation)**：計算 MSE, RMSE, R² 指標，並視覺化回歸線。
- **6. 部署 (Deployment)**：提供訓練好的模型檔案 (`.pkl`) 下載功能。

## 如何操作

### 1. 安裝環境
請確保您的系統已安裝 Python，然後安裝所需的套件：

```bash
pip install -r requirements.txt
```

### 2. 執行應用程式
在終端機輸入以下指令啟動 Streamlit：

```bash
streamlit run app.py
```

執行後，瀏覽器會自動開啟應用程式頁面（通常是 `http://localhost:8501` 或 `8502`）。

### 3. 操作步驟
1. 在左側側邊欄調整**樣本數量**、**雜訊變異量**與**隨機種子**。
2. 點擊「**生成資料**」按鈕。
3. 向下滾動頁面查看各個階段的分析結果。
4. 在「**進行預測**」區塊輸入自訂的 $x$ 值來查看模型的預測結果。
5. 在「**部署**」區塊點擊下載按鈕儲存模型。

## 依賴套件
- streamlit
- numpy
- pandas
- matplotlib
- scikit-learn
- joblib
