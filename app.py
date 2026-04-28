import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import io

# 頁面配置
st.set_page_config(page_title="線性回歸 CRISP-DM 演示", layout="wide")

# 側邊欄控制
st.sidebar.header("🛠️ 資料生成參數")
n_samples = st.sidebar.slider("樣本數量 (n)", 100, 1000, 500)
noise_variance = st.sidebar.slider("雜訊變異量", 0, 1000, 100)
random_seed = st.sidebar.number_input("隨機種子", value=42)
generate_btn = st.sidebar.button("生成資料")

@st.cache_data
def generate_synthetic_data(n, variance, seed):
    np.random.seed(seed)
    x = np.random.uniform(-100, 100, n)
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-50, 50)
    noise_mean = np.random.uniform(-10, 10)
    noise = np.random.normal(noise_mean, np.sqrt(variance), n)
    y = a * x + b + noise
    
    df = pd.DataFrame({'x': x, 'y': y})
    return df, a, b

# 1. 業務理解 (Business Understanding)
st.header("1. 業務理解 (Business Understanding)")
st.write("""
本專案的目標是使用 **線性回歸 (Linear Regression)** 建立一個預測模型。
在業務情境中，我們旨在理解自變數 ($x$) 與因變數 ($y$) 之間的關係，以便進行數據驅動的預測。
""")

# 2. 資料理解 (Data Understanding)
st.header("2. 資料理解 (Data Understanding)")
if 'data' not in st.session_state or generate_btn:
    df, true_a, true_b = generate_synthetic_data(n_samples, noise_variance, random_seed)
    st.session_state.data = df
    st.session_state.true_a = true_a
    st.session_state.true_b = true_b

df = st.session_state.data
true_a = st.session_state.true_a
true_b = st.session_state.true_b

col1, col2 = st.columns(2)
with col1:
    st.subheader("資料預覽")
    st.dataframe(df.head(10), use_container_width=True)
with col2:
    st.subheader("地面真值 (Ground Truth) 參數")
    st.write(f"**真實斜率 (a):** {true_a:.4f}")
    st.write(f"**真實截距 (b):** {true_b:.4f}")

# 3. 資料準備 (Data Preparation)
st.header("3. 資料準備 (Data Preparation)")
st.info("**關於特徵縮放的說明：** 我們使用 `StandardScaler` 來標準化特徵 $x$。這能確保模型優化過程更加穩定且快速，方法是將資料中心化於 0 並具有單位變異量。")

X = df[['x']].values
y = df['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write(f"訓練集大小：{len(X_train)} | 測試集大小：{len(X_test)}")

# 4. 建立模型 (Modeling)
st.header("4. 建立模型 (Modeling)")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 將縮放後的係數轉換回原始比例以進行比較
learned_a = model.coef_[0] / scaler.scale_[0]
learned_b = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])

st.success("已成功使用 scikit-learn LinearRegression 訓練模型。")

# 5. 模型評估 (Evaluation)
st.header("5. 模型評估 (Evaluation)")
y_pred = model.predict(X_test_scaled)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("均方誤差 (MSE)", f"{mse:.2f}")
metrics_col2.metric("均方根誤差 (RMSE)", f"{rmse:.2f}")
metrics_col3.metric("R² 判定係數", f"{r2:.4f}")

st.subheader("參數比較")
comparison_df = pd.DataFrame({
    "參數": ["斜率 (a)", "截距 (b)"],
    "真實值 (True)": [true_a, true_b],
    "學習值 (Learned)": [learned_a, learned_b]
})
st.table(comparison_df)

st.subheader("回歸視覺化")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X, y, alpha=0.5, label="資料點", color="#3498db", s=10)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
y_range_pred = model.predict(X_range_scaled)
ax.plot(X_range, y_range_pred, color="#e74c3c", linewidth=3, label="回歸線")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig)

# 預測區段
st.divider()
st.subheader("🔮 進行預測")
input_x = st.number_input("請輸入 x 值：", value=0.0)
input_scaled = scaler.transform(np.array([[input_x]]))
prediction = model.predict(input_scaled)[0]
st.write(f"當 x={input_x} 時，預測的 y 為：**{prediction:.4f}**")

# 6. 部署 (Deployment)
st.header("6. 部署 (Deployment)")
st.write("在此階段，我們匯出訓練好的模型，以便在生產環境中使用。")

model_data = {
    'model': model,
    'scaler': scaler
}

buffer = io.BytesIO()
joblib.dump(model_data, buffer)
buffer.seek(0)

st.download_button(
    label="📥 下載訓練好的模型 (joblib)",
    data=buffer,
    file_name="linear_model.pkl",
    mime="application/octet-stream"
)
