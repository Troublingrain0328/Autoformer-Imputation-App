
from sklearn.experimental import enable_iterative_imputer  # 显式启用 IterativeImputer
from sklearn.impute import IterativeImputer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import webbrowser
import time

# 设置 Streamlit 页面配置
st.set_page_config(page_title="深度学习数据处理 App", layout="wide")
st.title("🧠 深度学习数据预处理与评估平台")

# 上传数据
uploaded_file = st.file_uploader("📁 上传你的 CSV 数据", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 原始数据预览", df.head())

    # 解决日期列问题
    # 假设 'date_column' 是日期列
    if 'date_column' in df.columns:
        # 将日期列转换为 datetime 类型
        df['date_column'] = pd.to_datetime(df['date_column'])
        
        # 方法一：将日期列转换为时间戳（秒）
        df['date_timestamp'] = df['date_column'].astype(int) / 10**9  # 转换为时间戳（秒）
        
        # 或者方法二：提取日期特征（年、月、日、星期几等）
        df['year'] = df['date_column'].dt.year
        df['month'] = df['date_column'].dt.month
        df['day'] = df['date_column'].dt.day
        df['weekday'] = df['date_column'].dt.weekday
        df['hour'] = df['date_column'].dt.hour
        
        # 删除原始日期列
        df = df.drop(columns=['date_column'])
    
    # 缺失值插补与标准化
    st.subheader("🔧 数据插补与标准化")
    imputer = IterativeImputer(random_state=0)
    scaler = StandardScaler()

    # 执行插补和标准化
    imputed_data = imputer.fit_transform(df)
    scaled_data = scaler.fit_transform(imputed_data)

    # 创建插补后的数据框
    df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
    st.write("### 插补后的数据", df_imputed.head())

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(scaled_data, test_size=0.2, random_state=42)

    # 构建 PyTorch 数据集
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = torch.tensor(data, dtype=torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_loader = DataLoader(CustomDataset(X_train), batch_size=32, shuffle=True)

    st.success("数据已插补、标准化，并准备好用于模型训练。")

    # 示例模型（简单的自编码器）
    st.subheader("📉 模型训练示例")

    class AutoEncoder(torch.nn.Module):
        def __init__(self, input_dim):
            super(AutoEncoder, self).__init__()
            self.encoder = torch.nn.Linear(input_dim, 8)
            self.decoder = torch.nn.Linear(8, input_dim)

        def forward(self, x):
            x = torch.relu(self.encoder(x))
            x = self.decoder(x)
            return x

    model = AutoEncoder(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    num_epochs = st.slider("训练轮数", 1, 100, 10)
    losses = []

    progress = st.progress(0, text="模型训练中...")
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        progress.progress((epoch+1)/num_epochs, text=f"第 {epoch+1} / {num_epochs} 轮")

    st.line_chart(losses)

    # 模型评估
    st.subheader("📊 模型评估")
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    def get_result(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    result = get_result(X_test, preds)
    st.write(pd.DataFrame([result]))

    # 强制打开浏览器
    time.sleep(1)  # 等待一会儿，确保应用启动
    webbrowser.open('http://localhost:8501')  # 强制打开浏览器
