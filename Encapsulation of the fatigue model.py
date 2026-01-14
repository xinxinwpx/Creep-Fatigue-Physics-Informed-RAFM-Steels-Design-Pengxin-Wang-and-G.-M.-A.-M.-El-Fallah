# Creep–Fatigue Design of RAFM Steels Using Physics-Informed and Data-Driven Surrogate Models with Multi-Objective Optimisation
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib

# ===================================================================
# Step 1. 使用你的最佳超参数构建 MLP
# ===================================================================
best_params = {
    "hidden_units": 222,
    "n_layers": 2,
    "learning_rate_init": 0.0020824537221507816,
    "alpha": 0.00010360111329331677
}

# 创建 MLP 模型
hidden_layer_sizes = tuple([best_params["hidden_units"]] * best_params["n_layers"])
fatigue_model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    learning_rate_init=best_params["learning_rate_init"],
    alpha=best_params["alpha"],
    max_iter=1200,
    random_state=42
)

# ===================================================================
# Step 2. 训练模型并保存 scaler（仅运行一次）
# ===================================================================
def train_fatigue_model(data_path="疲劳建模对比.xlsx"):
    """
    训练疲劳模型并保存：
    - fatigue_model.pth
    - scaler_fatigue.pkl
    """
    df = pd.read_excel(data_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]  # target: log10Nf

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8:2 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 训练模型
    fatigue_model.fit(X_train, y_train)

    # 保存模型与 scaler
    joblib.dump(fatigue_model, "fatigue_model.pkl")
    joblib.dump(scaler, "scaler_fatigue.pkl")

    print("✅ 疲劳模型训练完成并已保存：fatigue_model.pkl, scaler_fatigue.pkl")


# ===================================================================
# Step 3. 加载模型（用于预测）
# ===================================================================
def load_fatigue_model():
    """
    加载训练好的疲劳预测模型与 scaler
    """
    global fatigue_model, scaler_fatigue
    fatigue_model = joblib.load("fatigue_model.pkl")
    scaler_fatigue = joblib.load("scaler_fatigue.pkl")
    print("📌 已加载疲劳模型与标准化器")


# ===================================================================
# Step 4. 封装后的预测接口（多目标优化使用）
# ===================================================================
def predict_fatigue_life(feature_vector):
    """
    输入:
        feature_vector: 1D array (长度 = 特征数)
    输出:
        predicted log10Nf (float)
    """
    # 转为 numpy
    x = np.array(feature_vector).reshape(1, -1)

    # 标准化
    x_norm = scaler_fatigue.transform(x)

    # 预测
    y_pred = fatigue_model.predict(x_norm)[0]

    return float(y_pred)
