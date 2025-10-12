
"""
主题：旅客票价预测模型

处理真实数据时，第一步永远是理解每个字段的含义并大胆地剔除无关和泄露信息的列。你的模型质量，始于你的特征选择。

目标变量 (Target Variable):

fare: 航线的平均票价，这是我们要预测的目标。

建议使用的特征 (Features to Use):

quarter: 季度。可以作为数值或类别特征，反映季节性。

nsmiles: 航线距离。通常是影响票价的最关键因素之一。

passengers: 乘客数量。反映了航线的热门程度。

airport_1, airport_2: 具体的出发和到达机场代码。这是关键的类别特征。

carrier_lg: 该航线上的主要航空公司。也是关键的类别特征。
"""
import pandas as pd
import numpy as np

# 导入所有需要的 scikit-learn 模块
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

# --- 1. 数据加载与清洗 ---
try:
    df = pd.read_csv('./US Airline Flight Routes and Fares.csv')
except FileNotFoundError:
    print("错误：请确保 'US Airline Flight Routes and Fares.csv' 文件在当前目录下。")
    exit()

# 根据我们的分析，选择有用的列
columns_to_keep = [
    'quarter', 'nsmiles', 'passengers', 'airport_1', 
    'airport_2', 'carrier_lg', 'fare'
]
df_clean = df[columns_to_keep].copy()

# 为简化问题，先丢弃含有缺失值的行 (在真实项目中会用更复杂的方法)
df_clean.dropna(inplace=True)

# 分离特征 (X) 和目标 (y)
X = df_clean.drop('fare', axis=1)
y = df_clean['fare']


# --- 2. 特征工程 (优化点) ---
print("正在进行特征工程：创建'Route'特征...")
# 创建交互特征 'Route'
X['Route'] = X['airport_1'] + '-' + X['airport_2']
# 丢弃原始的机场列
X = X.drop(['airport_1', 'airport_2'], axis=1)


# --- 3. 预处理流程定义 ---
# 更新特征列表
numeric_features = ['quarter', 'nsmiles', 'passengers']
categorical_features = ['carrier_lg', 'Route'] # 使用新的'Route'特征

# 创建统一的预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    # remainder='passthrough' 可以保留未指定的列（如果有的话）
    remainder='passthrough' 
)

# --- 4. 数据分割 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 5. 训练与评估：基准模型 (Linear Regression) ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("\n--- 正在训练线性回归模型 ---")
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("--- 线性回归模型评估结果 ---")
print(f"均方根误差 (RMSE): {rmse_lr:.4f}")
print(f"R² 分数: {r2_lr:.4f}")


# --- 6. 训练与评估：优化模型 (Random Forest + Hyperparameter Tuning) ---

# 定义要调优的模型
rfr = RandomForestRegressor(random_state=42)

# 定义超参数的搜索范围
param_dist = {
    'n_estimators': randint(100, 250),
    'max_depth': randint(10, 30),
    'min_samples_leaf': randint(5, 15)
}

# 创建 RandomizedSearchCV 对象
# n_iter 设置为10次以加快演示速度，竞赛时可适当增加
random_search = RandomizedSearchCV(
    estimator=rfr, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=3, 
    random_state=42, 
    n_jobs=-1,
    scoring='neg_mean_squared_error' 
)

# 将调优工具整合进最终的管道
rf_pipeline_tuned = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', random_search)
])

print("\n--- 正在训练随机森林并进行超参数调优 (可能需要几分钟) ---")
rf_pipeline_tuned.fit(X_train, y_train)

print("\n--- 调优完成，找到的最佳参数: ---")
print(random_search.best_params_)

# 使用调优后的最佳模型进行评估
y_pred_rf_tuned = rf_pipeline_tuned.predict(X_test)
rmse_rf_tuned = np.sqrt(mean_squared_error(y_test, y_pred_rf_tuned))
r2_rf_tuned = r2_score(y_test, y_pred_rf_tuned)

print("\n--- 调优后的随机森林模型评估结果 ---")
print(f"均方根误差 (RMSE): {rmse_rf_tuned:.4f}")
print(f"R² 分数: {r2_rf_tuned:.4f}")

# --- 7. 结论 ---
print("\n--- 模型对比结论 ---")
if r2_rf_tuned > r2_lr:
    print("调优后的随机森林模型表现更优。")
    print(f"相比线性回归, R²分数提升了 {(r2_rf_tuned - r2_lr):.4f}。")
else:
    print("线性回归模型表现更优或相当。")