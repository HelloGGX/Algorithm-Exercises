"""
背景: 你需要为一个二手车交易平台构建一个价格预测模型。给定车辆的一些基本属性，模型需要能快速、准确地估算出其可能的售价。
ML限时挑战 (新数据集版): 汽车价格预测

数据集字段 (以你提供的为准): symboling, normalized-losses, make, num-of-doors, body-style, drive-wheels, engine-location, wheel-base, length, width, height, curb-weight, engine-type, num-of-cylinders, engine-size, fuel-system, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg, price, ... (以及其他衍生列)

你的任务清单:

数据加载与特征选择:
加载数据集。
目标变量 (y): price。
特征选择 (X): 为了在有限时间内完成挑战，我们只使用以下几个最关键的特征。请从原始数据中筛选出这些列，并丢弃所有其他列。

数值型特征: wheel-base, curb-weight, engine-size, horsepower, city-mpg
类别型特征: make, body-style, drive-wheels, engine-type

(新挑战) 这个数据集中，缺失值常用 '?' 表示。在开始前，你需要先将所有 '?' 替换为 np.nan，然后为简单起见，直接使用 .dropna() 清除所有包含缺失值的行。

预处理管道:
定义好你的 numeric_features 和 categorical_features 列表。

构建 ColumnTransformer：
对数值型特征进行标准化 (StandardScaler)。 (由于我们已经dropna，此处的imputer可以省略)。
对类别型特征进行独热编码 (OneHotEncoder)，并设置 handle_unknown='ignore'。

模型管道:
构建一个完整的 Pipeline，将 preprocessor 和一个 RandomForestRegressor 模型串联起来。

评估逻辑:
包含分割数据、训练模型、进行预测以及计算并打印均方根误差 (RMSE)和R²分数的逻辑。
注意: 本次挑战不要求创建交互特征，核心是考察你在一个全新、真实的数据集上，快速搭建标准工作流的能力。
"""
from scipy.stats import randint
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- 1. 数据加载与清洗 ---
try:
  df = pd.read_csv('./clean_df.csv')
except FileNotFoundError:
  print('没有找到数据文件')
  exit()

keep_column = ['price','wheel-base', 'curb-weight', 'engine-size', 'horsepower', 'city-mpg', 'make', 'body-style', 'drive-wheels', 'engine-type']
clean_df = df[keep_column].copy()

clean_df.dropna(inplace=True)

X = clean_df.drop('price', axis=1)
y = clean_df['price']

# --- 2. 特征工程 (优化点) ---

# 特征选择
numeric_features = ['wheel-base', 'curb-weight', 'engine-size', 'horsepower', 'city-mpg']
categorical_features = ['make', 'body-style', 'drive-wheels', 'engine-type']

# --- 3. 预处理构建 ---
transformers = [
  ('num', Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ]), numeric_features),
  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
]
preprocesser = ColumnTransformer(transformers)

# --- 4. 数据分割 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 5. 训练与评估 ---
lr_pipeline = Pipeline(steps=[
  ('preprocessor', preprocesser),
  ('Regressor', RandomForestRegressor(random_state=42))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("--- 线性回归模型评估结果 ---")
print(f"均方根误差 (RMSE): {rmse_lr:.4f}")
print(f"R² 分数: {r2_lr:.4f}")


# 6. 定义要调优的模型
rfr = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
  estimator=rfr,
  param_distributions={
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_leaf': randint(2, 20), # 扩大范围
    'min_samples_split': randint(5, 40) # 增加一个参数
  },
  n_iter=100, 
  cv=5, 
  random_state=42, 
  n_jobs=-1,
  scoring='neg_mean_squared_error' 
)

rf_pipeline_tuned = Pipeline(steps=[
   ('preprocessor', preprocesser),
   ('regressor', random_search)
])

rf_pipeline_tuned.fit(X_train, y_train)

best_params = random_search.best_params_
print('best_params', random_search.best_params_)

final_model = RandomForestRegressor(random_state=42, **best_params)

final_pipeline  = Pipeline(steps=[
  ('preprocessor', preprocesser),
  ('regressor', final_model)
])

final_pipeline.fit(X_train, y_train)
y_pred_lr = final_pipeline.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("--- 线性回归模型评估结果 ---")
print(f"均方根误差 (RMSE): {rmse_lr:.4f}")
print(f"R² 分数: {r2_lr:.4f}")

