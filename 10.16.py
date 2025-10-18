"""
第一题 (数据结构与算法)

主题：热门航线统计

给定一个航线列表 routes，其中每个元素是形如 "JFK-LAX" 的字符串。你的任务是找出出现频率最高的 k 条航线。

示例: routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"], k = 2 输出: ["JFK-LAX", "SFO-JFK"] (顺序不重要)

提示: 这个问题考察对哈希表（字典）和堆（优先队列）的综合运用。
"""

import collections
import heapq

def top_k_frequent_routes(routes, k):
    if not routes or k == 0:
        return []
     
    # 1. 哈希表计数 (标准、直观的方式)
    freq_map = collections.Counter(routes)
    # 示例: freq_map = {'JFK-LAX': 3, 'SFO-JFK': 2, 'JFK-ORD': 1}
    print(freq_map)
    # 2. 维护一个大小为 k 的最小堆 (擂主榜)
    # 堆中存放元组 (频率, 航线)
    min_heap = []
     
    for route, freq in freq_map.items():
        if len(min_heap) < k:
            heapq.heappush(min_heap, (freq, route))
        else:
            if freq > min_heap[0][0]:
                # 新来的更强，踢掉最弱的，自己上
                heapq.heapreplace(min_heap, (freq, route))
            
    # 3. 提取结果
    # 此时堆里就是频率最高的 k 个元素
    # 我们只需要航线名，不需要频率
    result = [route for freq, route in min_heap]
    
    return result      
    
    
      
routes = ["JFK-LAX", "JFK-ORD", "SFO-JFK", "JFK-LAX", "SFO-JFK", "JFK-LAX"]  
res = top_k_frequent_routes(routes, 2)
print(res)    
             
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

