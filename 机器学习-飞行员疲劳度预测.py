import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

"""
主题：飞行员疲劳度预测

你需要构建一个模型，根据飞行员的工作日志数据，预测其在下一次飞行前的疲劳度等级 (FatigueLevel, 分为 'Low', 'Medium', 'High' 三类)。这是一个多分类问题。

任务要求：

特征选择:

数值特征: [TotalFlightHours, HoursSinceLastFlight, NumSectorsFlown]

类别特征: [AircraftType, DepartureTimeOfDay, RouteComplexity] ('Low', 'Medium', 'High')

特别注意: 类别特征 RouteComplexity 具有内在顺序 ('Low' < 'Medium' < 'High')。

构建预处理管道:

创建ColumnTransformer。

对数值特征使用StandardScaler。

对普通类别特征 (AircraftType, DepartureTimeOfDay) 使用OneHotEncoder。

对有序类别特征 (RouteComplexity) 使用**OrdinalEncoder**，并确保编码顺序正确 (categories=[['Low', 'Medium', 'High']])。

模型训练:

使用决策树分类器 (DecisionTreeClassifier) 进行训练。

模型评估:

报告准确率(Accuracy)和F1分数(F1-Score, weighted)。
"""

# --- 1. 数据加载与清洗 ---
try:
    # 假设CSV文件名是 'fatigue_data.csv'
    df = pd.read_csv('./fatigue_data.csv') 
except FileNotFoundError:
    print("错误：请确保数据文件在当前目录下。")
    exit()

df.dropna(inplace=True)

# b) 选择特征列和目标列
feature_cols = ['TotalFlightHours', 'HoursSinceLastFlight', 'NumSectorsFlown', 'AircraftType', 'DepartureTimeOfDay', 'RouteComplexity']
target_cols = ['FatigueLevel']

keep_cols = feature_cols + target_cols
clean_df = df[keep_cols].copy()

X = clean_df.drop('FatigueLevel', axis=1)
y = clean_df['FatigueLevel']

# --- 2. 定义特征列表 ---
numeric_features = ['TotalFlightHours', 'HoursSinceLastFlight', 'NumSectorsFlown']
nominal_categorical_features = ['AircraftType', 'DepartureTimeOfDay']
ordinal_categorical_feature = ['RouteComplexity']
# 定义有序类别的顺序
route_complexity_order = ['Low', 'Medium', 'High']

# --- 3. 定义预处理流程 ---
preprocessor = ColumnTransformer(transformers=[
  # --- 处理数值特征 ---
  ('num', Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ]), numeric_features),
  
  # --- 处理普通类别特征 ---
  ('cat', Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('oneHot', OneHotEncoder(handle_unknown='ignore'))
  ]), nominal_categorical_features),
  
  # --- 处理有序类别特征 ---
  ('cat_ord', Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('ordinal', OrdinalEncoder(categories=[route_complexity_order])) 
  ]), ordinal_categorical_feature) 
],
remainder='passthrough'
)

# --- 4. 分割数据 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=45)

model_pipeline = Pipeline(steps=[
  ('proprecessor', preprocessor),
  ('feature_selection', SelectKBest(score_func=f_classif, k=20)), 
  ('classifier', DecisionTreeClassifier(random_state=42))
])

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"F1 分数 (Weighted): {f1:.4f}")





