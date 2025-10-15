# 1.数据加载与清洗
from random import randint
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
  df = pd.read_csv('./US Airline Flight Routes and Fares 1993-2024.csv')
except FileNotFoundError:
  print("错误：请确保 'US Airline Flight Routes and Fares.csv' 文件在当前目录下。")
  exit()
  
# 根据我们的分析，选择有用的列
columns_to_keep = [
    'quarter', 'nsmiles', 'passengers', 'airport_1', 
    'airport_2', 'carrier_lg', 'fare'
]
df_clean = df[columns_to_keep].copy()
df_clean.dropna(inplace=True)

# 分离特征 (X) 和目标 (y)
X = df_clean.drop('fare', axis=1)
y = df_clean['fare']

# --- 2. 特征工程 (优化点) ---
X['route'] = X['airport_1'] + '-' + X['airport_2']
X= X.drop(['airport_1', 'airport_2'], axis=1)


# 3. 定义预处理流程
# 更新特征列表
numeric_features = ['quarter', 'nsmiles', 'passengers']
categorical_features = ['carrier_lg', 'route'] # 使用新的'Route'特征
transformers = [
  ('num', Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ]), numeric_features),
  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
]
preprocessor = ColumnTransformer(transformers)

# 4. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 5. 训练与评估
lr_pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)


# 6. 定义要调优的模型
rfr = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
  estimator=rfr,
  param_distributions={
    'n_estimators': randint(100, 250),
    'max_depth': randint(10, 30),
    'min_samples_leaf': randint(5, 15)
  },
  n_iter=10, 
  cv=3, 
  random_state=42, 
  n_jobs=-1,
  scoring='neg_mean_squared_error' 
)


rf_pipeline_tuned = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', random_search)
])

rf_pipeline_tuned.fit(X_train, y_train)

# 1. 获取最佳参数字典
best_params = random_search.best_params_
print("找到的最佳参数是: ", best_params)

final_model = RandomForestRegressor(random_state=42, **best_params)

# 3. 在我们的Pipeline工作流中，创建一个最终的管道
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', final_model)
])

# 4. 在全部训练数据上训练这辆“最终版”赛车
final_pipeline.fit(X_train, y_train)

# 5. 用这个最终管道进行预测和部署
y_pred = final_pipeline.predict(X_test)
joblib.dump(final_pipeline, 'final_model.joblib') # 保存最终模型