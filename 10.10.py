import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


"""
你收到一个flight_delays.csv文件，包含字段 [出发城市, 到达城市, 航空公司, 是否延误, 飞行时长, 乘客评分]。其中部分数据缺失。请你完成以下任务：
用“平均值”填充飞行时长的缺失值 。
对出发城市、到达城市、航空公司进行独热编码 。
使用处理后的数据训练一个逻辑回归模型，预测是否延误（1为延误，0为未延误） 。
输出该模型在测试集上的精确率(Precision)和召回率(Recall) 

加载 -> 清洗 -> 编码 -> 分割 -> 训练 -> 评估。
"""

df = pd.read_csv('flight_delays.csv')

# 1. 创建目标变量（y）
df['IsDelayed'] = (df['DelayMinutes'] > 0).astype(int) # 1表示延误，0表示未延误

# 2. 从时间戳中提取特征
df['ScheduledDeparture'] = pd.to_datetime(df['ScheduledDeparture'])
df['ScheduledArrival'] = pd.to_datetime(df['ScheduledArrival'])
df['DepartureHour'] = df['ScheduledDeparture'].dt.hour
df['DayOfWeek'] = df['ScheduledDeparture'].dt.dayofweek

# 计算计划飞行时长（单位：分钟）
df['ScheduledDurationMinutes'] = (df['ScheduledArrival'] - df['ScheduledDeparture']).dt.total_seconds() / 60

# 3. 选择特征列
features_to_use = ['Airline', 'Origin', 'Destination', 'Distance', 'DepartureHour', 'DayOfWeek', 'ScheduledDurationMinutes']
X = df[features_to_use]
y = df['IsDelayed']

# --- 步骤2: 定义预处理流程 ---
# 定义需要不同处理的列
numeric_features = ['Distance', 'DepartureHour', 'DayOfWeek', 'ScheduledDurationMinutes']
categorical_features = ['Airline', 'Origin', 'Destination']

# 创建预处理器
transformers=[
        ('num', Pipeline(steps=[
           ('imputer', SimpleImputer(strategy='median')),
           ('scaler', StandardScaler())
          ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
preprocessor = ColumnTransformer(transformers)

# --- 步骤3: 数据分割 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- 步骤4: 创建并训练模型 ---
# 将预处理器和模型串联成一个完整的管道
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# 训练模型
model_pipeline.fit(X_train, y_train)

# --- 步骤5: 评估模型 ---
y_pred = model_pipeline.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0) # zero_division防止因没有预测正例而报错
recall = recall_score(y_test, y_pred, zero_division=0)

print("\n--- 模型评估结果 ---")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")

# # 定义要保存的文件名
# model_filename = 'flight_delay_predictor.joblib'

# # 使用 joblib.dump 保存整个管道
# print(f"\n--- 正在将训练好的模型打包至 {model_filename} ---")
# joblib.dump(model_pipeline, model_filename)
# print("--- 打包完成！---")