import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

file_path = "선박, SCFI, KCCI 데이터 파일_Ver2(링카고).xlsx"

# KCCI 데이터 로드 (5번째 시트)
# KCCI 지수 데이터가 일주일 간격
kcci_data = pd.read_excel(file_path, sheet_name="KCCI (부산->)")
kcci_data["DATE"] = kcci_data["DATE"].astype('str')
kcci_data['Date'] = pd.to_datetime(kcci_data['DATE'])
kcci_data['Year'] = kcci_data['Date'].dt.year
kcci_data['Month'] = kcci_data['Date'].dt.month
kcci_data['Day'] = kcci_data['Date'].dt.day
kcci_data.drop(kcci_data.columns[2:16], axis=1, inplace=True)
kcci_data.drop("DATE", axis=1, inplace=True)

kcci_data = kcci_data.apply(lambda x: x.str.replace(',', '') if x.dtype == 'object' else x)
kcci_data = kcci_data.apply(pd.to_numeric, errors='coerce')
# print(kcci_data)

# 과거 주의 KCCI 지수 생성
n_lags = 5

for i in range(1, n_lags + 1):
    kcci_data[f'KCCI_lag_{i}'] = kcci_data['KCCI'].shift(i)

kcci_data.dropna(inplace=True)

# print(kcci_data)

# KCCI 모델 학습
y_kcci = kcci_data['KCCI']
X_kcci = kcci_data.drop(['KCCI'], axis=1)
X_kcci_train, X_kcci_test, y_kcci_train, y_kcci_test = train_test_split(X_kcci, y_kcci, test_size=0.1, random_state=42)

model_kcci = RandomForestRegressor(n_estimators=100, random_state=42)
model_kcci.fit(X_kcci_train, y_kcci_train)

## KCCI 예측 및 성능 평가
# y_kcci_pred = model_kcci.predict(X_kcci_test)
# kcci_mse = mean_squared_error(y_kcci_test, y_kcci_pred)
# print(f'KCCI Model Mean Squared Error: {kcci_mse}')

# y_kcci_pred = model_kcci.predict(np.array([[2024, 9, 30, 4313, 4467, 4530, 4659, 4673]]))  # SCFI 모델로 예측
# print(y_kcci_pred)

with open("model_kcci.pkl", "wb") as f :
    pickle.dump(model_kcci, f)
    
print("모델 생성 및 저장 성공")