import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

# 엑셀 파일 경로
file_path = "선박, SCFI, KCCI 데이터 파일_Ver2(링카고).xlsx"

# SCFI운임지수 데이터 로드
# SCFI 지수 데이터가 일주일 간격
scfi_data = pd.read_excel(file_path, sheet_name="SCFI (상하이항->)")
scfi_data['Date'] = pd.to_datetime(scfi_data['Unnamed: 0'])  # 날짜 데이터 처리
scfi_data['Year'] = scfi_data['Date'].dt.year
scfi_data['Month'] = scfi_data['Date'].dt.month
scfi_data['Day'] = scfi_data['Date'].dt.day
scfi_data.drop(scfi_data.columns[2:17], axis=1, inplace=True)
scfi_data.drop("Unnamed: 0", axis=1, inplace=True)
# print(scfi_data)

# 과거 n주의 SCFI 운임 지수 생성
n_lags = 5

for i in range(1, n_lags + 1):
    scfi_data[f'SCFI_lag_{i}'] = scfi_data['SCFI운임지수'].shift(i)

scfi_data.dropna(inplace=True)
# print(scfi_data)

# SCFI 모델 학습
y_scfi = scfi_data['SCFI운임지수']
X_scfi = scfi_data.drop(['Date', 'SCFI운임지수'], axis=1)

X_scfi_train, X_scfi_test, y_scfi_train, y_scfi_test = train_test_split(X_scfi, y_scfi, test_size=0.1, random_state=42)

model_scfi = RandomForestRegressor(n_estimators=100, random_state=42)
model_scfi.fit(X_scfi_train, y_scfi_train)

y_scfi_pred = model_scfi.predict(np.array([[2024, 9, 20, 2510.95, 2726.58, 2963.38, 3097.63, 3281.36]]))  # SCFI 모델로 예측
# print(y_scfi_pred)

# print(sum(abs(y_scfi_train - y_scfi_pred)) / len(X_scfi_train))

with open("model_scfi.pkl", "wb") as f :
    pickle.dump(model_scfi, f)
    
print("모델 생성 및 저장 성공")