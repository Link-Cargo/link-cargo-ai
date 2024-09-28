import pandas as pd
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

models = {}

path = "split_and_sort"
file_list = os.listdir(path)
# print(file_list)

# 선박 데이터 로드
shipping_data = pd.read_excel("선박, SCFI, KCCI 데이터 파일_Ver2(링카고).xlsx", sheet_name=0)

# "DATE" 열을 datetime 객체로 변환 후 YearMonth로 전처리
shipping_data["DATE"] = pd.to_datetime(shipping_data["DATE"])
shipping_data['YearMonth'] = shipping_data["DATE"].dt.to_period("M")
shipping_data = shipping_data.drop(["DATE"], axis=1)

for file in file_list :
    # 항로별 비용 데이터 로드
    data = pd.read_csv(f"{path}/{file}")
    data["발효일"] = pd.to_datetime(data["발효일"]) 
    
    # 항로 정보 추출
    fairway = data["항로"].unique()[0]
    
    # 불필요한 열 제거
    data = data.drop(["항로", "선적지", "컨테이너 종류", "컨테이너 크기", "화물품목"], axis=1)

    data['YearMonth'] = data["발효일"].dt.to_period("M")

    # 같은 YearMonth 별 모든 양륙지의 "금액" 평균값
    grouped = data.groupby("YearMonth").agg(Avg_fare = ("OF 금액", "mean")).reset_index() 

    # ======================================================================
    
    # 선박데이터 복제
    shipping_data_copy = shipping_data.copy(deep=True)
    # 항로별 비용 데이터 첫번째 YearMonth 포함 이후의 데이터들만 남김
    shipping_data_copy = shipping_data_copy[shipping_data_copy.YearMonth >= str(grouped["YearMonth"][0])] \
                         .reset_index(drop = True) 
    
    # 전처리 (결측치 -> 평균값으로 / '월' 정보만 추출)
    shipping_data_copy.fillna(shipping_data_copy.mean(numeric_only=True), inplace=True)
    shipping_data_copy['Month'] = shipping_data_copy['YearMonth'].dt.month
    
    # ======================================================================
    merged_data = pd.merge(grouped, shipping_data_copy, on="YearMonth", how="inner")
    
    # 모델 제작
    train_targets = merged_data["Avg_fare"]
    train_data = merged_data.drop(["YearMonth", "Avg_fare"], axis=1)

    # 데이터 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_data)
    
    # Month 데이터만 분리
    month = X_scaled[:, -1]
    shipping_features = X_scaled[:, :-1]
    
    # PCA 적용 (선박정보 데이터를 2개의 feature로 압축)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(shipping_features)
    combined_features = np.column_stack((X_pca, month))
    
    # 다중선형회귀 모델 학습
    linear_reg = LinearRegression()
    linear_reg.fit(combined_features, train_targets)

    # 데이터가 적을 때만 사용
    # 검증 데이터나 테스트 데이터가 없으므로 학습 데이터의 오차 평균을 사용하기 위함
    y_pred = linear_reg.predict(combined_features) 
    error = np.sum((abs(train_targets - y_pred)))/len(y_pred)
    
    models[fairway] = {
        "Model" : linear_reg,
        "Scaler" : scaler,
        "PCA" : pca,
        "Error" : error
    }
    
# print(models)
with open("models.pkl", "wb") as f :
    pickle.dump(models, f)
    
print("모델 생성 및 저장 성공")