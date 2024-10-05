import numpy as np
import pandas as pd

np.random.seed(1000)

n_samples = 2000

# 더미데이터 생성 (최근 수출 정보 가정)

# "상품명" - (4 카테고리: 전자제품, 식품, 장난감, 의류) 레이블은 각각 0, 1, 2, 3
product_name = np.random.choice([0, 1, 2, 3], size=n_samples)

# "수입국" -  (5 카테고리: 북미, 유럽, 중동, 중국, 동남아) 레이블은 각각 0, 1, 2, 3, 4
country_import = np.random.choice([0, 1, 2, 3, 4], size=n_samples)

# "선적일" -  (4 카테고리 : 봄, 여름, 가을, 겨울) 레이블은 각각 0, 1, 2, 3
# 1년간 서비스 하였다고 가정
shipment_date_raw = pd.to_datetime(np.random.choice(pd.date_range("2024-10-01", "2025-11-01"), size=n_samples))
seasons = shipment_date_raw.month % 12 // 3

df = pd.DataFrame({
    "product_name": product_name,
    "country_import": country_import,
    "seasons": seasons
})

# print(df.head(5))

df.to_csv('dummy_data_for_kmeans.csv', index=None)