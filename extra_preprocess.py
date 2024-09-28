import pandas as pd

# 엑셀 파일 읽어오기
file_path = 'fare_to_fairway_filterd.csv'  # 필터링된 데이터 파일 경로를 지정
df = pd.read_csv(file_path)

# "항로" 열에서 unique 값 가져오기
routes = df["항로"].unique()
# print(routes)

# 각 항로별로 파일 나누기
for route in routes:
    df_route = df[df["항로"] == route] # 항로별 분리
    df_sorted = df_route.sort_values(by="발효일", ascending=True) # "발효일" 기준 오름차순 정리
    
    output_file = f"split_and_sort/{route}.csv" # 각 범주의 이름으로 파일 생성
    
    df_sorted.to_csv(output_file, index=False)