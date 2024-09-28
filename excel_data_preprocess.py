import pandas as pd

# fare_to_fairway.xlsx 파일 읽어오기
file_path = 'fare_to_fairway.xlsx'
df = pd.read_excel(file_path)

# "컨테이너 크기" 열에서 "40 feet" 값을 가진 행 제거
df_filtered = df[df["컨테이너 크기"] != "40 feet"]

# "화물품목" 열에서 "일반화물"이 아닌 값을 가진 행 제거
df_filtered = df_filtered[df_filtered["화물품목"] == "일반화물"]

# "컨테이너 종류" 열에서 "dry"가 아닌 값을 가진 행 제거
df_filtered = df_filtered[df_filtered["컨테이너 종류"] == "dry"]

# "선적지" 열에서 "부산"이 아닌 값을 가진 행 제거
df_filtered = df_filtered[df_filtered["선적지"] == "부산"]

# 데이터 수가 부족한 "항로"는 삭제
# 분석 결과 - 기타, 남태평양, 한러, 한일
# 추후 데이터셋 보강에 따라 수정할 수 있습니다.
fairway = df_filtered[
            (df_filtered["항로"] == "기타") |
            (df_filtered["항로"] == "남태평양") |
            (df_filtered["항로"] == "한러") |
            (df_filtered["항로"] == "한일")
        ].index

df_filtered.drop(fairway, inplace=True)

# print(df_filtered)

# 결과를 새 CSV 파일로 저장
df_filtered.to_csv('fare_to_fairway_filterd.csv', index=False)  # 새 파일로 저장
# df_filtered.to_excel(file_path, index=False)  # 기존 파일 덮어쓰기