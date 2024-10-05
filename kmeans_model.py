from sklearn.cluster import KMeans
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 더미 데이터 불러오기
df = pd.read_csv("dummy_data_for_kmeans.csv")

# 모델 제작 및 학습
kmeans = KMeans(n_clusters=5, random_state=100) # 5개의 클러스터
kmeans.fit(df)

labels = kmeans.labels_

# 데이터에 속한 cluster 붙이기
df['cluster'] = kmeans.labels_

# 그래프 그리기
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

x = df["product_name"]
y = df["country_import"]
z = df["seasons"]

ax.scatter(x, y, z, c=labels, cmap='prism', s=45)

# 축 설정
ax.set_xlabel('product_categories')
ax.set_ylabel('country_import')
ax.set_zlabel('seasons')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["Elec.", "Foods", "Toy", "Clothes"])

ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(["N.A.", "Europe", "M.E.", "China", "S.E.A."])

ax.set_zticks([0, 1, 2, 3])
ax.set_zticklabels(["Spring", "Summer", "Autumn", "Winter"])

# 그래프 보여주기
plt.show()