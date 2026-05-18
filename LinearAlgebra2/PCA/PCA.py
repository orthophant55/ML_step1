# %% [markdown]
# # 선형대수학 응용 - 주성분분석(PCA)
#
# ## 목표
# PCA(Principal Component Analysis)를 직접 구현하면서 다음 개념을 이해한다.
#
# - 평균 중심화
# - 공분산 행렬
# - 고유값과 고유벡터
# - 주성분
# - 차원 축소
# - 설명 분산 비율
#
# PCA의 핵심 질문:
#
# > 데이터를 낮은 차원으로 정사영할 때,  
# > 원래 데이터의 구조를 가장 잘 보존하는 방향은 무엇인가?
#
# PCA는 데이터의 분산이 가장 큰 방향을 첫 번째 주성분으로 선택한다.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# %% [markdown]
# ## 1. 예제 데이터 만들기
#
# PCA를 직관적으로 보기 위해 2차원 데이터를 만든다.
# 두 feature가 양의 상관관계를 가지도록 구성한다.

# %%
n_samples = 100

x = np.random.normal(50, 10, n_samples)
y = 0.7 * x + np.random.normal(0, 5, n_samples)

X = np.column_stack([x, y])

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df.head()

# %%
plt.figure(figsize=(6, 6))
plt.scatter(df["Feature 1"], df["Feature 2"], alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Original Data")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. 평균 중심화
#
# PCA에서는 먼저 각 feature의 평균을 0으로 맞춘다.
#
# 이유:
#
# - PCA는 데이터의 퍼짐, 즉 분산 방향을 찾는 방법이다.
# - 데이터가 원점 중심에 있지 않으면 방향 해석이 왜곡될 수 있다.
# - 따라서 각 feature에서 평균을 빼서 데이터 중심을 원점으로 이동시킨다.

# %%
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

print("각 feature의 평균:")
print(X_mean)

print("\n평균 중심화 후 평균:")
print(np.mean(X_centered, axis=0))

# %%
plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Centered Feature 1")
plt.ylabel("Centered Feature 2")
plt.title("Mean Centered Data")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 3. 공분산 행렬 계산
#
# 공분산 행렬은 feature들이 서로 얼마나 함께 변하는지를 나타낸다.
#
# 데이터 행렬을 X라고 할 때, 평균 중심화된 데이터의 공분산 행렬은 다음과 같다.
#
# $$
# \Sigma = \frac{1}{n-1} X^T X
# $$
#
# 여기서:
#
# - 대각 성분: 각 feature의 분산
# - 비대각 성분: 두 feature 사이의 공분산

# %%
cov_matrix = np.cov(X_centered, rowvar=False)

print("공분산 행렬:")
print(cov_matrix)

# %% [markdown]
# ## 4. 고유값과 고유벡터 구하기
#
# PCA는 공분산 행렬의 고유값과 고유벡터를 사용한다.
#
# $$
# \Sigma v = \lambda v
# $$
#
# 여기서:
#
# - $v$: 고유벡터, 데이터가 퍼지는 방향
# - $\lambda$: 고유값, 해당 방향으로의 분산 크기
#
# 고유값이 클수록 그 방향이 데이터를 더 잘 설명한다.

# %%
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("고유값:")
print(eigenvalues)

print("\n고유벡터:")
print(eigenvectors)

# %% [markdown]
# ## 5. 고유값 기준으로 정렬하기
#
# PCA에서는 고유값이 큰 순서대로 고유벡터를 정렬한다.
# 가장 큰 고유값에 대응하는 고유벡터가 첫 번째 주성분이다.

# %%
sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("정렬된 고유값:")
print(eigenvalues_sorted)

print("\n정렬된 고유벡터:")
print(eigenvectors_sorted)

# %% [markdown]
# ## 6. 주성분 방향 시각화
#
# 첫 번째 주성분은 데이터의 분산이 가장 큰 방향이다.
# 두 번째 주성분은 첫 번째 주성분과 직교하면서 남은 분산을 가장 잘 설명하는 방향이다.

# %%
plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5)

origin = np.array([0, 0])

for i in range(len(eigenvalues_sorted)):
    vector = eigenvectors_sorted[:, i] * np.sqrt(eigenvalues_sorted[i]) * 2
    plt.arrow(
        origin[0], origin[1],
        vector[0], vector[1],
        head_width=1.5,
        length_includes_head=True
    )
    plt.text(vector[0] * 1.1, vector[1] * 1.1, f"PC{i+1}", fontsize=12)

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Centered Feature 1")
plt.ylabel("Centered Feature 2")
plt.title("Principal Component Directions")
plt.axis("equal")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 7. 2차원 데이터를 1차원으로 차원 축소하기
#
# 첫 번째 주성분 방향으로 데이터를 정사영하면 2차원 데이터를 1차원으로 줄일 수 있다.
#
# $$
# Z = XW
# $$
#
# 여기서:
#
# - $X$: 평균 중심화된 원본 데이터
# - $W$: 선택한 주성분 벡터
# - $Z$: 차원 축소된 데이터

# %%
W = eigenvectors_sorted[:, :1]

X_pca_1d = X_centered @ W

print("원본 데이터 shape:", X_centered.shape)
print("PCA 변환 후 shape:", X_pca_1d.shape)

pd.DataFrame(X_pca_1d, columns=["PC1"]).head()

# %% [markdown]
# ## 8. 1차원으로 줄인 데이터를 다시 2차원 공간에 복원해보기
#
# PCA로 1차원으로 줄이면 정보 일부는 사라진다.
# 그래도 첫 번째 주성분이 원래 데이터의 구조를 최대한 보존한다.

# %%
X_reconstructed = X_pca_1d @ W.T

plt.figure(figsize=(6, 6))
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.4, label="Original Centered Data")
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.8, label="Projected onto PC1")

for i in range(n_samples):
    plt.plot(
        [X_centered[i, 0], X_reconstructed[i, 0]],
        [X_centered[i, 1], X_reconstructed[i, 1]],
        linewidth=0.5,
        alpha=0.3
    )

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)
plt.xlabel("Centered Feature 1")
plt.ylabel("Centered Feature 2")
plt.title("Projection onto First Principal Component")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 9. 설명 분산 비율
#
# 각 주성분이 전체 분산 중 얼마를 설명하는지 계산한다.
#
# $$
# \text{Explained Variance Ratio}
# =
# \frac{\lambda_i}{\sum_j \lambda_j}
# $$

# %%
explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)

for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1} 설명 분산 비율: {ratio:.4f}")

print("\n누적 설명 분산 비율:")
print(np.cumsum(explained_variance_ratio))

# %%
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.plot(
    range(1, len(explained_variance_ratio) + 1),
    np.cumsum(explained_variance_ratio),
    marker="o"
)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 10. sklearn PCA와 비교하기
#
# 직접 구현한 PCA 결과와 sklearn의 PCA 결과를 비교한다.

# %%
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_centered)

print("sklearn PCA 주성분:")
print(pca.components_)

print("\nsklearn PCA 고유값:")
print(pca.explained_variance_)

print("\nsklearn PCA 설명 분산 비율:")
print(pca.explained_variance_ratio_)

# %% [markdown]
# 직접 구현한 결과와 sklearn 결과는 부호가 반대일 수 있다.
#
# 고유벡터는 방향이 중요하기 때문에,
#
# $$
# v
# $$
#
# 와
#
# $$
# -v
# $$
#
# 는 같은 축을 의미한다.
#
# 따라서 부호가 다르더라도 같은 주성분 방향을 나타내는 것이다.

# %% [markdown]
# ## 11. Iris 데이터에 PCA 적용하기
#
# 이번에는 실제 데이터셋인 Iris 데이터에 PCA를 적용해본다.
# 4차원 feature를 2차원으로 줄여 시각화한다.

# %%
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names

print("원본 Iris 데이터 shape:", X_iris.shape)

# %%
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

print("PCA 변환 후 shape:", X_iris_pca.shape)
print("설명 분산 비율:", pca_iris.explained_variance_ratio_)
print("누적 설명 분산 비율:", np.sum(pca_iris.explained_variance_ratio_))

# %%
plt.figure(figsize=(7, 5))

for target, name in enumerate(target_names):
    plt.scatter(
        X_iris_pca[y_iris == target, 0],
        X_iris_pca[y_iris == target, 1],
        label=name,
        alpha=0.8
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 12. 정리
#
# PCA의 전체 흐름은 다음과 같다.
#
# 1. 데이터 평균 중심화
# 2. 공분산 행렬 계산
# 3. 공분산 행렬의 고유값, 고유벡터 계산
# 4. 고유값이 큰 순서대로 고유벡터 정렬
# 5. 원하는 개수의 주성분 선택
# 6. 원본 데이터를 주성분 방향으로 정사영
#
# PCA는 단순히 차원을 줄이는 기법이 아니라,
# 데이터의 분산을 가장 잘 보존하는 새로운 좌표축을 찾는 방법이다.