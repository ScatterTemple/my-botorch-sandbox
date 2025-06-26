import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# データ生成
np.random.seed(0)
N = 500
X = np.random.uniform(-1, 1, size=(N, 2))  # 2次元入力: x1, x2
y = X[:, 0]**2 + 2 * X[:, 1]**2            # 出力: y = x1² + 2 * x2²

# 特徴量をスケーリング（推奨）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SelectKBest を使って上位 2 特徴量を選択（実際には全てのスコアを確認可能）
selector = SelectKBest(score_func=f_regression, k=2)
selector.fit(X_scaled, y)

# スコア（F値）と p 値を出力
print("Feature scores (F-values):", selector.scores_)
print("Feature p-values:", selector.pvalues_)

# 各特徴量に対応する名前（任意で追加）
feature_names = ['x1', 'x2']
for name, score, p in zip(feature_names, selector.scores_, selector.pvalues_):
    print(f"{name}: score = {score:.4f}, p-value = {p:.4e}")
