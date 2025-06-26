import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

# 問題の定義
problem = {
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[-1, 1],   # x1の範囲
               [-1, 1]]   # x2の範囲
}

# 評価する関数
def evaluate(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return x1**2 + 2 * x2**2

# サンプリング
param_values = saltelli.sample(problem, 100, calc_second_order=False)

# 出力を計算
Y = evaluate(param_values)

# Sobol感度解析（ANOVA分解）
Si = sobol.analyze(problem, Y, calc_second_order=False)

# 結果の表示
print("First-order sensitivity indices (S1):")
for name, s1 in zip(problem['names'], Si['S1']):
    print(f"{name}: {s1:.4f}")

print("\nTotal-order sensitivity indices (ST):")
for name, st in zip(problem['names'], Si['ST']):
    print(f"{name}: {st:.4f}")
