import numpy as np
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
import shap
import matplotlib.pyplot as plt

# 1. 問題の定義
problem = {
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[-1.0, 1.0], [-1.0, 1.0]]
}


# 2. ブラックボックス関数
def black_box(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return x1 ** 1 + 2 * x2 ** 1


# ---- Sobol 感度分析 ----
print("\n--- Sobol Sensitivity Analysis ---")
X_sobol = saltelli.sample(problem, 100, calc_second_order=True)
Y_sobol = black_box(X_sobol)
sobol_result = sobol.analyze(problem, Y_sobol, print_to_console=True)

# ---- Morris 感度分析 ----
print("\n--- Morris Sensitivity Analysis ---")
X_morris = morris_sampler.sample(problem, N=100, num_levels=4)
Y_morris = black_box(X_morris)
morris_result = morris.analyze(problem, X_morris, Y_morris, conf_level=0.95, print_to_console=True)

# ---- Shapley 感度分析 (SHAP) ----
print("\n--- SHAP Sensitivity Analysis ---")
# ランダムなサンプルを生成
X_shap = np.random.uniform(-1, 1, size=(100, 2))
Y_shap = black_box(X_shap)

# ラッパー関数にして SHAP に渡す
explainer = shap.Explainer(lambda x: black_box(x), X_shap)
shap_values = explainer(X_shap)
# プロット
# shap.plots.beeswarm(shap_values)

# 各特徴量のSHAP値の絶対値の平均(つまり、各特徴量の予測に対する寄与度)を棒グラフで表示
# 予測に対する寄与度が大きい順にソートされて表示される
shap.plots.bar(shap_values)  # max_displayで表示する変数の最大数を指定


# shap.plots.waterfall(shap_values[0])
