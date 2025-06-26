import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
import shap
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer
import shap
import matplotlib.pyplot as plt


# ブラックボックス関数（例）
def black_box(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (
            1 * x1 ** 1
            + 2 * x2 ** 1
    )


# 1. 問題の定義（変数名と範囲）
problem = {
    'num_vars': 2,
    'names': ['x1', 'x2'],
    'bounds': [[-1.0, 1.0], [-1.0, 1.0]]
}

# 2. 既存の入力データと出力データ（100個）
np.random.seed(42)
n = 20
# x_sample = np.random.uniform(-1, 1, size=(n, 2))
x_sample = np.random.randn(n, 2) * 0.1

y_sample = black_box(x_sample)

# # 3. 代理モデルの学習（ここではランダムフォレスト）
# model = RandomForestRegressor()

# 3a. Gaussian Process モデルの学習
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

model.fit(x_sample, y_sample)

# Sobol
print("\n--- Sobol Sensitivity Analysis ---")
param_values = saltelli.sample(problem, 1024, calc_second_order=True)
y_pred = model.predict(param_values)
sobol_result = sobol.analyze(problem, y_pred, calc_second_order=True, print_to_console=True)

# Morris
print("\n--- Morris Sensitivity Analysis ---")
param_values = morris_sampler.sample(problem, N=1024, num_levels=4)
y_pred = model.predict(param_values)
morris_result = morris.analyze(problem, param_values, y_pred, conf_level=0.95, print_to_console=True)

# SHAP
# 1. SHAP explainer の作成
# explainer = shap.Explainer(model)  # TreeExplainerが自動選択される
explainer = shap.KernelExplainer(model.predict, shap.sample(x_sample, 20))
shap_values = explainer(x_sample)

# 2. SHAP summary プロット（平均的な寄与の可視化）
# shap.summary_plot(shap_values, x_sample, feature_names=problem["names"])
# 各特徴量のSHAP値の絶対値の平均(つまり、各特徴量の予測に対する寄与度)を棒グラフで表示
# 予測に対する寄与度が大きい順にソートされて表示される
shap.plots.bar(shap_values.values)  # max_displayで表示する変数の最大数を指定

