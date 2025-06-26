import datetime

import numpy as np
from torch import tensor
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll


# ===== システムの定義 =====
def problem(x_: np.ndarray) -> np.ndarray:
    """Calc the output.

    Args:
        x_ (np.ndarray): (n, d) shaped np.ndarray.

    Returns:
        np.ndarray: The (n, 1) shaped output value.

    """

    # 1 次
    return (x_ * np.arange(1, 1 + len(x_[0])).reshape(1, -1)).sum(axis=1, keepdims=True)

    # 2 次
    # return (x_ ** 2 * np.arange(1, 1 + len(x_[0])).reshape(1, -1)).sum(axis=1, keepdims=True)


# ===== 問題設定 =====
dim = 3  # 入力の次元
n = 100  # サンプル数
# p = 6  # minimum(p), dim <= p + 1
# n = p ** 2   # サンプル数
# bounds = [[0, 1], [-10, 10], [0, 100]]  # 入力の上下限
bounds = [[0, 1]] * dim  # 入力の上下限
# bounds = [[0, 1], [0, 2], [0, 3]]  # 入力の上下限
sampling_method = 'random-uniform'  # 'random-uniform', 'random-norm' or 'LHS'


# ===== 型変換、ポカヨケ =====
bounds = np.array(bounds, dtype=float)
assert len(bounds) == dim


# ===== サンプリング =====
if sampling_method == 'random-uniform':
    x = (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n, dim) + bounds[:, 0]
elif sampling_method == 'random-norm':
    x = np.random.randn(n, dim) * 0.1 + 0.5
else:
    raise NotImplementedError(f'{sampling_method=}')
y = problem(x)

print('サンプリング:')
for xi, yi in zip(x, y):
    print(xi, '-->', yi)


# ===== 一部の感度分析手法で使うサロゲートモデルの作成 =====
gp = SingleTaskGP(
    train_X=tensor(x),
    train_Y=tensor(y),
    input_transform=Normalize(x.shape[-1], bounds=tensor(bounds).T),
    outcome_transform=Standardize(y.shape[-1]),
)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


def gp_surrogate_numpy(x_: np.ndarray):
    return gp.posterior(tensor(x_)).mean.detach().numpy()


print('サロゲートモデル:')
y_surrogate = gp_surrogate_numpy(x)
for xi, yi, ysi in zip(x, y, y_surrogate):
    print(xi, '-->', ysi, '/', yi, '=', ysi / yi)


# ===== 各種感度解析 =====

# ----- 回帰分析による分散分解 -----

# import
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# --- 1 次のみ ---

# 標準化用スケーラー
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
y_std = scaler.fit_transform(y)

# 一次項のみで標準化回帰
model_std = LinearRegression()
model_std.fit(x_std, y_std)

# 標準化偏回帰係数
std_coef = model_std.coef_

# 分散寄与率を求めるために1変数ずつ除外したR^2を計算
r2_full = model_std.score(x_std, y_std)
r2_drop = []

for i in range(x_std.shape[1]):
    x_drop = np.delete(x_std, i, axis=1)
    model_drop: LinearRegression = LinearRegression().fit(x_drop, y_std)  # type: ignore
    r2_drop.append(model_drop.score(x_drop, y_std))

# 分散寄与率（ΔR^2）
var_contrib = [r2_full - r2_d for r2_d in r2_drop]
var_ratio = np.array(var_contrib) / np.sum(var_contrib)

# 結果表示
assert y.shape[-1] == 1, ' 目的変数が 2 個以上の場合は結果表示が未実装です。'
print("\n[線形回帰 (1次のみ): 指標まとめ]")
df_result = pd.DataFrame({
    '標準化偏回帰係数': std_coef.ravel(),
    'ΔR²（分散寄与）': var_contrib,
    '分散比（正規化）': var_ratio,
    'sqrt ΔR²（分散寄与）': np.sqrt(var_contrib),
    'sqrt 分散比（正規化）': np.sqrt(var_ratio),
}, index=[f'x{i+1}' for i in range(dim)])

print(df_result)


# --- 2 次の交互作用まで ---

# 特徴量の生成（交互作用含む二次項まで）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)
feature_names = poly.get_feature_names_out([f'x{i+1}' for i in range(x.shape[1])])

# 標準化
scaler = StandardScaler()
X_poly_std = scaler.fit_transform(X_poly)
y_std = scaler.fit_transform(y)

# 線形回帰
model = LinearRegression()
model.fit(X_poly_std, y_std)

# 全体R²の計算
r2_full = model.score(X_poly_std, y_std)

# ΔR²（寄与）計算：1特徴量ずつ除いた場合のR²差分
r2_drop = []
for i in range(X_poly_std.shape[1]):
    X_reduced = np.delete(X_poly_std, i, axis=1)
    model_reduced = LinearRegression().fit(X_reduced, y_std)
    r2_drop.append(model_reduced.score(X_reduced, y_std))

delta_r2 = [r2_full - r2_d for r2_d in r2_drop]
var_ratio = np.array(delta_r2) / np.sum(delta_r2)

# 結果表示
df_result = pd.DataFrame({
    '項目': feature_names,
    '標準化偏回帰係数': model.coef_.ravel(),
    'ΔR²（分散寄与）': delta_r2,
    '分散比（正規化）': var_ratio,
    'sqrt ΔR²（分散寄与）': np.sqrt(delta_r2),
    'sqrt 分散比（正規化）': np.sqrt(var_ratio),
})

# 表示
pd.set_option('display.max_rows', None)
print("\n[交互作用を含む線形回帰モデルの感度分析結果]")
print(df_result.sort_values('ΔR²（分散寄与）', ascending=False).reset_index(drop=True))


# ----- Sobol 感度分析 -----

# import
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

# 問題定義
problem_spec = {
    'num_vars': dim,
    'names': [f'x{i+1}' for i in range(dim)],
    'bounds': bounds.tolist()
}

# サンプリング
param_values = sobol_sample.sample(problem_spec, 512, calc_second_order=True)
Y = gp_surrogate_numpy(param_values)
assert Y.shape[-1] == 1, '目的関数が 2 以上ある場合は sobol 感度計算の実装を変えてください。'
Y = Y.reshape(-1)

# Sobol 感度分析
sobol_result = sobol_analyze.analyze(problem_spec, Y, calc_second_order=True, print_to_console=False)

print("\n[Sobol 感度分析]")
for i, name in enumerate(problem_spec['names']):
    print(f"{name:5s}: S1={sobol_result['S1'][i]:.3f}, "
          f"ST={sobol_result['ST'][i]:.3f}, "
          f"sqrt(ST)={np.sqrt(sobol_result['ST'][i]):.3f}")


# ----- Morris 感度分析 -----

# import
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze

# 問題定義
problem_spec = {
    'num_vars': dim,
    'names': [f'x{i+1}' for i in range(dim)],
    'bounds': bounds.tolist()
}

# サンプリング
param_values = morris_sample.sample(problem_spec, N=100, num_levels=4, optimal_trajectories=None)
Y = gp_surrogate_numpy(param_values)
assert Y.shape[-1] == 1, '目的関数が 2 以上ある場合は morris 感度計算の実装を変えてください。'
Y = Y.reshape(-1)


# 感度分析
morris_result = morris_analyze.analyze(problem_spec, param_values, Y, conf_level=0.95, print_to_console=False)

print("\n[Morris 感度分析]")
for name, mu in zip(problem_spec['names'], morris_result['mu_star']):
    print(f"{name:5s}: mu*={mu:.3f}")


# ----- ランダムフォレストによる感度分析 -----

# import
from sklearn.ensemble import RandomForestRegressor

# モデル作成
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(x, y)

# importance
importances = model_rf.feature_importances_

print("\n[ランダムフォレスト: 特徴量の重要度]")
for i, imp in enumerate(importances):
    print(f"x{i+1}: {imp:.3f}, (sqrt) {np.sqrt(imp):.3f}")


# ----- SHAP による importance 分析 -----

# import
import shap

# shap を計算
explainer = shap.Explainer(gp_surrogate_numpy, x)
shap_values = explainer(x)

print("\n[SHAPによる特徴量重要度（平均絶対値）]")
shap_importance = np.abs(shap_values.values).mean(axis=0).ravel()
for i, val in enumerate(shap_importance):
    print(f"x{i+1}: {val:.3f}")

print("\n[SHAPによる特徴量重要度（最大最小の差）]")
shap_importance = (shap_values.values.max(axis=0) - shap_values.values.min(axis=0)).ravel()
for i, val in enumerate(shap_importance):
    print(f"x{i+1}: {val:.3f}")

print("\n[SHAPによる特徴量重要度（標準偏差）]")
shap_importance = shap_values.values.std(axis=0).ravel()
for i, val in enumerate(shap_importance):
    print(f"x{i+1}: {val:.3f}")

# # プロット（オプション）
# shap.plots.bar(shap_values)
# shap.plots.beeswarm(shap_values)


# ===== optuna =====
import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.importance import get_param_importances
from optuna.importance import (
    FanovaImportanceEvaluator,
    MeanDecreaseImpurityImportanceEvaluator,
    PedAnovaImportanceEvaluator,
)
import datetime

# パラメータ名を "x1", "x2", ..., とする
param_names = [f"x{i + 1}" for i in range(x.shape[1])]

# Optuna Study の作成
study = optuna.create_study(direction="minimize")

x = np.asarray(x)
y = np.asarray(y).ravel()
n_samples, n_features = x.shape

# 各サンプルを Trial として登録
for i in range(n_samples):
    params = {param_names[j]: float(x[i, j]) for j in range(n_features)}
    trial = FrozenTrial(
        number=i,
        value=float(y[i]),
        params=params,
        distributions={name: optuna.distributions.FloatDistribution(float(np.min(x[:, j])), float(np.max(x[:, j])))
                       for j, name in enumerate(param_names)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        state=TrialState.COMPLETE,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        trial_id=i
    )
    study.add_trial(trial)

# パラメータ重要度を取得
importances = get_param_importances(
    study,
    evaluator=FanovaImportanceEvaluator(),
    # evaluator=MeanDecreaseImpurityImportanceEvaluator(),
    # evaluator=PedAnovaImportanceEvaluator(),
)

# 結果を配列で返す（1番目を基準に正規化）
imp_values = np.array([importances.get(name, 0.0) for name in param_names])
print('[optuna]')
print('importance:', imp_values)
print('importance(sqrt)', np.sqrt(imp_values))
