from 関数版_色々な分析手法 import *

import numpy as np


# ===== システムの定義 =====
def problem(x_: np.ndarray) -> np.ndarray:
    """Calc the output.

    Args:
        x_ (np.ndarray): (n, d) shaped np.ndarray.

    Returns:
        np.ndarray: The (n, 1) shaped output value.

    """

    # 1 次
    # return (x_ * np.arange(1, 1 + len(x_[0])).reshape(1, -1)).sum(axis=1, keepdims=True)

    # 2 次
    return (x_ ** 2 * np.arange(1, 1 + len(x_[0])).reshape(1, -1)).sum(axis=1, keepdims=True)


# ===== 問題設定 =====
dim = 3  # 入力の次元
n = 10  # サンプル数
# p = 6  # minimum(p), dim <= p + 1
# n = p ** 2   # サンプル数
# bounds = [[0, 1], [-10, 10], [0, 100]]  # 入力の上下限
# bounds = [[0, 1]] * dim  # 入力の上下限
bounds = [[0, 1], [0, 2], [0, 3]]  # 入力の上下限
# sampling_method = 'random-uniform'  # 'random-uniform', 'random-norm' or 'LHS'
sampling_method = 'random-norm'  # 'random-uniform', 'random-norm' or 'LHS'

# ===== 型変換、ポカヨケ =====
bounds = np.array(bounds, dtype=float)
assert len(bounds) == dim


def main():

    # ===== サンプリング =====
    if sampling_method == 'random-uniform':
        x = (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n, dim) + bounds[:, 0]
    elif sampling_method == 'random-norm':
        x = np.random.randn(n, dim) * 0.1 + 0.5
    else:
        raise NotImplementedError(f'{sampling_method=}')
    y = problem(x)

    def calc_齟齬(result):
        return [np.abs(result - np.arange(1, 1 + len(result))).sum()]

    ret = {}
    ret.update({'ANOVA 1st 分析': calc_齟齬(anova_1st_order(x, y))})
    ret.update({'ANOVA 2nd 分析': calc_齟齬(anova_2nd_order(x, y)[:2])})
    ret.update({'Sobol 感度分析': calc_齟齬(sobol_sens_analysis(x, y, bounds))})
    ret.update({'Morris 感度分析': calc_齟齬(morris_sens_analysis(x, y, bounds))})
    ret.update({'optuna importance': calc_齟齬(optuna_importance(x, y))})
    ret.update({'ランダムフォレスト': calc_齟齬(random_forest_importance(x, y))})  # 的外れ
    ret.update({'SHAPLEY絶対値平均': calc_齟齬(shap(x, y, bounds))})  # あまりに遅い

    return ret
