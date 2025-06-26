import optuna
from optuna.visualization import plot_param_importances
from optuna.samplers import QMCSampler

# 評価関数（例: y = x1² + 2 * x2² を最小化）
def objective(trial):
    x1 = trial.suggest_uniform('x1', -1.0, 1.0)
    x2 = trial.suggest_uniform('x2', -1.0, 1.0)
    y = x1**2 + 2 * x2**2
    return y

# 最適化を実行
study = optuna.create_study(direction='minimize', sampler=QMCSampler())
study.optimize(objective, n_trials=100)

# 重要度の可視化（Jupyter Notebook用）
fig = plot_param_importances(study)
fig.show()
