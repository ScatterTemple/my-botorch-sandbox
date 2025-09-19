import numpy as np
from scipy.spatial import cKDTree
from typing import Callable

from torch import Tensor


# https://arxiv.org/pdf/2508.14803

# 拘束付きサンプリングの均一さを表す指標
# ======================================

# 分離半径。点群内で最も近い二点間を通過できる最大の球の半径。小さいほど密なところがある。
def separation_radius(X: np.ndarray) -> float:
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=[2])  # 最近傍（自分を含めて2番目）
    min_dist = np.min(dists)
    return 0.5 * min_dist


# 被覆半径。点群を通過できる最大の球の半径。大きいほど疎なところがある。
def cover_radius(X: np.ndarray, c: Callable[[np.ndarray], np.ndarray] | None, grid_points: int = 20) -> float:
    d = X.shape[1]
    grid_axes = [np.linspace(0, 1, grid_points)] * d
    mesh = np.meshgrid(*grid_axes, indexing="ij")
    grid = np.vstack([m.flatten() for m in mesh]).T  # (grid_points^d, d)

    # 制約を満たす点だけ抽出
    if c:
        mask = c(grid) > 0
        grid_feasible = grid[mask]
    else:
        grid_feasible = grid
    if grid_feasible.shape[0] == 0:
        raise ValueError("制約 c(x) > 0 を満たすグリッド点が存在しません。")

    tree = cKDTree(X)
    dists, _ = tree.query(grid_feasible, k=1)
    return np.max(dists)


# メッシュ比 = 被覆半径 / 分離半径。 1 に近いほど、密なところも疎なところもない。
def mesh_ratio(X: np.ndarray, c: Callable[[np.ndarray], np.ndarray] | None, grid_points: int = 20) -> float:
    cov = cover_radius(X, c, grid_points)
    sep = separation_radius(X)
    return cov / sep


# ===== テスト例 =====
if __name__ == "__main__":

    # 理想的な均一な数列
    # ------------------
    X = np.array([[i/3] for i in range(4)])

    # 制約関数の例
    def constraint(X_: np.ndarray) -> np.ndarray:
        # return 1.0 - np.sum(X_**2, axis=1)
        return (X_ - 1/3)[:, 0]

    print('理想')
    print("Fill radius:", separation_radius(X))  # 厳密に 1/6
    print("Cover radius:", cover_radius(X, constraint, grid_points=50))  # だいたい 1/6 (grid 次第)
    print("Mesh ratio:", mesh_ratio(X, constraint, grid_points=50))  # だいたい 1

    # 制約関数
    def constraint2(X_: np.ndarray) -> np.ndarray:
        return 0.75 ** 2 - np.sum(X_**2, axis=1)

    # ランダムサンプリング
    # --------------------
    print('ランダムサンプリング')
    for i in range(5):
        n, d = 50, 2
        c = None  # 拘束なし
        X = np.random.rand(n, d)  # [0,1]^2 にランダムサンプル
        # print("Fill radius:", separation_radius(X))
        # print("Cover radius:", cover_radius(X, c, grid_points=200))
        # print("Mesh ratio:", mesh_ratio(X, c, grid_points=200))
        print(mesh_ratio(X, c, grid_points=200))
        # 5 回の結果: かなり大きい。
        # --------------------------
        # 38.38477799029163
        # 423.5907439959716
        # 50.1699439400088
        # 100.81280415082301
        # 18.534949233005783
        # => 疎なところも密なところもある

    # LHS サンプリング
    # ----------------
    print('LHS サンプリング')
    from scipy.stats import qmc

    for i in range(5):
        sampler = qmc.LatinHypercube(d=2)
        X = sampler.random(n=50)
        c = None  # 拘束なし
        # print("Fill radius:", separation_radius(X))
        # print("Cover radius:", cover_radius(X, c, grid_points=200))
        # print("Mesh ratio:", mesh_ratio(X, c, grid_points=200))
        print(mesh_ratio(X, c, grid_points=200))
        # 5 回の結果: ややマシ。
        # ----------------------------
        # 14.566089693329063
        # 18.0852289166908
        # 15.493701830627284
        # 17.011027379314587
        # 14.023253854179831
        # => ランダムよりややマシ。ばらつきも小さい気がする。

    # optuna QMC サンプリング
    # -----------------------
    print('QMC (halton) サンプリング')
    import warnings
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.filterwarnings("ignore")

    def func(trial: optuna.Trial) -> float:
        trial.suggest_float('x1', 0, 1)
        trial.suggest_float('x2', 0, 1)
        return 1.

    for i in range(5):
        sampler = optuna.samplers.QMCSampler(qmc_type='halton')
        study = optuna.create_study(sampler=sampler)
        study.optimize(func, n_trials=50)
        df = study.trials_dataframe()
        X = df[['params_x1', 'params_x2']].values
        c = None  # 拘束なし
        # print("Fill radius:", separation_radius(X))
        # print("Cover radius:", cover_radius(X, c, grid_points=200))
        # print("Mesh ratio:", mesh_ratio(X, c, grid_points=200))
        print(mesh_ratio(X, c, grid_points=200))
        # 5 回の結果: さらにマシ。
        # ----------------------------
        # 6.618271619462645
        # 6.618271619462645
        # 15.666906230473913
        # 7.334090241116641
        # 6.618271619462645
        # => LHS よりさらにマシ。sobol より halton のほうがちょっといいかもだが、若干ばらつくか？

    # LHS + botorch カスタム獲得関数サンプリング
    # ------------------------------------------
    print('LHS + botorch カスタム獲得関数サンプリング')
    import torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize, Normalize
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from botorch.acquisition import ExpectedImprovement

    class SigmaAcquisition(ExpectedImprovement):
        def forward(self, X: Tensor) -> Tensor:
            mean, sigma = self._mean_and_sigma(X)
            return sigma

    def func(X_):
        return torch.sin(4 * np.pi * (X_[:, 0].unsqueeze(1) + X_[:, 1].unsqueeze(1)))

    for i in range(5):
        # 初期サンプリング
        sampler = qmc.LatinHypercube(d=2)
        X = sampler.random(n=10)
        c = None  # 拘束なし

        # botorch
        bounds = torch.stack([torch.zeros(2), torch.ones(2)])
        x = torch.tensor(X)
        y = func(x)
        for j in range(40):
            standardizer = Standardize(m=1)
            standardizer.forward(y)
            _, yvar = standardizer.untransform(y, 1e-6 * torch.ones_like(y))
            gp = SingleTaskGP(
                x,
                y,
                train_Yvar=yvar,
                input_transform=Normalize(d=2, bounds=bounds),
                outcome_transform=standardizer,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = SigmaAcquisition(gp, 0.)
            candidate, acq_value = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )
            yp = func(candidate)
            x = torch.concat([x, candidate])
            y = torch.concat([y, yp])
        X = x.detach().numpy()
        # print("Fill radius:", separation_radius(X))
        # print("Cover radius:", cover_radius(X, c, grid_points=200))
        # print("Mesh ratio:", mesh_ratio(X, c, grid_points=200))
        print(mesh_ratio(X, c, grid_points=200))
        # 5 回の実行結果。一番優秀。
        # 目的関数の設定次第で精度が変わるが、
        # 寄与度の低い変数は分散にも寄与しないので。
        # ------------------------------------------
        # 2.224760343057195
        # 2.252263810590621
        # 3.2426441886710786
        # 2.047544641570767
        # 2.475605882344007


