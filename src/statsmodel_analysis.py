import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

# データを DataFrame 化
df = pd.DataFrame(X, columns=["x1", "x2"])
df["y"] = y

# モデルをフィッティング
model = smf.ols('y ~ x1 + x2', data=df).fit()

# ANOVAテーブル出力
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
