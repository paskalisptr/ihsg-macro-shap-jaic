"""RQ2 robustness: rolling-origin (expanding window) evaluation over the test period.

04_models.py fits once on the training years. Here the training window grows and every model is
refit every REFIT_EVERY trading days, so each test day is predicted by a model that has seen all
data before its block. Hyperparameters stay fixed at the values tuned in 04_models.py (not
re-tuned per refit). Reads rq2_best_params.json. Run after 04_models.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
G, seed = cfg["groups"], cfg["seed"]
TAB = ROOT / "outputs" / "tables"
REFIT_EVERY = 63  # about one quarter of trading days

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
params = json.loads((TAB / "rq2_best_params.json").read_text())
y = df["return_ihsg"]
cut = pd.Timestamp(cfg["dates"]["train_end"])
test_idx = df.index[df.index > cut]

schemes = {"domestic": G["domestic"], "global": G["global"], "full": G["domestic"] + G["global"]}
schemes = {k: [f"{n}_lag1" for n in v] for k, v in schemes.items()}
preds = {f"{s}_{m}": pd.Series(index=test_idx, dtype=float) for s in schemes for m in ("lr", "rf", "xgb")}

for i in range(0, len(test_idx), REFIT_EVERY):
    block = test_idx[i:i + REFIT_EVERY]
    train_idx = df.index[df.index < block[0]]  # everything before the block, nothing after
    for s, cols in schemes.items():
        Xtr, ytr, Xb = df.loc[train_idx, cols], y[train_idx], df.loc[block, cols]
        mu, sd = Xtr.mean(), Xtr.std()
        preds[f"{s}_lr"][block] = LinearRegression().fit((Xtr - mu) / sd, ytr).predict((Xb - mu) / sd)
        rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **params[f"rf_{s}"]).fit(Xtr, ytr)
        xgb = XGBRegressor(random_state=seed, **params[f"xgb_{s}"]).fit(Xtr, ytr)
        preds[f"{s}_rf"][block] = rf.predict(Xb)
        preds[f"{s}_xgb"][block] = xgb.predict(Xb)

actual = y[test_idx]
n = len(actual)
lags = int(4 * (n / 100) ** (2 / 9))  # same HAC lag rule as 05_inference.py
majority = max((y[y.index <= cut] > 0).mean(), (y[y.index <= cut] <= 0).mean())
naive_rmse = float(np.sqrt((actual ** 2).mean()))


def dm_test(e1, e2):
    """Diebold-Mariano on squared error, HAC standard error, Harvey-Leybourne-Newbold correction."""
    d = e1 ** 2 - e2 ** 2
    se = sm.OLS(d, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).bse.iloc[0]
    t = d.mean() / se * np.sqrt((n - 1) / n)
    return t, 2 * (1 - stats.t.cdf(abs(t), n - 1))


rows = []
for label, p in preds.items():
    e = p - actual
    rmse = float(np.sqrt((e ** 2).mean()))
    hits = int((np.sign(p) == np.sign(actual)).sum())
    rows.append({"label": label, "rmse": rmse, "rmse_vs_naive_pct": 100 * (rmse / naive_rmse - 1),
                 "dm_p_vs_naive": dm_test(e, -actual)[1], "dir_acc": hits / n,
                 "p_dir_vs_majority": stats.binomtest(hits, n, majority).pvalue})
out = pd.DataFrame(rows)
out.round(6).to_csv(TAB / "rq2_rolling.csv", index=False)

pairs = [(f"full_{m}", f"global_{m}") for m in ("lr", "rf", "xgb")] + \
        [(f"global_{m}", f"domestic_{m}") for m in ("lr", "rf", "xgb")]
cmp = pd.DataFrame([{"model_a": a, "model_b": b, "dm_stat": (r := dm_test(preds[a] - actual, preds[b] - actual))[0],
                     "p": r[1]} for a, b in pairs])
cmp.round(4).to_csv(TAB / "rq2_rolling_pairs.csv", index=False)

print(out.round(4).to_string(index=False))
print(cmp.round(4).to_string(index=False))
print(f"naive RMSE = {naive_rmse:.6f}   n = {n}   majority = {majority:.4f}")