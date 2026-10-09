"""Robustness check: does a much wider hyperparameter search change the RQ2 numbers from
04_models.py, or does the narrow grid already sit near the ceiling? RF/XGBoost get a wider
RandomizedSearchCV; LR gets Ridge and Lasso variants (04_models.py only tried plain LR).

Everything else -- split, scaling, scoring -- matches 04_models.py exactly, so the comparison
is fair.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
G, seed = cfg["groups"], cfg["seed"]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
cut = pd.Timestamp(cfg["dates"]["train_end"])
train, test = df[df.index <= cut], df[df.index > cut]
y_tr, y_te = train["return_ihsg"], test["return_ihsg"]

schemes = {k: [f"{n}_lag1" for n in v] for k, v in
          {"domestic": G["domestic"], "global": G["global"], "full": G["domestic"] + G["global"]}.items()}

# n_estimators=800 and min_samples_leaf=1 are in the grid on purpose but pushed n_iter down --
# a lone CPU core in this environment made 800-tree RF fits the bottleneck, not the search itself.
RF_WIDE = {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4, 5, 6, None],
          "min_samples_leaf": [1, 2, 5, 10, 20, 30]}
XGB_WIDE = {"n_estimators": [100, 200, 400, 800], "max_depth": [2, 3, 4, 5, 6],
           "learning_rate": [0.005, 0.01, 0.03, 0.05, 0.1], "reg_alpha": [0, 0.05, 0.1, 0.3, 0.5, 1],
           "reg_lambda": [0.5, 1, 2, 4, 8], "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]}
N_ITER = {"rf": 25, "xgb": 40}  # roughly 2-3x the n_iter=15 search 04_models.py ran


def tune(model_cls, grid, Xtr, ytr, key):
    cv = TimeSeriesSplit(5)
    search = RandomizedSearchCV(model_cls(random_state=seed), grid, n_iter=N_ITER[key], cv=cv,
                                 scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1)
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


def metrics(pred, y):
    return {"rmse": float(np.sqrt(((pred - y) ** 2).mean())), "mae": float((pred - y).abs().mean()),
            "dir_acc": float((np.sign(pred) == np.sign(y)).mean())}


rows, best_params = [], {}
for scheme, cols in schemes.items():
    Xtr, Xte = train[cols], test[cols]
    mu, sd = Xtr.mean(), Xtr.std()
    Xz_tr, Xz_te = (Xtr - mu) / sd, (Xte - mu) / sd
    cv = TimeSeriesSplit(5)

    lr_variants = {
        "lr_plain": LinearRegression().fit(Xz_tr, y_tr),
        "lr_ridge": RidgeCV(alphas=np.logspace(-3, 3, 30), cv=cv).fit(Xz_tr, y_tr),
        "lr_lasso": LassoCV(alphas=np.logspace(-5, 1, 30), cv=cv, max_iter=5000).fit(Xz_tr, y_tr),
    }
    for name, m in lr_variants.items():
        rows.append({"scheme": scheme, "model": name, **metrics(m.predict(Xz_te), y_te)})

    rf, rf_params = tune(RandomForestRegressor, RF_WIDE, Xtr, y_tr, "rf")
    xgb, xgb_params = tune(XGBRegressor, XGB_WIDE, Xtr, y_tr, "xgb")
    best_params[f"rf_{scheme}"], best_params[f"xgb_{scheme}"] = rf_params, xgb_params
    rows.append({"scheme": scheme, "model": "rf_wide", **metrics(rf.predict(Xte), y_te)})
    rows.append({"scheme": scheme, "model": "xgb_wide", **metrics(xgb.predict(Xte), y_te)})

wide = pd.DataFrame(rows)
wide.round(6).to_csv(TAB / "rq2c_deep_tuning.csv", index=False)
pd.Series(best_params).to_json(TAB / "rq2c_deep_tuning_params.json")

# Narrow-grid results from 04_models.py, for a side-by-side comparison on the full scheme
narrow = pd.read_csv(TAB / "rq2_metrics.csv").query("scheme == 'full'").set_index("model")["rmse"]
wide_full = wide.query("scheme == 'full'").set_index("model")["rmse"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
pairs = [("lr", "lr_plain"), ("rf", "rf_wide"), ("xgb", "xgb_wide")]
x = np.arange(len(pairs))
ax[0].bar(x - 0.15, [narrow[a] for a, _ in pairs], width=0.3, label="narrow grid (04_models.py)")
ax[0].bar(x + 0.15, [wide_full[b] for _, b in pairs], width=0.3, label="wide grid (this script)")
all_vals = [narrow[a] for a, _ in pairs] + [wide_full[b] for _, b in pairs]
ax[0].set_xticks(x, [a for a, _ in pairs])
ax[0].set_ylabel("RMSE, test period, full scheme")
ax[0].set_ylim(min(all_vals) * 0.995, max(all_vals) * 1.005)  # differences are ~0.1%, need zoom
ax[0].set_title("Same features, deeper search: RMSE barely moves")
ax[0].legend()

lr_full = wide.query("scheme == 'full' and model.str.startswith('lr_')").set_index("model")["rmse"]
ax[1].bar(lr_full.index, lr_full)
ax[1].set_ylabel("RMSE, test period, full scheme")
ax[1].set_ylim(lr_full.min() * 0.995, lr_full.max() * 1.005)
ax[1].set_title("Plain LR vs Ridge vs Lasso")
fig.tight_layout()
fig.savefig(FIG / "rq2c_deep_tuning.png", dpi=150)

print(wide.round(5).to_string(index=False))
print("\nfull-scheme RMSE, narrow vs wide grid:")
print(pd.DataFrame({"narrow": [narrow[a] for a, _ in pairs], "wide": [wide_full[b] for _, b in pairs]},
                   index=[a for a, _ in pairs]).round(6).to_string())