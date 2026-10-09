"""RQ2 setup: fit LR, RF, XGBoost on domestic-only, global-only and full features, walk-forward split.

Hyperparameters for RF and XGBoost come from RandomizedSearchCV with TimeSeriesSplit, train only.
Saves one model file per scheme, plus a long table of test-set predictions for 05_inference.py
(Diebold-Mariano, binomial test) and 06_shap.py.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
G, seed = cfg["groups"], cfg["seed"]
TAB, FIG, MOD = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "outputs" / "models"
FIG.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
cut = pd.Timestamp(cfg["dates"]["train_end"])
train, test = df[df.index <= cut], df[df.index > cut]
y_tr, y_te = train["return_ihsg"], test["return_ihsg"]

schemes = {"domestic": G["domestic"], "global": G["global"], "full": G["domestic"] + G["global"]}
schemes = {k: [f"{n}_lag1" for n in v] for k, v in schemes.items()}

RF_GRID = {"n_estimators": [200, 400], "max_depth": [2, 3, 4, None], "min_samples_leaf": [5, 10, 20]}
XGB_GRID = {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.03, 0.1],
            "reg_alpha": [0, 0.1, 0.3], "reg_lambda": [1, 2, 4]}


def tune(model_cls, grid, Xtr, ytr):
    cv = TimeSeriesSplit(5)
    search = RandomizedSearchCV(model_cls(random_state=seed), grid, n_iter=15, cv=cv,
                                 scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=-1)
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


metrics, preds, tuned = [], [], {}
majority = max((y_tr > 0).mean(), (y_tr <= 0).mean())  # direction baseline, not 50/50 (train is 52.8% up days)

for scheme, cols in schemes.items():
    Xtr, Xte = train[cols], test[cols]
    mu, sd = Xtr.mean(), Xtr.std()  # LR standardization fit on train only, reused for test
    lr = LinearRegression().fit((Xtr - mu) / sd, y_tr)
    rf, rf_params = tune(RandomForestRegressor, RF_GRID, Xtr, y_tr)
    xgb, xgb_params = tune(XGBRegressor, XGB_GRID, Xtr, y_tr)
    tuned[f"rf_{scheme}"], tuned[f"xgb_{scheme}"] = rf_params, xgb_params

    joblib.dump((lr, mu, sd), MOD / f"lr_{scheme}.joblib")
    joblib.dump(rf, MOD / f"rf_{scheme}.joblib")
    joblib.dump(xgb, MOD / f"xgb_{scheme}.joblib")

    fitted = {"lr": pd.Series(lr.predict((Xte - mu) / sd), index=Xte.index),
              "rf": pd.Series(rf.predict(Xte), index=Xte.index),
              "xgb": pd.Series(xgb.predict(Xte), index=Xte.index)}
    for model, pred in fitted.items():
        rmse = float(np.sqrt(((pred - y_te) ** 2).mean()))
        mae = float((pred - y_te).abs().mean())
        acc = float((np.sign(pred) == np.sign(y_te)).mean())
        metrics.append({"scheme": scheme, "model": model, "rmse": rmse, "mae": mae, "dir_acc": acc})
        preds.append(pd.DataFrame({"scheme": scheme, "model": model, "pred": pred, "actual": y_te}))

metrics = pd.DataFrame(metrics)
naive_rmse = float(np.sqrt((y_te ** 2).mean()))  # predicting zero every day
metrics.round(6).to_csv(TAB / "rq2_metrics.csv", index=False)
pd.concat(preds).to_csv(TAB / "rq2_predictions.csv")
pd.Series(tuned).to_json(TAB / "rq2_best_params.json")

# RMSE by scheme and model, naive and majority-class lines for scale
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ["lr", "rf", "xgb"]
w = 0.25
for i, scheme in enumerate(schemes):
    sub = metrics[metrics["scheme"] == scheme].set_index("model").loc[order]
    ax.bar(np.arange(3) + i * w, sub["rmse"], width=w, label=scheme)
ax.axhline(naive_rmse, color="grey", ls="--", lw=1, label="naive (predict 0)")
ax.set_xticks(np.arange(3) + w, order)
ax.set_ylabel("RMSE, test period")
ax.set_ylim(metrics["rmse"].min() * 0.98, max(metrics["rmse"].max(), naive_rmse) * 1.01)  # differences are ~1%, need zoom
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_rmse_by_scheme.png", dpi=150)

# Directional accuracy vs the majority-class baseline (52.8% up days in training, not 50/50)
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, scheme in enumerate(schemes):
    sub = metrics[metrics["scheme"] == scheme].set_index("model").loc[order]
    ax.bar(np.arange(3) + i * w, sub["dir_acc"], width=w, label=scheme)
ax.axhline(majority, color="grey", ls="--", lw=1, label=f"majority class ({majority:.1%})")
ax.set_xticks(np.arange(3) + w, order)
ax.set_ylabel("directional accuracy, test period")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_diracc_by_scheme.png", dpi=150)

print(metrics.round(4).to_string(index=False))
print(f"naive RMSE (predict 0) = {naive_rmse:.6f}   majority-class share = {majority:.4f}")