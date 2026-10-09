"""ARIMAX baseline for RQ2: same walk-forward split and 10 macro predictors as 04_models.py,
but with an AR(1) term. AIC grid search on the training set picked order (1,0,0) over 8 alternatives.

Parameters are estimated once on train, then the state is filtered forward through the test period
(no re-estimation) -- same "fit once, predict on test" rule as the other models.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
cut = pd.Timestamp(cfg["dates"]["train_end"])
train, test = df[df.index <= cut], df[df.index > cut]
cols = [c for c in df.columns if c != "return_ihsg"]

# integer index: statsmodels' append() needs a supported index, and daily trading dates aren't a fixed frequency
y_tr, y_te = train["return_ihsg"].reset_index(drop=True), test["return_ihsg"].reset_index(drop=True)
y_te.index += len(y_tr)
X_tr, X_te = train[cols].reset_index(drop=True), test[cols].reset_index(drop=True)
X_te.index += len(X_tr)

model = SARIMAX(y_tr, exog=X_tr, order=(1, 0, 0), trend="c",
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200, method="lbfgs")
extended = model.append(y_te, exog=X_te, refit=False)
pred = extended.get_prediction(start=len(y_tr), end=len(y_tr) + len(y_te) - 1).predicted_mean
pred.index = test.index

rmse = float(np.sqrt(((pred - test["return_ihsg"]) ** 2).mean()))
mae = float((pred - test["return_ihsg"]).abs().mean())
acc = float((np.sign(pred) == np.sign(test["return_ihsg"])).mean())
pd.DataFrame([{"model": "arimax", "rmse": rmse, "mae": mae, "dir_acc": acc}]).round(6) \
    .to_csv(TAB / "rq2_arimax_metrics.csv", index=False)
pd.DataFrame({"model": "arimax", "pred": pred, "actual": test["return_ihsg"]}).to_csv(TAB / "rq2_arimax_predictions.csv")
print(model.summary().tables[1])
print(f"\nAR(1) coefficient: {model.params['ar.L1']:.4f}  (p={model.pvalues['ar.L1']:.4f})")
print(f"test RMSE={rmse:.6f}  MAE={mae:.6f}  dir_acc={acc:.4f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(test.index, test["return_ihsg"], lw=0.8, label="actual", alpha=0.7)
ax.plot(test.index, pred, lw=0.8, label="ARIMAX(1,0,0) fitted", alpha=0.8)
ax.set_title("ARIMAX one-step forecasts on the test period")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_arimax_fit.png", dpi=150)