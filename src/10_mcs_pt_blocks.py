"""RQ2 and RQ3 extras: Model Confidence Set, Pesaran-Timmermann direction test, SHAP share per block.

1. MCS (Hansen, Lunde and Nason 2011) on squared-error loss over the test set: which of the nine ML
   specifications, ARIMAX and the naive forecast are statistically indistinguishable from the best.
2. Pesaran-Timmermann test: is the sign forecast independent of the realised sign.
3. Domestic share of SHAP in each rolling block (same refit schedule and fixed hyperparameters as
   08_rolling_origin.py, full-feature models, SHAP on the block's days). Shows whether the share is stable.
Run after 04_models.py, 04b_ARIMAX.py and 08_rolling_origin.py. Needs the `arch` package.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import yaml
from arch.bootstrap import MCS
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
G, seed = cfg["groups"], cfg["seed"]
TAB = ROOT / "outputs" / "tables"

# ---- 1. Model Confidence Set -------------------------------------------------------------------
ml = pd.read_csv(TAB / "rq2_predictions.csv", index_col=0, parse_dates=True)
ml["label"] = ml["scheme"] + "_" + ml["model"]
ar = pd.read_csv(TAB / "rq2_arimax_predictions.csv", index_col=0, parse_dates=True)
ar["label"] = "arimax"
long = pd.concat([ml[["label", "pred", "actual"]], ar[["label", "pred", "actual"]]])
pred = long.pivot_table(index=long.index, columns="label", values="pred")
actual = long.groupby(long.index)["actual"].first()
loss = (pred.sub(actual, axis=0)) ** 2
loss["naive"] = actual ** 2  # naive predicts 0

mcs = MCS(loss, size=0.10, reps=2000, method="R", block_size=10, seed=seed)
mcs.compute()
pvals = mcs.pvalues["Pvalue"].rename("mcs_p")
pvals.to_frame().assign(in_set_90=mcs.pvalues["Pvalue"] >= 0.10).round(4).to_csv(TAB / "rq2_mcs.csv")
print("Model Confidence Set (alpha = 0.10), p-value >= 0.10 means kept:")
print(pvals.round(4).sort_values(ascending=False).to_string())


# ---- 2. Pesaran-Timmermann ---------------------------------------------------------------------
def pesaran_timmermann(p, a):
    """PT (1992): H0 = sign of the forecast is independent of the sign of the outcome. One-sided p."""
    n = len(a)
    py, px = (a > 0).mean(), (p > 0).mean()
    hit = (np.sign(p) == np.sign(a)).mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_hit = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) + (2 * px - 1) ** 2 * py * (1 - py)
              + 4 * py * px * (1 - py) * (1 - px) / n) / n
    denom = v_hit - v_star
    if denom <= 0:  # forecast sign almost constant, PT variance undefined
        return hit, np.nan, np.nan
    z = (hit - p_star) / np.sqrt(denom)
    return hit, z, 1 - stats.norm.cdf(z)


pt = pd.DataFrame([{"label": c, **dict(zip(("hit_rate", "pt_z", "pt_p"), pesaran_timmermann(pred[c], actual)))}
                   for c in pred.columns])
pt.round(4).to_csv(TAB / "rq2_pesaran_timmermann.csv", index=False)
print("\nPesaran-Timmermann:")
print(pt.round(4).to_string(index=False))

# ---- 3. SHAP domestic share per rolling block --------------------------------------------------
df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
params = json.loads((TAB / "rq2_best_params.json").read_text())
y = df["return_ihsg"]
test_idx = df.index[df.index > pd.Timestamp(cfg["dates"]["train_end"])]
cols = [f"{n}_lag1" for n in G["domestic"] + G["global"]]
is_dom = np.array([c.removesuffix("_lag1") in G["domestic"] for c in cols])


def dom_share(v):
    imp = np.abs(v).mean(axis=0)
    return imp[is_dom].sum() / imp.sum()


rows = []
for i in range(0, len(test_idx), 63):
    block = test_idx[i:i + 63]
    tr = df.index[df.index < block[0]]
    Xtr, ytr, Xb = df.loc[tr, cols], y[tr], df.loc[block, cols]
    mu, sd = Xtr.mean(), Xtr.std()
    lr = LinearRegression().fit((Xtr - mu) / sd, ytr)
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **params["rf_full"]).fit(Xtr, ytr)
    xgb = XGBRegressor(random_state=seed, **params["xgb_full"]).fit(Xtr, ytr)
    Xz = (Xb - mu) / sd
    bg = (Xtr - mu) / sd
    sv = {"rf": shap.TreeExplainer(rf).shap_values(Xb), "xgb": shap.TreeExplainer(xgb).shap_values(Xb),
          "lr": shap.LinearExplainer(lr, shap.maskers.Independent(bg, max_samples=len(bg))).shap_values(Xz)}
    rows.append({"block_start": block[0].date(), "block_end": block[-1].date(), "n_days": len(block),
                 **{f"dom_share_{m}": dom_share(v) for m, v in sv.items()}})
blocks = pd.DataFrame(rows)
blocks.round(4).to_csv(TAB / "rq3_shap_blocks.csv", index=False)
print("\nDomestic SHAP share per rolling block:")
print(blocks.round(4).to_string(index=False))