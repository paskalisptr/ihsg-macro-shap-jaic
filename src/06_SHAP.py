"""RQ3: SHAP share of domestic vs global variables, on the full-scheme models from 04_models.py.

TreeExplainer for RF/XGBoost, LinearExplainer for LR. Group share and per-feature share both
reported, since global has 7 variables and domestic only 3 -- a raw group total favors global by
construction. CI from a moving-block bootstrap on test days, block length 20 (matches the rolling
window used elsewhere in the pipeline), to respect the return series' autocorrelation.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
G, seed = cfg["groups"], cfg["seed"]
TAB, FIG, MOD = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "outputs" / "models"

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
test = df[df.index > cfg["dates"]["train_end"]]
cols = [f"{n}_lag1" for n in G["domestic"] + G["global"]]
X = test[cols]
is_dom = np.array([c.removesuffix("_lag1") in G["domestic"] for c in cols])

rf, xgb = joblib.load(MOD / "rf_full.joblib"), joblib.load(MOD / "xgb_full.joblib")
lr, mu, sd = joblib.load(MOD / "lr_full.joblib")
Xz = (X - mu) / sd  # LR was trained on standardized features

shap_vals = {"rf": shap.TreeExplainer(rf).shap_values(X),
            "xgb": shap.TreeExplainer(xgb).shap_values(X),
            "lr": shap.LinearExplainer(lr, shap.maskers.Independent(Xz, max_samples=len(Xz))).shap_values(Xz)}


def group_share(v):
    imp = np.abs(v).mean(axis=0)
    dom, glob = imp[is_dom].sum(), imp[~is_dom].sum()
    return {"dom_share": dom / (dom + glob), "dom_per_feature": dom / is_dom.sum(),
            "glob_per_feature": glob / (~is_dom).sum()}


def block_bootstrap_ci(v, block=20, n_boot=1000):
    """Resample contiguous blocks of days (with replacement) to keep the return series' own autocorrelation."""
    rng = np.random.default_rng(seed)
    n = len(v)
    starts = np.arange(n - block + 1)
    shares = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n // block + 1)])[:n]
        shares.append(group_share(v[idx])["dom_share"])
    return np.percentile(shares, [2.5, 97.5])

summary = pd.DataFrame({m: group_share(v) for m, v in shap_vals.items()}).T
summary[["ci_lo", "ci_hi"]] = pd.DataFrame({m: block_bootstrap_ci(v) for m, v in shap_vals.items()}).T
summary.round(4).to_csv(TAB / "rq3_shap_group_share.csv")

per_feature = pd.DataFrame({m: np.abs(v).mean(axis=0) for m, v in shap_vals.items()}, index=cols)
per_feature.round(6).to_csv(TAB / "rq3_shap_per_feature.csv")

# Group share (with CI) across the three model architectures: is RQ3's finding model-specific or robust?
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(summary.index, summary["dom_share"],
      yerr=[summary["dom_share"] - summary["ci_lo"], summary["ci_hi"] - summary["dom_share"]],
      capsize=4, color="#1f77b4")
ax.axhline(3 / 10, color="grey", ls="--", lw=1, label="3 of 10 features (share if all equally weighted)")
ax.set_ylabel("domestic share of total |SHAP|")
ax.set_title("RQ3: domestic share of SHAP, 95% block-bootstrap CI")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq3_group_share.png", dpi=150)

# Per-feature mean |SHAP|, sorted, colored by group -- the fair comparison since domestic has fewer features
fig, ax = plt.subplots(figsize=(7, 5))
order = per_feature["xgb"].sort_values().index
colors = ["#1f77b4" if c.removesuffix("_lag1") in G["domestic"] else "#d62728" for c in order]
ax.barh(order, per_feature.loc[order, "xgb"], color=colors)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#1f77b4", "#d62728")]
ax.legend(handles, ["domestic", "global"])
ax.set_xlabel("mean |SHAP|, XGBoost")
ax.set_title("Per-feature SHAP, full model")
fig.tight_layout()
fig.savefig(FIG / "rq3_per_feature_xgb.png", dpi=150)

print(summary.round(4).to_string())
print()
print(per_feature.round(5).to_string())