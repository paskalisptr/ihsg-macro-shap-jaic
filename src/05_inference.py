"""RQ2 formal tests: Diebold-Mariano on squared error (does one specification predict better than
another), binomial test on direction (does a model call the sign better than the training-period
base rate). Reads outputs/tables/rq2_predictions.csv (04_models.py) and rq2_arimax_predictions.csv
(04b_arimax.py).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"

ml = pd.read_csv(TAB / "rq2_predictions.csv", index_col=0, parse_dates=True)
ml["label"] = ml["scheme"] + "_" + ml["model"]
arimax = pd.read_csv(TAB / "rq2_arimax_predictions.csv", index_col=0, parse_dates=True)
arimax["label"] = "arimax"
long = pd.concat([ml[["label", "pred", "actual"]], arimax[["label", "pred", "actual"]]])

train = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)["return_ihsg"]
train = train[train.index <= cfg["dates"]["train_end"]]
majority = max((train > 0).mean(), (train <= 0).mean())
n = long.groupby("label").size().iloc[0]
lags = int(4 * (n / 100) ** (2 / 9))


def dm_test(e1, e2):
    """Diebold-Mariano on squared-error loss, HAC standard error, Harvey-Leybourne-Newbold correction."""
    d = e1**2 - e2**2
    se = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags}).bse.iloc[0]
    dm = d.mean() / se * np.sqrt((n + 1 - 2 + 1 / n) / n)  # HLN correction, h=1
    return dm, 2 * (1 - stats.t.cdf(abs(dm), n - 1))


err = {label: g.set_index(g.index)["pred"] - g["actual"] for label, g in long.groupby("label")}
err["naive"] = -long.loc[long["label"] == "domestic_lr", "actual"]  # naive predicts 0, so error = -actual

comparisons = ([(f"{s}_{m}", "naive") for s in ("domestic", "global", "full") for m in ("lr", "rf", "xgb")]
              + [("arimax", "naive")]
              + [(f"full_{m}", f"global_{m}") for m in ("lr", "rf", "xgb")]
              + [(f"full_{m}", f"domestic_{m}") for m in ("lr", "rf", "xgb")]
              + [(f"global_{m}", f"domestic_{m}") for m in ("lr", "rf", "xgb")]
              + [("arimax", "global_lr"), ("arimax", "full_lr")])
dm = pd.DataFrame([{"model_a": a, "model_b": b, "dm_stat": (s := dm_test(err[a], err[b]))[0], "p": s[1]}
                   for a, b in comparisons])

binom = pd.DataFrame([{"label": label, "hit_rate": (k := int((np.sign(g["pred"]) == np.sign(g["actual"])).sum())) / n,
                       "p_vs_majority": stats.binomtest(k, n, majority).pvalue,
                       "p_vs_50_50": stats.binomtest(k, n, 0.5).pvalue}
                      for label, g in long.groupby("label")])

dm.round(4).to_csv(TAB / "rq2_dm_test.csv", index=False)
binom.round(4).to_csv(TAB / "rq2_binom_test.csv", index=False)

# Does domestic add value beyond global? (full vs global, one bar per model)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
key = dm[dm["model_b"].str.startswith("global_") & dm["model_a"].str.startswith("full_")]
ax[0].barh(key["model_a"], key["dm_stat"], color=["#d62728" if p < 0.05 else "#7f7f7f" for p in key["p"]])
ax[0].axvline(0, color="black", lw=0.8)
for v in (-1.96, 1.96):
    ax[0].axvline(v, color="grey", ls="--", lw=0.8)
ax[0].set_xlabel("DM statistic (negative = full has lower RMSE)")
ax[0].set_title("Domestic beyond global? (red = p<0.05)")

# Does ARIMAX's AR(1) term earn its keep, against naive and against the plain linear model?
arx = dm[dm["model_a"] == "arimax"]
ax[1].barh(arx["model_b"], arx["dm_stat"], color=["#d62728" if p < 0.05 else "#7f7f7f" for p in arx["p"]])
ax[1].axvline(0, color="black", lw=0.8)
for v in (-1.96, 1.96):
    ax[1].axvline(v, color="grey", ls="--", lw=0.8)
ax[1].set_xlabel("DM statistic (negative = ARIMAX has lower RMSE)")
ax[1].set_title("ARIMAX vs naive and vs plain linear")
fig.tight_layout()
fig.savefig(FIG / "rq2_dm_key_comparisons.png", dpi=150)

# Directional accuracy with significance against the majority-class baseline
fig, ax = plt.subplots(figsize=(9, 4.5))
b = binom.set_index("label").loc[[f"{s}_{m}" for s in ("domestic", "global", "full") for m in ("lr", "rf", "xgb")] + ["arimax"]]
ax.bar(range(len(b)), b["hit_rate"], color=["#2ca02c" if p < 0.05 else "#1f77b4" for p in b["p_vs_majority"]])
ax.axhline(majority, color="grey", ls="--", lw=1, label=f"majority class ({majority:.1%})")
ax.set_xticks(range(len(b)), b.index, rotation=45, ha="right")
ax.set_ylabel("directional accuracy")
ax.set_title("Green = beats the majority-class baseline, binomial p<0.05")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "rq2_binom_accuracy.png", dpi=150)

print(dm.round(4).to_string(index=False))
print()
print(binom.round(4).to_string(index=False))