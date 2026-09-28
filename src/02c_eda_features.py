"""EDA of the predictors: distributions, links to the return, lag structure, stability, drift.

Everything that guides modelling uses the training period only (train_end in config.yaml).
The drift table looks at the test period to describe the shift and is never used for tuning.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
df.columns = df.columns.str.replace("_lag1", "")
cut = pd.Timestamp(cfg["dates"]["train_end"])
tr, te = df[df.index <= cut], df[df.index > cut]
X, y = tr.drop(columns="return_ihsg"), tr["return_ihsg"]

stats_tab = pd.DataFrame({"mean": X.mean(), "sd": X.std(), "skew": X.skew(), "excess_kurtosis": X.kurt(),
                          "min": X.min(), "max": X.max(), "zero_share": (X == 0).mean()})

# Link to the return: linear, rank-based, and nonlinear (mutual information against a shuffled-target floor)
rel = pd.DataFrame({c: [*stats.pearsonr(X[c], y), *stats.spearmanr(X[c], y)] for c in X},
                   index=["pearson", "p_pearson", "spearman", "p_spearman"]).T
rel["p_spearman_bh"] = multipletests(rel["p_spearman"], method="fdr_bh")[1]  # 10 tests at once
rel["mutual_info"] = mutual_info_regression(X, y, random_state=cfg["seed"])
rng = np.random.default_rng(cfg["seed"])
rel["mi_shuffled_floor"] = np.mean([mutual_info_regression(X, rng.permutation(y.values), random_state=cfg["seed"])
                                    for _ in range(20)], axis=0)
rel = rel.reindex(rel["spearman"].abs().sort_values(ascending=False).index)

# Does an older value of a predictor say more than the one-day lag? (Spearman with the feature lagged 1..5 days)
lag_corr = pd.DataFrame({f"lag{k}": X.apply(lambda c: y.corr(c.shift(k - 1), method="spearman"))
                         for k in range(1, 6)})

# In-sample direction hit rate of each predictor alone, oriented by its training correlation.
# Optimistic on purpose: it shows how far one variable gets, not what a model will do out of sample.
# Days where the predictor sits exactly at its median give no call and are left out ("coverage").
hit = {}
for c in X:
    guess = np.sign((X[c] - X[c].median()) * np.sign(rel.loc[c, "spearman"]))
    called = guess != 0
    k, n = int((guess[called] == np.sign(y[called])).sum()), int(called.sum())
    hit[c] = {"coverage": n / len(y), "hit_rate": k / n, "binom_p_vs_50": stats.binomtest(k, n, 0.5).pvalue}
hit = pd.DataFrame(hit).T.sort_values("hit_rate", ascending=False)

# Train vs test shift (description only)
drift = pd.DataFrame({c: {"ks_stat": stats.ks_2samp(X[c], te[c]).statistic, "ks_p": stats.ks_2samp(X[c], te[c]).pvalue,
                          "mean_shift_in_train_sd": (te[c].mean() - X[c].mean()) / X[c].std()} for c in X}).T

for name, t in {"eda_feature_stats": stats_tab, "eda_feature_target": rel, "eda_lag_corr": lag_corr,
                "eda_single_feature_hit": hit, "eda_train_test_drift": drift}.items():
    t.round(4).to_csv(TAB / f"{name}.csv")

# Figures
corr = tr.corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Spearman correlation, training period")
fig.tight_layout()
fig.savefig(FIG / "eda_corr_heatmap.png", dpi=150)

fig, ax = plt.subplots(figsize=(11, 4))
for c in rel.index[:3]:
    y.rolling(250).corr(X[c]).plot(ax=ax, label=c)
ax.axhline(0, color="k", lw=0.8)
ax.legend()
ax.set_title("Rolling 250-day correlation with the return (top 3 predictors, training period)")
fig.tight_layout()
fig.savefig(FIG / "eda_rolling_corr.png", dpi=150)

print(rel.round(4).to_string(), "\n\n", lag_corr.round(3).to_string(), "\n\n", hit.round(4).to_string(),
      "\n\n", drift.round(3).to_string(), "\n\nshare of up days in training:", round((y > 0).mean(), 4))