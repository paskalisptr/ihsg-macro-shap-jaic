"""Pulls the RQ1-RQ3 results from outputs/tables/ into one summary table and one summary figure.

Run after 03_ols_wald.py, 05_inference.py and 06_shap.py. Does not refit anything.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"

wald = pd.read_csv(TAB / "rq1_wald.csv").query("std_errors.str.startswith('HAC')")
dm = pd.read_csv(TAB / "rq2_dm_test.csv")
full_vs_global = dm[dm["model_b"].str.startswith("global_") & dm["model_a"].str.startswith("full_")]
arimax_vs_naive = dm[(dm["model_a"] == "arimax") & (dm["model_b"] == "naive")].iloc[0]
shap_share = pd.read_csv(TAB / "rq3_shap_group_share.csv", index_col=0)

summary = pd.DataFrame([
    {"rq": "RQ1", "question": "domestic group significant (OLS Wald)",
     "statistic": wald.loc[wald["group"] == "domestic", "F"].iloc[0],
     "p": wald.loc[wald["group"] == "domestic", "p"].iloc[0]},
    {"rq": "RQ1", "question": "global group significant (OLS Wald)",
     "statistic": wald.loc[wald["group"] == "global", "F"].iloc[0],
     "p": wald.loc[wald["group"] == "global", "p"].iloc[0]},
    {"rq": "RQ2", "question": "domestic adds value beyond global (DM, worst-case model)",
     "statistic": full_vs_global.loc[full_vs_global["p"].idxmin(), "dm_stat"],
     "p": full_vs_global["p"].min()},
    {"rq": "RQ2", "question": "ARIMAX beats naive (DM)",
     "statistic": arimax_vs_naive["dm_stat"], "p": arimax_vs_naive["p"]},
    {"rq": "RQ3", "question": "domestic share of SHAP (range across LR/RF/XGBoost)",
     "statistic": shap_share["dom_share"].min(), "p": float("nan")},
])
summary["verdict"] = ["significant" if p < 0.05 else "not significant" for p in summary["p"]]
summary.loc[summary["question"].str.startswith("domestic share"), "verdict"] = \
    f"{shap_share['dom_share'].min():.0%} to {shap_share['dom_share'].max():.0%}"
summary.round(4).to_csv(TAB / "rq_summary.csv", index=False)

fig, ax = plt.subplots(1, 4, figsize=(16, 4))

w = wald.set_index("group").loc[["domestic", "global"]]
ax[0].bar(w.index, w["F"], color=["#d62728" if p < 0.05 else "#7f7f7f" for p in w["p"]])
ax[0].set_title("RQ1: OLS Wald F (red = p<0.05)")
ax[0].set_ylabel("F")

colors = ["#d62728" if p < 0.05 else "#7f7f7f" for p in full_vs_global["p"]]
ax[1].bar(full_vs_global["model_a"], full_vs_global["dm_stat"], color=colors)
ax[1].axhline(0, color="black", lw=0.8)
for v in (-1.96, 1.96):
    ax[1].axhline(v, color="grey", ls="--", lw=0.8)
ax[1].set_title("RQ2: DM, full vs global")
ax[1].set_ylabel("DM statistic")

arx_color = "#d62728" if arimax_vs_naive["p"] < 0.05 else "#7f7f7f"
ax[2].bar(["arimax vs naive"], [arimax_vs_naive["dm_stat"]], color=arx_color)
ax[2].axhline(0, color="black", lw=0.8)
for v in (-1.96, 1.96):
    ax[2].axhline(v, color="grey", ls="--", lw=0.8)
ax[2].set_title("RQ2: does ARIMAX beat naive?")
ax[2].set_ylabel("DM statistic")

ax[3].bar(shap_share.index, shap_share["dom_share"],
         yerr=[shap_share["dom_share"] - shap_share["ci_lo"], shap_share["ci_hi"] - shap_share["dom_share"]],
         capsize=4, color="#1f77b4")
ax[3].axhline(0.3, color="grey", ls="--", lw=1, label="3 of 10 features")
ax[3].set_title("RQ3: domestic share of SHAP")
ax[3].set_ylabel("share")
ax[3].legend(fontsize=8)

fig.suptitle("Domestic vs global macro factors and IHSG daily returns")
fig.tight_layout()
fig.savefig(FIG / "rq_summary.png", dpi=150)

print(summary.round(4).to_string(index=False))