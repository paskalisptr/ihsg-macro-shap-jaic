"""Line charts for RQ2 and RQ3 (test period).

Fig A: cumulative squared-error gain over the naive forecast (naive loss minus model loss, summed
       day by day). A line that rises means the model beat naive; flat means no gain; the slope shows
       when the gain happened.
Fig B: domestic share of SHAP in each rolling block (from 10_mcs_pt_blocks.py), with the 30% line
       (3 of 10 features).
Run after 10_mcs_pt_blocks.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ml = pd.read_csv(TAB / "rq2_predictions.csv", index_col=0, parse_dates=True)
ml = ml[ml["model"] == "lr"]  # one model per scheme keeps the plot readable; LR has the lowest RMSE
actual = ml[ml["scheme"] == "global"]["actual"]

fig, ax = plt.subplots(figsize=(8, 4.5))
for scheme, color in (("global", "#1f77b4"), ("full", "#2ca02c"), ("domestic", "#d62728")):
    p = ml[ml["scheme"] == scheme]["pred"]
    gain = (actual ** 2 - (p - actual) ** 2).cumsum()
    ax.plot(gain.index, gain, label=f"{scheme} LR", color=color, lw=1.5)
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_ylabel("cumulative squared-error gain vs naive")
ax.set_title("RQ2: where the predictive gain accumulates (test period)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "rq2_cumulative_gain.png", dpi=150)

b = pd.read_csv(TAB / "rq3_shap_blocks.csv", parse_dates=["block_start", "block_end"])
mid = b["block_start"] + (b["block_end"] - b["block_start"]) / 2
fig, ax = plt.subplots(figsize=(8, 4.5))
for m, color in (("lr", "#1f77b4"), ("xgb", "#ff7f0e"), ("rf", "#2ca02c")):
    ax.plot(mid, b[f"dom_share_{m}"], marker="o", label=m, color=color, lw=1.5)
ax.axhline(0.3, color="grey", ls="--", lw=1, label="3 of 10 features")
ax.set_ylim(0, 1)
ax.set_ylabel("domestic share of SHAP")
ax.set_title("RQ3: is the domestic share stable across rolling blocks?")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "rq3_shap_share_blocks.png", dpi=150)
print("saved rq2_cumulative_gain.png, rq3_shap_share_blocks.png")