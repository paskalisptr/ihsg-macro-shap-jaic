"""EDA of the IHSG return: data quality, distribution, memory, volatility clustering, calendar effects.

Tests that steer modelling choices use the training period only (train_end in config.yaml).
The test period appears in the descriptive table for reporting and is never used to pick anything.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")  # kpss warns when its p-value hits the edge of the lookup table

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

y = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)["return_ihsg"]
cut = pd.Timestamp(cfg["dates"]["train_end"])
train, test = y[y.index <= cut], y[y.index > cut]

# Quality, full sample
gaps = y.index.to_series().diff().dt.days
quality = pd.Series({"observations": len(y), "train_obs": len(train), "test_obs": len(test),
                     "longest_gap_days": gaps.max(), "gaps_over_5_days": int((gaps > 5).sum()),
                     "zero_returns": int((y == 0).sum()), "abs_z_over_4": int((np.abs(stats.zscore(y)) > 4).sum())})


def describe(s):
    return pd.Series({"n": len(s), "mean": s.mean(), "sd": s.std(), "annual_vol": s.std() * np.sqrt(252),
                      "skew": s.skew(), "excess_kurtosis": s.kurt(), "min": s.min(), "max": s.max(),
                      "share_up": (s > 0).mean(), "jarque_bera_p": stats.jarque_bera(s).pvalue})


desc = pd.DataFrame({"train": describe(train), "test (report only)": describe(test), "all": describe(y)})
yearly = y.groupby(y.index.year).agg(n="count", mean="mean", sd="std", share_up=lambda s: (s > 0).mean())

# Tests on the training period
rows = [("ADF (H0: unit root)", *adfuller(train)[:2]),
        ("KPSS (H0: stationary)", *kpss(train, nlags="auto")[:2]),
        ("ARCH-LM lag 10 (H0: no ARCH effect)", *het_arch(train, nlags=10)[:2])]
for lag in (5, 10, 20):
    rows.append((f"Ljung-Box returns, lag {lag}", *acorr_ljungbox(train, lags=[lag]).iloc[0]))
    rows.append((f"Ljung-Box squared returns, lag {lag}", *acorr_ljungbox(train**2, lags=[lag]).iloc[0]))
days, months = train.groupby(train.index.dayofweek), train.groupby(train.index.month)
rows.append(("Kruskal-Wallis by weekday", *stats.kruskal(*[g.values for _, g in days])))
rows.append(("Kruskal-Wallis by month", *stats.kruskal(*[g.values for _, g in months])))
tests = pd.DataFrame(rows, columns=["test", "statistic", "p"]).round(4)
weekday = days.agg(["count", "mean", "std"]).rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])))

for name, t in {"eda_quality": quality, "eda_target_stats": desc, "eda_yearly": yearly,
                "eda_target_tests": tests.set_index("test"), "eda_weekday": weekday}.items():
    t.to_csv(TAB / f"{name}.csv")

# Figures
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(y, lw=0.5)
ax[0].set_title("IHSG daily log return")
ax[1].plot(y.rolling(20).std(), lw=1)
ax[1].set_title("Rolling 20-day standard deviation")
for a in ax:
    a.axvline(cut, color="red", ls="--", lw=1)  # left of the line: train, right: test
fig.tight_layout()
fig.savefig(FIG / "eda_return_volatility.png", dpi=150)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for i, (s, label) in enumerate([(train, "returns"), (train**2, "squared returns")]):
    plot_acf(s, lags=30, ax=ax[i, 0], title=f"ACF, {label}")
    plot_pacf(s, lags=30, ax=ax[i, 1], title=f"PACF, {label}", method="ywm")
fig.tight_layout()
fig.savefig(FIG / "eda_acf_pacf.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
grid = np.linspace(train.min(), train.max(), 300)
ax[0].hist(train, bins=60, density=True)
ax[0].plot(grid, stats.norm.pdf(grid, train.mean(), train.std()), "r")
ax[0].set_title("Train returns vs normal density")
stats.probplot(train, plot=ax[1])
fig.tight_layout()
fig.savefig(FIG / "eda_distribution.png", dpi=150)

print(quality.to_string(), "\n\n", desc.round(4).to_string(), "\n\n", tests.to_string(index=False))