"""Put every series on IHSG trading days, transform it, lag it one day, write data_processed/dataset_final.csv.

    python src/02_preprocess.py               # policy_window from config.yaml
    python src/02_preprocess.py --window 20   # sensitivity run, output files get a _w20 suffix
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", message=".*adfuller.*")  # statsmodels return-type notice

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
T, D, G = cfg["transform"], cfg["dates"], cfg["groups"]
W = int(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else T["policy_window"]
tag = "" if W == T["policy_window"] else f"_w{W}"

names = G["domestic"] + G["global"]
assert sorted(names) == sorted(T["log_return"] + T["diff"] + T["window_change"] + T["level"]), \
    "each variable must appear in exactly one transform list"


def load(name):
    return pd.read_csv(ROOT / "data_raw" / f"{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]


ihsg = load("ihsg").loc[D["download_start"]:]
days = ihsg.index  # IHSG trading days are the master calendar


def on_days(s):
    """Forward-fill onto IHSG trading days. Fine for daily data and for step series like policy rates."""
    return s.reindex(s.index.union(days)).ffill().reindex(days)


levels = pd.DataFrame({n: on_days(load(n)) for n in names})
prices = levels[T["log_return"]].where(lambda d: d > 0)  # WTI settled at -37.63 on 2020-04-20, log is undefined there
assert prices.loc[D["analysis_start"]:].notna().all().all(), "non-positive price inside the analysis window"

x = pd.concat([np.log(prices).diff(),
               levels[T["diff"]].diff(),
               levels[T["window_change"]].diff(W),
               levels[T["level"]]], axis=1)[names]

# Lag every predictor by one trading day, so the return on day t only sees data up to day t-1.
X = x.shift(1).add_suffix("_lag1")
df = X.join(np.log(ihsg).diff().rename("return_ihsg")).dropna().loc[D["analysis_start"]:]

assert df.index.is_monotonic_increasing and df.index.is_unique
assert (df.index[0] - pd.Timestamp(D["analysis_start"])).days < 7, "lost rows at the start, move download_start earlier"
for c in X:  # a feature equal to its same-day value means the shift never happened
    assert not np.allclose(df[c], x[c.removesuffix("_lag1")].loc[df.index]), f"{c} looks unshifted"

checks = pd.DataFrame({"zero_share": (df == 0).mean(),
                       "adf_p": {c: adfuller(df[c])[1] for c in df}}).round(4)
(ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
(ROOT / "data_processed").mkdir(exist_ok=True)
checks.to_csv(ROOT / "outputs" / "tables" / f"feature_checks{tag}.csv")
df.to_csv(ROOT / "data_processed" / f"dataset_final{tag}.csv")

print(df.shape, df.index.min().date(), "to", df.index.max().date())
print(checks)
unit_root = checks.index[checks["adf_p"] >= 0.05].tolist()
if unit_root:
    print("ADF does not reject a unit root for", unit_root, "- report it as a limitation")