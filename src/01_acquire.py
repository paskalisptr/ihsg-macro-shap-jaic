"""Download market data (Yahoo Finance, FRED) into data_raw/. BI Rate, INDONIA and EPU are not touched here."""
from pathlib import Path

import yaml
import yfinance as yf
from fredapi import Fred

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / "data_raw"
start, end = cfg["dates"]["download_start"], cfg["dates"]["end"]

if not cfg["fred_api_key"]:
    raise SystemExit("Paste your FRED API key into fred_api_key in config.yaml first.")
RAW.mkdir(exist_ok=True)

px = yf.download(list(cfg["yahoo"]), start=start, end=end, auto_adjust=True)["Close"].rename(columns=cfg["yahoo"])
for name in px:
    px[name].dropna().to_csv(RAW / f"{name}.csv")

fred = Fred(api_key=cfg["fred_api_key"])
for series_id, name in cfg["fred"].items():
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.rename(name).ffill().to_csv(RAW / f"{name}.csv")  # ffill covers US holidays that fall on weekdays

print("downloaded:", sorted(p.stem for p in RAW.glob("*.csv") if not p.stem.endswith("_raw")))