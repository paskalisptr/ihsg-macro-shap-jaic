"""Parse the three hand-copied files in data_raw/ (BI Rate, INDONIA, EPU), then check that all 11 variables exist.

Copy bi_rate_raw.csv, indonia_raw.csv and epu_daily_raw.csv from the old project. Do not open and re-save them
in Excel. Run this after 01_acquire.py.
"""
import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
RAW = ROOT / "data_raw"
start, end = cfg["dates"]["download_start"], cfg["dates"]["end"]

# Indonesian month names, matched on the first three letters ("Agustus" and "Agt" both work)
MONTHS = dict(jan=1, feb=2, mar=3, apr=4, mei=5, jun=6, jul=7, agu=8, agt=8, sep=9, okt=10, nov=11, des=12)


def parse_date(x):
    """'18 Juni 2026' and '3 Jan 2020' are read by hand. Excel-style dates ('23-Jun-26', '02/01/2020') go to pandas."""
    try:
        d, m, y = str(x).split()
        return pd.Timestamp(int(y), MONTHS[m[:3].lower()], int(d))
    except (ValueError, KeyError):
        return pd.to_datetime(x, dayfirst=True, errors="coerce")


assert parse_date("29 Mei 2026") == pd.Timestamp(2026, 5, 29)
assert parse_date("23-Jun-26") == pd.Timestamp(2026, 6, 23)


def read_manual(name, must_have):
    """Excel writes ';' or ',' depending on regional settings, so try both."""
    assert (RAW / name).exists(), f"copy {name} into data_raw/ first"
    for sep in (";", ","):
        df = pd.read_csv(RAW / name, sep=sep)
        df.columns = df.columns.str.replace("<br>", " ", regex=False).str.strip()  # headers pasted from a web page
        if must_have in df.columns:
            return df
    raise AssertionError(f"{name}: column '{must_have}' not found, check the header")


def to_series(df, date_col, value_col, name):
    df = df.assign(date=df[date_col].map(parse_date))
    bad = df["date"].isna()  # a dropped BI Rate change would shift every later level, so stop instead
    assert not bad.any(), f"{name}: {bad.sum()} unreadable dates, first: {df.loc[bad, date_col].iloc[0]!r}"
    v = df[value_col].astype(str).str.replace("%", "").str.replace(",", ".").str.strip().astype(float)
    s = pd.Series(v.values, index=df["date"], name=name).sort_index()
    assert s.index.is_unique, f"{name}: duplicated dates"
    return s


e = read_manual("epu_daily_raw.csv", "daily_policy_index")
epu = pd.Series(e["daily_policy_index"].values, index=pd.to_datetime(e[["year", "month", "day"]]), name="epu").sort_index()
assert epu.index.is_unique, "epu: duplicated dates"

bi = to_series(read_manual("bi_rate_raw.csv", "BI-7Day-RR"), "Tanggal", "BI-7Day-RR", "bi_rate")
ind = to_series(read_manual("indonia_raw.csv", "IndONIA (%)"), "Tanggal Publikasi", "IndONIA (%)", "indonia")
assert bi.between(0.5, 20).all() and ind.between(0.5, 20).all(), "rate outside 0.5-20: check decimals or % conversion"

# EPU and INDONIA are cut to the download window. BI Rate is not: the file lists rate changes only,
# and cutting it would drop the level in force at the start.
epu, ind = epu.loc[start:end], ind.loc[start:end]
for s in (epu, ind):  # a truncated copy-paste would shorten the sample without any error
    assert s.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=7), f"{s.name} starts too late"
    assert s.index.max() >= pd.Timestamp(end) - pd.Timedelta(days=7), f"{s.name} ends too early"
for s in (epu, bi, ind):
    s.to_csv(RAW / f"{s.name}.csv")
    print(f"{s.name:8s} {s.index.min().date()} to {s.index.max().date()}  n={len(s)}")

need = ["ihsg"] + cfg["groups"]["domestic"] + cfg["groups"]["global"]
missing = [n for n in need if not (RAW / f"{n}.csv").exists()]
assert not missing, f"missing raw files: {missing}"
(RAW / "MANIFEST.sha256").write_text(  # fingerprints to prove later that the raw data did not change
    "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in sorted(RAW.glob("*.csv"))))
print(f"all {len(need)} variables present, MANIFEST.sha256 written")