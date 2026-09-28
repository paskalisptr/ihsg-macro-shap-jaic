# Domestic vs Global Macro Factors and IHSG Daily Returns

Reproducible pipeline for a JAIC paper comparing the contribution of domestic and global macroeconomic
variables to next-day IHSG (Jakarta Composite Index) log returns, 2021-2026, using OLS, three ML models and SHAP.

## Research questions

1. **Explanatory.** Do the domestic and the global variable groups each explain daily IHSG returns significantly? (OLS + Wald tests)
2. **Predictive.** Do domestic variables add out-of-sample predictive value beyond global ones? (ablation on 3 models + Diebold-Mariano + binomial test)
3. **Attribution.** What share of SHAP importance goes to domestic vs global variables, and is it consistent across models?

Models: Linear Regression, Random Forest, XGBoost. Primary metric: RMSE (MAE and R² supporting); directional accuracy is secondary.

## Status

| Stage | Script | State |
|---|---|---|
| 1 Acquisition | `src/01_acquire.py` | done |
| 2 Preprocessing | `src/02_preprocess.py` | done |
| 3 OLS + Wald (RQ1) | `src/03_ols_wald.py` | planned |
| 4 Models + ablation | `src/04_models.py` | planned |
| 5 DM + binomial inference (RQ2) | `src/05_inference.py` | planned |
| 6 SHAP group share (RQ3) | `src/06_shap.py` | planned |
| 7 Report tables/figures | `src/07_report.py` | planned |

## Setup and run

```bash
pip install -r requirements.txt
cp .env.example .env                    # then open .env and paste your FRED API key (free: fred.stlouisfed.org, My Account > API Keys)
python src/01_acquire.py                # downloads Yahoo + FRED, parses manual files, writes data_raw/MANIFEST.sha256
python src/02_preprocess.py             # writes data_processed/dataset_final.csv and outputs/tables/feature_checks.csv
python src/02_preprocess.py --window 20 # sensitivity runs (also try 120)
```

Automatic downloads: IHSG, USD/IDR, S&P 500, WTI, DXY, VIX (Yahoo) and Fed Funds Rate, US Treasury 10Y (FRED).
Manual (already in `data_raw/`, exact file names required): `bi_rate_raw.csv`, `indonia_raw.csv`, `epu_daily_raw.csv`.
`01_acquire.py` stops with an error if any of the 11 variables is missing after the run.

After the first successful run, freeze versions: `pip freeze > requirements.lock.txt`.
Yahoo Finance can revise history, so re-downloading later may change numbers slightly; `MANIFEST.sha256`
fingerprints the raw files used for the reported results.

## Data

| Variable | Group | Source | Transform (all predictors lagged one trading day) |
|---|---|---|---|
| USD/IDR | domestic | Yahoo `USDIDR=X` | log return |
| INDONIA | domestic | Bank Indonesia, manual CSV | first difference |
| BI Rate | domestic | Bank Indonesia, manual CSV | change over last 60 trading days |
| S&P 500, WTI, DXY | global | Yahoo | log return |
| VIX | global | Yahoo | level |
| US Treasury 10Y | global | FRED `DGS10` | first difference |
| Fed Funds Rate | global | FRED `DFF` | change over last 60 trading days |
| EPU (daily) | global | policyuncertainty.com, manual CSV | first difference |

Target: IHSG daily log return (Yahoo `^JKSE`). Master calendar: IHSG trading days; other series are forward-filled onto it.
Analysis window 2021-01-04 to 2026-06-29 (1,316 observations). Manual files in `data_raw/*_raw.csv` come from the
Bank Indonesia website and the EPU dataset; keep them under version control.

## Known limitations (to report in the paper)

- **Policy-rate features are weak by construction.** BI Rate changed on only 18 dates in the analysis window. The
  60-day change removes the near-total sparsity of a daily difference (about 50% zeros for BI Rate, 34% for Fed
  Funds, versus 99% and 95%), but ADF does not reject a unit root for either (see `outputs/tables/feature_checks.csv`),
  and this holds for windows of 20, 60 and 120 days. Use HAC standard errors in OLS, report the window sensitivity,
  and add a robustness run with the plain daily difference.
- Daily returns have a very low signal-to-noise ratio; expect modest R² and directional accuracy near 55-60%.
  A result far above that should first be audited for look-ahead leakage.
