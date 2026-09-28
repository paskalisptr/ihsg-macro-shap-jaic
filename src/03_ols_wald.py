"""RQ1: OLS with robust standard errors, then Wald tests on the domestic block and the global block."""
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import yaml
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parents[1]
groups = yaml.safe_load((ROOT / "config.yaml").read_text())["groups"]

df = pd.read_csv(ROOT / "data_processed" / "dataset_final.csv", index_col=0, parse_dates=True)
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
Z = sm.add_constant((X - X.mean()) / X.std())  # z-scored predictors
blocks = {g: [f"{n}_lag1" for n in groups[g]] for g in ("domestic", "global")}
lags = int(4 * (len(df) / 100) ** (2 / 9))  # Newey-West rule of thumb, 7 for this sample

# HAC is the main specification, HC3 the robustness check. HAC with 60 lags rejects a true null
# in 13% of placebo runs at the 5% level, so it is not used.
fits = {f"HAC{lags}": sm.OLS(y, Z).fit(cov_type="HAC", cov_kwds={"maxlags": lags}),
        "HC3": sm.OLS(y, Z).fit(cov_type="HC3")}
main = fits[f"HAC{lags}"]

rows = []
for label, res in fits.items():
    for g, cols in blocks.items():
        w = res.wald_test(", ".join(f"{c} = 0" for c in cols), use_f=True, scalar=True)
        rows.append({"std_errors": label, "group": g, "restrictions": len(cols),
                     "F": float(w.statistic), "p": float(w.pvalue)})
wald = pd.DataFrame(rows)
coef = pd.DataFrame({"beta": main.params, "hac_se": main.bse, "p": main.pvalues})
vif = pd.Series([variance_inflation_factor(Z.values, i) for i in range(1, Z.shape[1])], index=X.columns)

wald.round(4).to_csv(ROOT / "outputs" / "tables" / "rq1_wald.csv", index=False)
coef.round(6).to_csv(ROOT / "outputs" / "tables" / "rq1_coefficients.csv")

print(wald.round(4).to_string(index=False))
print(coef.round(5))
print(f"R2={main.rsquared:.4f}  adjR2={main.rsquared_adj:.4f}  DW={durbin_watson(main.resid):.3f}  max VIF={vif.max():.2f}  n={len(df)}")