"""RQ1: OLS with robust standard errors, then Wald tests on the domestic block and the global block."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parents[1]
groups = yaml.safe_load((ROOT / "config.yaml").read_text())["groups"]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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

wald.round(4).to_csv(TAB / "rq1_wald.csv", index=False)
coef.round(6).to_csv(TAB / "rq1_coefficients.csv")

# Coefficients with 95% HAC confidence intervals, colored by group, drop constant
group_of = {f"{n}_lag1": g for g, cols in blocks.items() for n in groups[g]}
plot_coef = coef.drop("const").reindex(coef.drop("const")["beta"].abs().sort_values().index)
colors = ["#1f77b4" if group_of[i] == "domestic" else "#d62728" for i in plot_coef.index]

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (name, row) in enumerate(plot_coef.iterrows()):
    ax.errorbar(row["beta"], i, xerr=1.96 * row["hac_se"], fmt="o", color=colors[i])
ax.set_yticks(range(len(plot_coef)), plot_coef.index)
ax.axvline(0, color="grey", lw=0.8)
ax.set_title(f"OLS coefficients, 95% HAC CI ({lags} lags)")
handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="") for c in ("#1f77b4", "#d62728")]
ax.legend(handles, ["domestic", "global"], loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "rq1_coefficients.png", dpi=150)

# Residual diagnostics: the two things a reader checks first on an OLS fit
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].scatter(main.fittedvalues, main.resid, s=8, alpha=0.5)
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xlabel("fitted"), ax[0].set_ylabel("residual"), ax[0].set_title("Residuals vs fitted")
stats.probplot(main.resid, plot=ax[1])
ax[1].set_title("Residual QQ-plot")
fig.tight_layout()
fig.savefig(FIG / "rq1_residual_diagnostics.png", dpi=150)

print(wald.round(4).to_string(index=False))
print(coef.round(5))
print(f"R2={main.rsquared:.4f}  adjR2={main.rsquared_adj:.4f}  DW={durbin_watson(main.resid):.3f}  max VIF={vif.max():.2f}  n={len(df)}")