"""RQ3 consistency: do OLS (RQ1) and SHAP (RQ3) rank the 10 predictors the same way?

OLS importance = |beta| from the z-scored full model (03_ols_wald.py), so coefficients are
comparable across variables. SHAP importance = mean |SHAP| per feature on the test set
(06_SHAP.py), one column per model. Agreement is Spearman rank correlation and cosine similarity.
Run after 03_ols_wald.py and 06_SHAP.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"

ols = pd.read_csv(TAB / "rq1_coefficients.csv", index_col=0).drop(index="const")["beta"].abs()
shap_tab = pd.read_csv(TAB / "rq3_shap_per_feature.csv", index_col=0)
ols = ols.loc[shap_tab.index]  # same feature order in both tables

rows = []
for model in shap_tab.columns:
    s = shap_tab[model]
    rho, p = stats.spearmanr(ols, s)
    cos = float(np.dot(ols, s) / (np.linalg.norm(ols) * np.linalg.norm(s)))
    rows.append({"shap_model": model, "spearman": rho, "spearman_p": p, "cosine": cos})
out = pd.DataFrame(rows)
out.round(4).to_csv(TAB / "rq3_rank_agreement.csv", index=False)
print(out.round(4).to_string(index=False))