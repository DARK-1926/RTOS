"""Round E theory check: the INFORMATION LIMIT of early (release-time) warning.

Proposition (information limit).  At release, any monitor can only use information available before the frame
starts.  Let S be the best such summary of the contention state.  If log e2e = S + eps with eps independent of S,
then no release-time score has a better ROC than P(miss | S); in particular, with Gaussian S and eps,
    AUROC* = AUROC of the score S for the label 1{S + eps > c},   which depends only on
    R2 = Var(S)/Var(log e2e) and the miss rate p.
So the release-time AUROC of every method is limited by how much of the latency variance the (perfectly known)
contention state explains; per-frame plaxity (Toba) has R2 = 0 at release and AUROC* = 0.5 (Theorem 1).

Empirical check (on every run): S_oracle = regime label K (known to the experimenter, not to any monitor).
    R2_K   = between-regime share of log-e2e variance on the validation stream
    AUC_K  = AUROC of the oracle score E[log e2e | K] (estimated on calibration; unseen regimes get the
             validation mean, i.e. a true oracle)
    AUC*   = Gaussian formula from (R2_K, p)
compared with the release AUROC achieved by CC / KP / EWMA-CP (results/sweep_runs.csv).
Run from gen/ after sweep_analysis.py:   python info_limit.py
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sweep_analysis import runs


def gauss_auc(R2, p, n=200000, seed=0):
    if R2 <= 0:
        return 0.5
    r = np.random.default_rng(seed)
    s = r.normal(0, np.sqrt(R2), n); e = r.normal(0, np.sqrt(1 - R2), n)
    c = norm.ppf(1 - p)
    y = (s + e > c).astype(int)
    return roc_auc_score(y, s) if 0 < y.sum() < n else np.nan


def main():
    rows = []
    for cond, run, cal, val, D, _ in runs():
        if "regime" not in val or val.regime.nunique() < 2:
            continue
        y = (val.e2e > D).values.astype(int)
        if not 0 < y.sum() < len(y):
            continue
        lv = np.log(val.e2e.values)
        g = pd.Series(lv).groupby(val.regime.values)
        R2 = float(g.transform("mean").var() / lv.var())
        auc_k = roc_auc_score(y, g.transform("mean").values)
        rows.append(dict(cond=cond, run=run, D=D, miss_rate=y.mean(), R2_K=R2, AUC_oracleK=auc_k,
                         AUC_star_gauss=gauss_auc(R2, y.mean())))
        print(cond, run, round(R2, 3), round(auc_k, 3), flush=True)
    L = pd.DataFrame(rows)
    S = pd.read_csv("results/sweep_runs.csv")
    W = S.pivot_table(index=["cond", "run"], columns="method", values="auc_rel").reset_index()
    L = L.merge(W, on=["cond", "run"], how="left")
    L.to_csv("results/info_limit.csv", index=False)
    print(L[["cond", "run", "R2_K", "AUC_star_gauss", "AUC_oracleK", "Toba-eq", "CC", "KP", "EWMA-CP"]].round(3).to_string())


if __name__ == "__main__":
    main()
