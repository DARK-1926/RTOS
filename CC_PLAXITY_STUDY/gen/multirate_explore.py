"""Exploratory (critique E1): local coverage. Compare single-rate ACI (gamma 0.005/0.01/0.02/0.05) with MIN-OF-EXPERTS ACI:
every expert j runs its own ACI threshold on its own counterfactual errors; the monitor alarms if ANY expert alarms
(theta = min_j theta_j). Then e_ours <= e_j for every j and frame, so the long-run bound of each expert still holds.
Metrics: long-run unwarned rate, worst 100/300-frame window, FPR.   python multirate_explore.py"""
import numpy as np, pandas as pd
from report_curves import g11_runs
from sweep_analysis import all_scores
from kp_methods import static_theta

def experts(S, y, alpha, gammas, th0, lag=1):
    n = len(y); s = S.max(1); th = np.full(len(gammas), th0, float); ej = np.zeros((n, len(gammas))); al = np.zeros(n, bool)
    for t in range(n):
        aj = s[t] > th; al[t] = aj.any()
        ej[t] = (y[t] == 1) & ~aj
        if t - lag >= 0:
            th = th - np.array(gammas) * (ej[t - lag] - alpha)
    return al, ej

rows = []
for cond, run, cal, val, D, nom in g11_runs():
    y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int); P, C, *_ = all_scores(cal, val, D, nom)
    for m in ["KP", "Toba-eq"]:
        S = P[m][:, :2]; th0 = static_theta(C[m][:, :2], yc, 0.05)
        G = [0.005, 0.01, 0.02, 0.05]
        for name, gs in [(f"ACI g={g}", [g]) for g in G] + [("min-of-experts", G)]:
            al, ej = experts(S, y, 0.05, gs, th0)
            e = ((y == 1) & ~al).astype(float)
            rows.append(dict(run=run, method=m, aci=name, unw=e.mean(), worst100=pd.Series(e).rolling(100).mean().max(),
                             worst300=pd.Series(e).rolling(300).mean().max(), fpr=al[y == 0].mean()))
    print(run, flush=True)
R = pd.DataFrame(rows); R.to_csv("results/multirate_g11.csv", index=False)
pd.set_option("display.width", 200); print(R.groupby(["method", "aci"])[["unw", "worst100", "worst300", "fpr"]].mean().round(3))
