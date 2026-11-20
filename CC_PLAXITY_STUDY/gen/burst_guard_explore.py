"""EXPLORATORY: controllers that target local (windowed) coverage of unwarned misses, on the 8 telemetry runs.
Burst guard (BG): effective threshold  theta_eff_t = theta_t - kappa * [ sum of the last W KNOWN errors - alpha W ]_+ ,
where theta_t is the ordinary ACI state updated with the errors of the effective rule.
Guarantee: the guard only lowers the alarm threshold. If theta_t < 0 then theta_eff_t < 0, every frame alarms and e = 0, so the
state's lower barrier is unchanged and the ACI one-sided bound (alpha + (theta0 + (L+1) gamma)/(gamma T) + L/T) still holds.
Compared: ACI at gamma 0.01 (current), 0.03, 0.05; BG(W, kappa); group(mu3) + BG.
Metrics are reported at matched FPR too: we also find the gamma-0.01 ACI alpha that gives the same FPR as each controller,
so a controller cannot look better just by alarming more.
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import RUNS
from kp_methods import static_theta, aci_bound_upper
from kpt_methods import KalmanPlaxityT
from sweep_analysis import LAG

ALPHA = 0.05


def ctrl(Pk, y, g, alpha, gamma, theta0, lag=1, W=0, kappa=0.0):
    n = len(y); th = dict(theta0); first = np.full(n, -1); err = np.zeros(n)
    for t in range(n):
        lo = max(0, t - lag - W + 1); hi = t - lag + 1          # known errors: frames < = t-lag
        guard = kappa * max(0.0, err[lo:hi].sum() - alpha * W) if (W and hi > 0) else 0.0
        theta = th[g[t]] - guard
        hit = np.nonzero(Pk[t] > theta)[0]
        first[t] = hit[0] if len(hit) else -1
        err[t] = float(y[t] == 1 and first[t] < 0)
        if t - lag >= 0:
            j = g[t - lag]; th[j] = th[j] - gamma * (err[t - lag] - alpha)
    return first, err


data = []
runs = RUNS + [("pilot", "data/G13pilot", "g13", 131), ("pilot", "data/G13pilot", "g13", 132)]
for cond, d, pre, s in runs:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    k = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w"))
    V, C = mat(k.predict(val, D))[:, :2], mat(k.predict(cal, D))[:, :2]
    muv, muc = k.prior(val)[0], k.prior(cal)[0]
    c = np.quantile(muc[np.isfinite(muc)], [1 / 3, 2 / 3])
    data.append((f"{cond}_{s}", V, C, y, yc, val.regime.values, np.digitize(muv, c), np.digitize(muc, c)))

CTRLS = {"ACI g.01": dict(gamma=0.01), "ACI g.03": dict(gamma=0.03), "ACI g.05": dict(gamma=0.05),
         "BG W20 k.05": dict(gamma=0.01, W=20, kappa=0.05), "BG W20 k.10": dict(gamma=0.01, W=20, kappa=0.10),
         "BG W30 k.10": dict(gamma=0.01, W=30, kappa=0.10), "BG W50 k.05": dict(gamma=0.01, W=50, kappa=0.05),
         "BG W20 k.20": dict(gamma=0.01, W=20, kappa=0.20)}
rows = []
for run, V, C, y, yc, reg, gv, gc in data:
    for grp in ["global", "mu3"]:
        GV, GC = (np.zeros(len(y), int), np.zeros(len(yc), int)) if grp == "global" else (gv, gc)
        for name, p in CTRLS.items():
            th0 = {j: static_theta(C[GC == j], yc[GC == j], ALPHA) if (GC == j).any() else 0.5 for j in range(3)}
            f, e = ctrl(V, y, GV, ALPHA, p["gamma"], th0, LAG, p.get("W", 0), p.get("kappa", 0.0))
            ok = all(e[GV == j].mean() <= ALPHA + aci_bound_upper((GV == j).sum(), p["gamma"], LAG, max(th0[j], 0))
                     for j in np.unique(GV))
            fpr = (f >= 0)[y == 0].mean()
            # matched-FPR reference: plain gamma-.01 global ACI with alpha chosen so its FPR is closest to this one's
            best = None
            for a in np.arange(0.005, 0.0801, 0.0025):
                t0 = {0: static_theta(C, yc, a)}
                f2, e2 = ctrl(V, y, np.zeros(len(y), int), a, 0.01, t0, LAG)
                d_ = abs((f2 >= 0)[y == 0].mean() - fpr)
                if best is None or d_ < best[0]:
                    best = (d_, pd.Series(e2).rolling(100).mean().max(), e2.mean())
            rows.append(dict(run=run, grp=grp, ctrl=name, unw=e.mean(), fpr=fpr, worst100=pd.Series(e).rolling(100).mean().max(),
                             unw_K2=e[reg == "K2"].mean(), bound_ok=ok, ref_worst100=best[1], ref_unw=best[2]))
R = pd.DataFrame(rows); R.to_csv("results/burst_guard_explore.csv", index=False)
pd.set_option("display.width", 250)
A = R.groupby(["grp", "ctrl"])[["unw", "fpr", "worst100", "ref_worst100", "ref_unw", "unw_K2", "bound_ok"]].mean().round(3)
A["wins_vs_matchedFPR"] = R.assign(w=R.worst100 < R.ref_worst100).groupby(["grp", "ctrl"]).w.sum()
A["wins_vs_ACI01"] = R.merge(R[(R.grp == "global") & (R.ctrl == "ACI g.01")][["run", "worst100"]], on="run", suffixes=("", "_b")) \
    .assign(w=lambda x: x.worst100 < x.worst100_b).groupby(["grp", "ctrl"]).w.sum()
print(A.to_string())
