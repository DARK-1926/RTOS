"""EXPLORATORY (not confirmatory): group-conditional ACI to fix unwarned-miss bursts (G8 failure).
Idea: one ACI threshold per group g, where g is decided at release time from information available then
(cut points fitted on calibration frames only). Each group runs its own ACI recursion on its own frames, so
each group has its own one-sided bound  mean_g(e) <= alpha + (theta0_g + (L+1) gamma) / (gamma T_g) + L / T_g.
Groupings compared:
  global    : one threshold (current method)
  mu3       : terciles of the KPT telemetry prior mean mu_t (predicted contention level)
  u3        : terciles of tm_u (other processes' CPU share)
  s3        : terciles of the release score P_0
Runs: all 8 telemetry runs (pilot 131/132, G13c 141-143, G14 151-153).
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import RUNS
from kp_methods import static_theta, aci_bound_upper
from kpt_methods import KalmanPlaxityT
from sweep_analysis import GAMMA, LAG

ALPHA = 0.05


def group_aci(Pk, y, g, alpha, gamma, theta0, lag=1):
    """theta0: dict group -> initial threshold. Frame t uses theta[g_t]; frame t-lag's error updates theta[g_{t-lag}]."""
    n = len(y); th = dict(theta0); first = np.full(n, -1); err = np.zeros(n); path = np.empty(n)
    for t in range(n):
        path[t] = th[g[t]]
        hit = np.nonzero(Pk[t] > th[g[t]])[0]
        first[t] = hit[0] if len(hit) else -1
        err[t] = float(y[t] == 1 and first[t] < 0)
        if t - lag >= 0:
            j = g[t - lag]; th[j] = th[j] - gamma * (err[t - lag] - alpha)
    return first, err, path


def cuts(x, q=(1 / 3, 2 / 3)):
    x = x[np.isfinite(x)]; return np.quantile(x, q)


def assign(x, c):
    return np.digitize(np.nan_to_num(x, nan=np.nanmedian(x)), c)


rows, per = [], []
runs = RUNS + [("pilot", "data/G13pilot", "g13", 131), ("pilot", "data/G13pilot", "g13", 132)]
for cond, d, pre, s in runs:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    k = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w"))
    V, C = mat(k.predict(val, D))[:, :2], mat(k.predict(cal, D))[:, :2]
    muv, muc = k.prior(val)[0], k.prior(cal)[0]
    G = {"global": (np.zeros(len(val), int), np.zeros(len(cal), int))}
    for name, xv, xc in [("mu3", muv, muc), ("u3", val.tm_u.values, cal.tm_u.values), ("s3", V[:, 0], C[:, 0])]:
        c = cuts(xc); G[name] = (assign(xv, c), assign(xc, c))
    for name, (gv, gc) in G.items():
        th0 = {j: static_theta(C[gc == j], yc[gc == j], ALPHA) if (gc == j).any() else 0.5 for j in np.unique(np.r_[gv, gc])}
        f, e, _ = group_aci(V, y, gv, ALPHA, GAMMA, th0, LAG)
        ok = all(e[gv == j].mean() <= ALPHA + aci_bound_upper((gv == j).sum(), GAMMA, LAG, max(th0[j], 0))
                 for j in np.unique(gv))
        r = dict(run=f"{cond}_{s}", grouping=name, unw=e.mean(), fpr=(f >= 0)[y == 0].mean(),
                 recall=(f >= 0)[y == 1].mean(), worst100=pd.Series(e).rolling(100).mean().max(), bound_ok=ok)
        for reg in ["K0", "K1", "K2", "K3"]:
            m = (val.regime == reg).values
            r[f"unw_{reg}"] = e[m].mean() if m.any() else np.nan
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.groupby("grouping")[["unw", "fpr", "recall", "worst100", "unw_K0", "unw_K1", "unw_K2", "unw_K3", "bound_ok"]].mean().round(3))
W = R.pivot(index="run", columns="grouping", values="worst100")
print("\nworst100 per run:\n", W.round(3))
for gname in ["mu3", "u3", "s3"]:
    print(f"{gname}: worst100 < global in {(W[gname] < W['global']).sum()}/{len(W)}")
R.to_csv("results/group_aci_explore.csv", index=False)
