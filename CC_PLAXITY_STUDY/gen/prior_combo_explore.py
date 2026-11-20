"""EXPLORATORY: which prior generalises across pipelines? KP (EWMA history), KPT (telemetry only, u r w), KPTH (history + telemetry,
m u r w; nested, so it cannot fit calibration worse than either). 15 runs: lidar G13pilot/G13c/G14/H13/H14 (14) + EuRoC drone 181 (1)."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import since_switch
from kp_methods import KalmanPlaxity
from kpt_methods import KalmanPlaxityT
from kpti_methods import KalmanPlaxityTI
from kpf_methods import KalmanPlaxityF
from cc_plaxity import auroc, ece
RUNS = [("pilot", "data/G13pilot", "g13", s) for s in (131, 132)] + [("G13c", "data/G13c", "g13", s) for s in (141, 142, 143)] + \
       [("G14", "data/G14", "g14", s) for s in (151, 152, 153)] + [("H13", "data/H13", "g13", s) for s in (161, 162, 163)] + \
       [("H14", "data/H14", "g14", s) for s in (171, 172, 173)] + [("EUROC", "data/H15", "vo", 181)]
rows = []
for cond, d, pre, s in RUNS:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int)
    reg = val.regime.values; k3 = reg == "K3"; sw = since_switch(reg) < 15
    M = {"KP": KalmanPlaxity(cal, nq=64), "KPT": KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")),
         "KPTH": KalmanPlaxityT(cal, nq=64, features=("m", "u", "r", "w")),
         "KPTI": KalmanPlaxityTI(cal, nq=64), "KPF": KalmanPlaxityF(cal, nq=64)}
    for m, k in M.items():
        V = mat(k.predict(val, D))
        rows.append(dict(cond=cond, seed=s, method=m, act=auroc(V[:, :2].max(1), y), rel=auroc(V[:, 0], y),
                         sw=auroc(V[sw, 0], y[sw]), ece_k3=ece(V[k3, 0], y[k3]), r2=getattr(k, "r2_prior", np.nan), rho=getattr(k, "rho", np.nan)))
R = pd.DataFrame(rows); R.to_csv("results/prior_combo_explore.csv", index=False)
pd.set_option("display.width", 200)
P = lambda c: R.pivot_table(index=["cond", "seed"], columns="method", values=c)
for c in ["act", "rel", "sw", "ece_k3"]:
    T = P(c); lo = c == "ece_k3"; best = T[["KP", "KPT"]].min(1) if lo else T[["KP", "KPT"]].max(1)
    print(f"== {c}  mean: {T.mean().round(3).to_dict()}")
    for m in ["KPTH", "KPTI", "KPF"]:
        b1 = (T[m] < T.KP) if lo else (T[m] > T.KP); b2 = (T[m] < T.KPT) if lo else (T[m] > T.KPT)
        loss = (T[m] - best) if lo else (best - T[m])
        print(f"  {m:7s} beats KP {b1.sum():2d}/15, beats KPT {b2.sum():2d}/15, worst loss vs best-of(KP,KPT) {loss.max():.3f}, EUROC {T.loc[('EUROC',181), m]:.3f}")

print("KPF rho chosen on calibration:", R[R.method == "KPF"].set_index(["cond", "seed"]).rho.to_dict())
