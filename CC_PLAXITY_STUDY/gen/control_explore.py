"""EXPLORATORY: control case study on the 19 UNO Q runs (all already seen). Policies at alpha 0.05."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack
from h_confirm import bg_aci
from kp_methods import static_theta
from kpf_methods import KalmanPlaxityF
from baselines_j import LRAll
from sweep_analysis import toba_scores, GAMMA, LAG
from wh_methods import wh_aci
from control_study import simulate, modes
from j_confirm import PRIMARY, SECONDARY
rows = []
for cond, d, pre, s in PRIMARY + SECONDARY:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    h = 0.2 if pre == "vo" else 0.1
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    kpf = KalmanPlaxityF(cal, nq=64); Vf, Cf = mat(kpf.predict(val, D)), mat(kpf.predict(cal, D)); lr = LRAll(cal, D)
    S = {"KPF": (stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc)), "LR": (mat(lr.predict(val)), lr.cal_oof()), "Toba": (T, Tc)}
    pol = {"none": modes(y, kind="none"), "always-fallback": modes(y, kind="always"), "oracle": modes(y, kind="oracle")}
    for m_, (V, C) in S.items():
        th0 = static_theta(C[:, :2], yc, 0.05)
        pol[m_ + "+BG"] = modes(y, bg_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG)[0])
        for mm, K in [(4, 20), (2, 20)]:
            pol[f"{m_}+WH({mm},{K})"] = modes(y, wh_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG, m=mm, K=K)[0])
    for p, md in pol.items():
        rows.append(dict(cond=cond, seed=s, h=h, policy=p, miss=y.mean(), fallback=(md == 1).mean(), hold=(md == 2).mean(), **simulate(md, h)))
    print(cond, s, flush=True)
R = pd.DataFrame(rows); R.to_csv("results/control_explore.csv", index=False); pd.set_option("display.width", 250)
print(R.groupby(["h", "policy"])[["fallback", "hold", "J", "crash_per_1000", "p99_angle"]].mean().round(4).to_string())
