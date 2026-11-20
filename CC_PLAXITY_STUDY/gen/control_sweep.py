"""EXPLORATORY sensitivity sweep (UNO Q runs): plant speed w0 and fallback noise ratio. All settings are reported."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack
from h_confirm import bg_aci
from kp_methods import static_theta
from kpf_methods import KalmanPlaxityF
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
    kpf = KalmanPlaxityF(cal, nq=64); Vf, Cf = mat(kpf.predict(val, D)), mat(kpf.predict(cal, D))
    V, C = stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc); th0 = static_theta(C[:, :2], yc, 0.05); tt = static_theta(Tc[:, :2], yc, 0.05)
    pol = {"none": modes(y, kind="none"), "always-fallback": modes(y, kind="always"), "oracle": modes(y, kind="oracle"),
           "Toba+BG": modes(y, bg_aci(T[:, :2], y, 0.05, GAMMA, tt, LAG)[0]), "KPF+BG": modes(y, bg_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG)[0]),
           "KPF+WH(4,20)": modes(y, wh_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG, m=4, K=20)[0]),
           "KPF+WH(2,20)": modes(y, wh_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG, m=2, K=20)[0])}
    for w0 in (2.0, 3.0, 4.0, 5.0):
        for ratio in (3.0, 10.0, 30.0):
            for p, md in pol.items():
                rows.append(dict(cond=cond, seed=s, h=h, w0=w0, ratio=ratio, policy=p, **simulate(md, h, w0=w0, ratio=ratio, nseed=5)))
    print(cond, s, flush=True)
R = pd.DataFrame(rows); R.to_csv("results/control_sweep.csv", index=False); pd.set_option("display.width", 250)
for c in ["J", "crash_per_1000"]:
    print(c); print(R.pivot_table(index=["h", "w0", "ratio"], columns="policy", values=c).round(3).to_string())
