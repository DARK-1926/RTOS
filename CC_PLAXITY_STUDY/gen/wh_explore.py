"""EXPLORATORY (old UNO Q runs only): price of the deterministic (m,K) window guarantee. Scores: KPF+stack, LR-all, Toba-eq."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack
from h_confirm import bg_aci
from kp_methods import aci_alarms, static_theta
from kpf_methods import KalmanPlaxityF
from baselines_j import LRAll
from sweep_analysis import toba_scores, GAMMA, LAG
from wh_methods import wh_aci, max_window
from j_confirm import PRIMARY, SECONDARY
MK = [(1, 10), (2, 20), (3, 50), (5, 100)]
rows = []
for cond, d, pre, s in PRIMARY + SECONDARY:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    kpf = KalmanPlaxityF(cal, nq=64); Vf, Cf = mat(kpf.predict(val, D)), mat(kpf.predict(cal, D))
    lr = LRAll(cal, D)
    S = {"KPF+stack": (stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc)), "LR-all": (mat(lr.predict(val)), lr.cal_oof()), "Toba-eq": (T, Tc)}
    for mth, (V, C) in S.items():
        th0 = static_theta(C[:, :2], yc, 0.05)
        _, e0, _ = aci_alarms(V[:, :2], y, 0.05, GAMMA, th0, LAG); fb, eb = bg_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG)
        for m, K in MK:
            fw, ew, fo = wh_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG, m=m, K=K)
            rows.append(dict(cond=cond, seed=s, method=mth, m=m, K=K, aci_max=max_window(e0, K), bg_max=max_window(eb, K), wh_max=max_window(ew, K),
                             bg_fpr=(fb >= 0)[y == 0].mean(), wh_fpr=(fw >= 0)[y == 0].mean(), forced=fo.mean(), bg_unw=eb.mean(), wh_unw=ew.mean(),
                             miss_max=max_window(y, K)))
    print(cond, s, flush=True)
R = pd.DataFrame(rows); R.to_csv("results/wh_explore.csv", index=False)
R["bound"] = R.m + LAG - 1; R["aci_viol"] = R.aci_max > R.bound; R["bg_viol"] = R.bg_max > R.bound; R["wh_viol"] = R.wh_max > R.bound; R["dfpr"] = R.wh_fpr - R.bg_fpr
pd.set_option("display.width", 250)
print(R.groupby(["method", "m", "K"])[["miss_max", "aci_max", "bg_max", "wh_max", "aci_viol", "bg_viol", "wh_viol", "bg_fpr", "wh_fpr", "dfpr", "forced", "wh_unw"]].mean().round(3).to_string())
