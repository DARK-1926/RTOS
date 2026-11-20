"""Exploratory diagnosis of unwarned-miss bursts under ACI (alpha 0.05) with KPT+stack on all telemetry runs."""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack, since_switch, RUNS
from kp_methods import aci_alarms, static_theta
from kpt_methods import KalmanPlaxityT
from sweep_analysis import toba_scores, GAMMA, LAG
runs = RUNS + [("pilot", "data/G13pilot", "g13", 131), ("pilot", "data/G13pilot", "g13", 132)]
rows = []
for cond, d, pre, s in runs:
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    k = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")); V, C = mat(k.predict(val, D)), mat(k.predict(cal, D))
    th0 = static_theta(C[:, :2], yc, 0.05); f, e, th = aci_alarms(V[:, :2], y, 0.05, GAMMA, th0, LAG)
    sw = since_switch(val.regime.values)
    df = pd.DataFrame(dict(run=f"{cond}_{s}", e=e, y=y, reg=val.regime.values, since=sw, theta=th, p=V[:, :2].max(1)))
    rows.append(df)
A = pd.concat(rows)
print("unwarned-miss rate by regime:\n", A.groupby("reg").e.mean().round(3))
print("by frames since switch:\n", A.groupby(pd.cut(A.since, [-1, 4, 14, 49, 1e9])).e.mean().round(3))
w = A.groupby("run").e.transform(lambda x: x.rolling(100).mean())
A["w100"] = w; top = A[A.w100 >= A.groupby("run").w100.transform("max") * 0.9]
print("frames inside the worst windows: regime mix\n", top.reg.value_counts(normalize=True).round(2))
print("theta when unwarned miss happens vs overall:", A[A.e == 1].theta.mean().round(3), A.theta.mean().round(3))
print("p(miss score) of unwarned misses: median", A[A.e == 1].p.median().round(3), " of warned misses:", A[(A.y == 1) & (A.e == 0)].p.median().round(3))
