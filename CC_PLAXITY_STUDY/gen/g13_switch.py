"""Exploratory: frames shortly after a contention switch (first N frames after the regime label changes, validation).
Release AUROC / ECE / actionable recall at FPR 10% on those frames. python g13_switch.py <dir> <seeds...>"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import auroc, ece, CP_DONE
from g13_analysis import build, mat
from kp_methods import KalmanPlaxity
from kpt_methods import KalmanPlaxityT
from sweep_analysis import EWMACP, toba_scores
d, seeds = sys.argv[1], [int(x) for x in sys.argv[2:]]; rows = []
for s in seeds:
    cal = build(f"{d}/g13_s{s}_calib.csv", "calib", 0); val = build(f"{d}/g13_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int)
    reg = val.regime.values; ch = np.r_[True, reg[1:] != reg[:-1]]
    since = np.zeros(len(val), int); c = 0
    for i in range(len(val)):
        c = 0 if ch[i] else c + 1; since[i] = c
    S = {"KP": mat(KalmanPlaxity(cal, nq=64).predict(val, D)), "EWMA-CP": mat(EWMACP(cal).predict(val, D)),
         "KP-T": mat(KalmanPlaxityT(cal, nq=64).predict(val, D)),
         "KP-T(tel only)": mat(KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")).predict(val, D)), "Toba-eq": toba_scores(cal, val, D)}
    for win, sel in [("first 5 after switch", since < 5), ("first 15 after switch", since < 15), ("steady (>=15)", since >= 15)]:
        for m, V in S.items():
            rows.append(dict(seed=s, window=win, method=m, n=sel.sum(), miss=y[sel].mean(), auc_rel=auroc(V[sel, 0], y[sel]),
                             auc_act=auroc(V[sel, :2].max(1), y[sel]), ece_rel=ece(V[sel, 0], y[sel])))
R = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(R.groupby(["window", "method"])[["n", "miss", "auc_rel", "auc_act", "ece_rel"]].mean().round(3).to_string())
