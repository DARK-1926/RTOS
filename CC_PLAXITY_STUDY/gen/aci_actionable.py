"""Exploratory (existing runs): ACI on ACTIONABLE warnings = alarm by N2 start (checkpoints 0-1), the last point where
the closed loop can still degrade.  Error = frame misses and had no alarm by N2 start.  Same ACI, gamma, lag.
Run from gen/: python aci_actionable.py -> results/aci_actionable_runs.csv"""
import numpy as np, pandas as pd
from sweep_analysis import runs, all_scores, GAMMA, LAG
from kp_methods import aci_alarms, aci_bound, static_theta
rows = []
for cond, run, cal, val, D, nom in runs():
    y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    P, calP, cost, nf = all_scores(cal, val, D, nom)
    for m in calP:
        S, C = P[m][:, :2], calP[m][:, :2]
        for a in [0.02, 0.05]:
            th0 = static_theta(C, yc, a); st = S.max(1) > th0
            f, e, _ = aci_alarms(S, y, a, GAMMA, th0, LAG)
            rows.append(dict(cond=cond, run=run, method=m, alpha=a, miss_rate=y.mean(), static_unwarned=np.mean((y == 1) & ~st),
                             static_fpr=st[y == 0].mean(), aci_unwarned=e.mean(), aci_fpr=(f >= 0)[y == 0].mean(),
                             aci_bound=aci_bound(len(y), GAMMA, LAG)))
    print(cond, run, flush=True)
    pd.DataFrame(rows).to_csv("results/aci_actionable_runs.csv", index=False)
