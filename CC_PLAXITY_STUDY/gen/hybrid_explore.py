"""Exploratory (Round G design): KP early + Toba late. Score matrix [KP0, KP1, Toba(N3/N4), Toba(N5), Toba(N6)].
Metrics as in sweep_analysis. Run from gen/: python hybrid_explore.py [g11|all]"""
import sys, numpy as np, pandas as pd
from sweep_analysis import runs, all_scores, sweep_metrics
from report_curves import g11_runs
rows = []
src = g11_runs() if (len(sys.argv) < 2 or sys.argv[1] == "g11") else runs()
for cond, run, cal, val, D, nom in src:
    y = (val.e2e > D).values.astype(int); P, *_ = all_scores(cal, val, D, nom)
    H = np.column_stack([P["KP"][:, 0], P["KP"][:, 1], P["Toba-eq"][:, 2], P["Toba-eq"][:, 3], P["Toba-eq"][:, 4]])
    for m, S in [("Toba-eq", P["Toba-eq"]), ("KP", P["KP"]), ("KP+Toba-late", H)]:
        rows.append(dict(cond=cond, run=run, method=m, **sweep_metrics(S, y, val, D)))
    print(cond, run, flush=True)
R = pd.DataFrame(rows); R.to_csv(f"results/hybrid_explore_{sys.argv[1] if len(sys.argv) > 1 else 'g11'}.csv", index=False)
print(R.groupby("method")[["auc_rel", "auc_act", "auc_any", "ap_any", "recall_fpr10", "lead_ms_fpr10"]].mean().round(3))
