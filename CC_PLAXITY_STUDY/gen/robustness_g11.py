"""Critique fixes D3 + E1 on G11 (real ROS 2). Descriptive robustness, not a new hypothesis test.
D3: deadlines at calibration p75/p85/p90/p95 -> actionable AUROC (KP vs Toba, block-bootstrap CI), ACI price at alpha 0.05.
E1: worst rolling-window unwarned-actionable-miss rate (windows 100/300/500 frames) under ACI, KP and Toba scores.
python robustness_g11.py -> results/robust_g11.csv"""
import numpy as np, pandas as pd
from report_curves import g11_runs
from sweep_analysis import all_scores, GAMMA, LAG
from kp_methods import aci_alarms, static_theta
from cc_plaxity import block_boot_diff, auroc
rows = []
for cond, run, cal, val, _, nom in g11_runs():
    for q in [0.75, 0.85, 0.90, 0.95]:
        D = float(np.round(np.quantile(cal.e2e, q))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
        P, C, *_ = all_scores(cal, val, D, nom); b = np.arange(len(val)) // 50
        d = block_boot_diff(P["KP"][:, :2].max(1), P["Toba-eq"][:, :2].max(1), y, b, auroc)
        r = dict(run=run, q=q, D=D, miss=y.mean(), auc_act_KP=auroc(P["KP"][:, :2].max(1), y),
                 auc_act_Toba=auroc(P["Toba-eq"][:, :2].max(1), y), dAUC=d[0], lo=d[1], hi=d[2])
        for m in ["KP", "Toba-eq", "EWMA-CP"]:
            th0 = static_theta(C[m][:, :2], yc, 0.05); f, e, _ = aci_alarms(P[m][:, :2], y, 0.05, GAMMA, th0, LAG)
            r[f"fpr_{m}"] = (f >= 0)[y == 0].mean(); r[f"unw_{m}"] = e.mean()
            for w in [100, 300, 500]:
                r[f"worst{w}_{m}"] = pd.Series(e).rolling(w).mean().max()
        rows.append(r); print(run, q, flush=True)
R = pd.DataFrame(rows); R.to_csv("results/robust_g11.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string())
