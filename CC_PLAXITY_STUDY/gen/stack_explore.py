"""Exploratory: late-checkpoint stacking. For checkpoints 0-1 use KP; for each later checkpoint k fit, on CALIBRATION frames
only, a logistic model y ~ logit(KP_k) + logit(Toba_k) and use its probability on validation. (Toba's N6 check is folded into
the last checkpoint as max.)  python stack_explore.py [g11|all] -> results/stack_explore_<tag>.csv"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sweep_analysis import runs, all_scores, sweep_metrics
from report_curves import g11_runs

def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

tag = sys.argv[1] if len(sys.argv) > 1 else "g11"; rows = []
for cond, run, cal, val, D, nom in (g11_runs() if tag == "g11" else runs()):
    y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    P, C, *_ = all_scores(cal, val, D, nom)
    K, Kc, T, Tc = P["KP"], C["KP"], P["Toba-eq"], C["Toba-eq"]
    T3, T3c = np.maximum(T[:, 3], T[:, 4]), np.maximum(Tc[:, 3], Tc[:, 4])
    H = [K[:, 0], K[:, 1]]
    for k, (tv, tc) in zip([2, 3], [(T[:, 2], Tc[:, 2]), (T3, T3c)]):
        Xc = np.column_stack([lg(Kc[:, k]), lg(tc)]); Xv = np.column_stack([lg(K[:, k]), lg(tv)])
        if 0 < yc.sum() < len(yc):
            m = LogisticRegression(C=1.0).fit(Xc, yc); H.append(m.predict_proba(Xv)[:, 1])
        else:
            H.append(K[:, k])
    H = np.column_stack(H)
    for name, S in [("Toba-eq", T), ("KP", K), ("KP+stack", H)]:
        rows.append(dict(cond=cond, run=run, method=name, **sweep_metrics(S, y, val, D)))
    print(cond, run, flush=True)
R = pd.DataFrame(rows); R.to_csv(f"results/stack_explore_{tag}.csv", index=False)
print(R.groupby("method")[["auc_act", "auc_any", "ap_any", "recall_fpr10", "lead_ms_fpr10"]].mean().round(3))
for c in ["auc_any", "ap_any"]:
    W = R.pivot_table(index="run", columns="method", values=c)
    print(c, "stack>=Toba", (W["KP+stack"] >= W["Toba-eq"]).sum(), "stack>=KP", (W["KP+stack"] >= W["KP"]).sum(), "of", len(W))
