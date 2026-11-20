"""Addendum K: deadline flexibility without retraining.  python k_confirm.py [smoke]
Fit once at D0 = q85 (calibration), evaluate at D' in q75/q80/q90/q95 without refitting (see PROTOCOL Addendum K)."""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import auroc, ece
from g13_analysis import build, mat
from kpf_methods import KalmanPlaxityF
from baselines_j import LRAll, GBDTAll
from j_confirm import PRIMARY, SECONDARY, SMOKE

QS = [0.75, 0.80, 0.90, 0.95]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def m2(V, y):
    return dict(auc_act=auroc(V[:, :2].max(1), y), ece01=0.5 * (ece(V[:, 0], y) + ece(V[:, 1], y)))


def run_one(cond, d, pre, s, primary):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D0 = float(np.round(np.quantile(cal.e2e, 0.85)))
    kpf = KalmanPlaxityF(cal, nq=64)
    lr0, gb0 = mat(LRAll(cal, D0).predict(val)), mat(GBDTAll(cal, D0).predict(val))
    rows = []
    for q in QS:
        D = float(np.round(np.quantile(cal.e2e, q))); y = (val.e2e > D).values.astype(int)
        S = {"KPF": mat(kpf.predict(val, D)), "LR-all@D0": lr0, "GBDT-all@D0": gb0, "LR-all retrained": mat(LRAll(cal, D).predict(val))}
        for m, V in S.items():
            rows.append(dict(cond=cond, seed=s, primary=primary, q=q, D=D, D0=D0, miss=y.mean(), method=m, **m2(V, y)))
    say(f"--- {cond} seed {s}: D0={D0} D'={[r['D'] for r in rows[::4]]}")
    return rows


def main(mode):
    runs = [(r, False) for r in SMOKE] if mode == "smoke" else [(r, True) for r in PRIMARY] + [(r, False) for r in SECONDARY]
    rows = []
    for run, pr in runs:
        rows += run_one(*run, pr)
    R = pd.DataFrame(rows); tag = "smoke" if mode == "smoke" else "confirm"; R.to_csv(f"results/k_{tag}_runs.csv", index=False)
    pd.set_option("display.width", 250)
    if mode != "smoke":
        P = R[R.primary]; pv = lambda c: P.pivot_table(index=["cond", "seed", "q"], columns="method", values=c)
        a, e = pv("auc_act"), pv("ece01")
        chk = [("K1 ECE: KPF < LR-all@D0", int((e["KPF"] < e["LR-all@D0"]).sum()), 20, 24),
               ("K2 ECE: KPF < GBDT-all@D0", int((e["KPF"] < e["GBDT-all@D0"]).sum()), 20, 24),
               ("K3 act AUROC: KPF >= LR-all@D0 - 0.01", int((a["KPF"] >= a["LR-all@D0"] - 0.01).sum()), 20, 24),
               ("K4 act AUROC: KPF >= retrained LR-all - 0.02", int((a["KPF"] >= a["LR-all retrained"] - 0.02).sum()), 20, 24)]
        say("\n===== Addendum K (primary: 6 runs x 4 deadlines) =====")
        for name, x, need, n in chk:
            say(f"{name:50s} {x}/{n} (rule >={need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
    say("\nmeans by primary, deadline quantile, method:\n" +
        R.groupby(["primary", "q", "method"])[["auc_act", "ece01"]].mean().unstack("method").round(3).to_string())
    open(f"results/k_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "confirm")
