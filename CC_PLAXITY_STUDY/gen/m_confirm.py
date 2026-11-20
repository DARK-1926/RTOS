"""Addendum M: deterministic weakly-hard (m, K) guarantee on unwarned misses (wh_methods.wh_aci), tested on the Pi 5 runs.
    python m_confirm.py [smoke]   -> results/m_confirm_runs.csv, results/m_confirm_summary.txt
Frozen BEFORE any Pi confirmatory data was pulled. Budgets fixed from exploration on the 19 UNO Q runs (wh_explore.py)."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack
from h_confirm import bg_aci
from kp_methods import aci_alarms, aci_bound_upper, static_theta
from kpf_methods import KalmanPlaxityF
from baselines_j import LRAll
from sweep_analysis import toba_scores, GAMMA, LAG
from wh_methods import wh_aci, max_window
from l_confirm import RUNS

MK = [(4, 20), (6, 50), (10, 100)]; ALPHA = 0.05
SMOKE = [("pilot", "data/G13pilot", "g13", 131)]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def run_one(cond, d, pre, s):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    kpf = KalmanPlaxityF(cal, nq=64); Vf, Cf = mat(kpf.predict(val, D)), mat(kpf.predict(cal, D)); lr = LRAll(cal, D)
    S = {"KPF+stack": (stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc)), "LR-all": (mat(lr.predict(val)), lr.cal_oof()), "Toba-eq": (T, Tc)}
    rows = []
    for mth, (V, C) in S.items():
        th0 = static_theta(C[:, :2], yc, ALPHA); ub = ALPHA + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0))
        _, e0, _ = aci_alarms(V[:, :2], y, ALPHA, GAMMA, th0, LAG); fb, eb = bg_aci(V[:, :2], y, ALPHA, GAMMA, th0, LAG)
        for m, K in MK:
            fw, ew, fo = wh_aci(V[:, :2], y, ALPHA, GAMMA, th0, LAG, m=m, K=K)
            rows.append(dict(cond=cond, seed=s, method=mth, m=m, K=K, n=len(y), miss=y.mean(), miss_max=max_window(y, K), aci_max=max_window(e0, K),
                             bg_max=max_window(eb, K), wh_max=max_window(ew, K), bg_fpr=(fb >= 0)[y == 0].mean(), wh_fpr=(fw >= 0)[y == 0].mean(),
                             forced=fo.mean(), bg_unw=eb.mean(), wh_unw=ew.mean(), ub=ub))
    say(f"--- {cond} seed {s}: n_val={len(y)} D={D} miss={y.mean():.3f}")
    return rows


def main(mode):
    rows = []
    for run in (SMOKE if mode == "smoke" else RUNS):
        rows += run_one(*run)
    R = pd.DataFrame(rows); tag = "smoke" if mode == "smoke" else "confirm"; R.to_csv(f"results/m_{tag}_runs.csv", index=False)
    R["bound"] = R.m + LAG - 1; F = R[R.method == "KPF+stack"]; pd.set_option("display.width", 250)
    if mode != "smoke":
        pv = R.pivot_table(index=["cond", "seed", "m", "K"], columns="method", values="wh_fpr")
        chk = [("M1 KPF: worst window <= m (theorem), all runs x budgets", int((F.wh_max <= F.bound).sum()), 18, 18),
               ("M2 KPF: BG-ACI without the rule exceeds the budget", int((F.bg_max > F.bound).sum()), 12, 18),
               ("M3 KPF: FPR cost of the rule <= 0.03", int((F.wh_fpr - F.bg_fpr <= 0.03).sum()), 15, 18),
               ("M4 KPF: long-run one-sided bound still holds", int((F.wh_unw <= F.ub).sum()), 18, 18),
               ("M5 same guarantee, FPR: KPF < Toba-eq", int((pv["KPF+stack"] < pv["Toba-eq"]).sum()), 15, 18),
               ("M6 theorem holds for LR-all and Toba-eq too", int((R[R.method != "KPF+stack"].wh_max <= R[R.method != "KPF+stack"].bound).sum()), 36, 36)]
        say("\n===== Addendum M (Pi 5 runs; budgets (4,20) (6,50) (10,100), alpha 0.05) =====")
        for name, x, need, n in chk:
            say(f"{name:58s} {x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
    say("\nmeans:\n" + R.groupby(["method", "m", "K"])[["miss_max", "aci_max", "bg_max", "wh_max", "bg_fpr", "wh_fpr", "forced", "bg_unw", "wh_unw"]].mean().round(3).to_string())
    open(f"results/m_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "confirm")
