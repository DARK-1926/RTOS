"""Data for the complete report (descriptive; recomputes the same scores as sweep_analysis / g11_analysis).
For every run (54 earlier + 3 G11):
  1. alarms at Toba's own job-count thresholds (th_min 0.8, th_max 1.0) for ALL methods incl. KP and EWMA-CP
     -> results/report_toba_thr.csv   (recall, FPR, precision, lead time of true alarms, share warned by N2 start)
  2. threshold sweep (one global threshold on P(miss), first checkpoint above it) at fixed FPR targets
     -> results/report_curves.csv     (FPR target, recall, precision, mean lead of true alarms, actionable recall)
Run from gen/:  python report_curves.py
"""
import os, glob
import numpy as np, pandas as pd
from sweep_analysis import runs, all_scores
from toba_eq_alarms import alarms_generic, alarms_toba, tdf
from toba_baseline import TobaPlaxityBaseline
from g_analysis import load
from toba_eq_alarms import attach

FPRS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def g11_runs():
    for s in [111, 112, 113]:
        fn = f"data/G11/combined/open_s{s}.csv"; cal, val = load(fn); raw = pd.read_csv(fn)
        cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
        val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
        yield "G11", f"g11_s{s}", cal, val, float(np.round(np.quantile(cal.e2e, 0.85))), "K0"


def times(val):
    return np.column_stack([val.e0, val.e1, val.e2, val.e3, val.e3 + val.n5])


def main():
    A, C = [], []
    def gen():
        yield from runs(); yield from g11_runs()
    for cond, run, cal, val, D, nom in gen():
        y = (val.e2e > D).values.astype(int); T = times(val)
        P, *_ = all_scores(cal, val, D, nom)
        tb = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.8, th_max=1.0).fit(tdf(cal), deadline_ms=D)
        for m, S in P.items():
            if m == "Toba-eq":
                f, lead = alarms_toba(tb, val, D); act = (f >= 1) & (f <= 2)
            else:
                f, lead = alarms_generic({k: S[:, k] for k in range(4)}, val, D, tb.thresholds); act = (f >= 0) & (f <= 1)
            a = f >= 0; tp = a & (y == 1)
            A.append(dict(cond=cond, run=run, method=m, D=D, miss_rate=y.mean(), recall=a[y == 1].mean(), fpr=a[y == 0].mean(),
                          precision=tp.sum() / max(a.sum(), 1), lead_ms=np.nanmean(lead[tp]) if tp.any() else np.nan,
                          actionable_recall=(act & (y == 1)).sum() / max(y.sum(), 1)))
            smax = S.max(1); neg = np.sort(smax[y == 0])
            for ft in FPRS:
                thr = neg[min(int(np.ceil((1 - ft) * len(neg))) - 1, len(neg) - 1)]
                al = S > thr; first = np.where(al.any(1), al.argmax(1), -1); a = first >= 0; tp = a & (y == 1)
                ld = D - T[np.arange(len(y)), np.maximum(first, 0)][:, ] if True else None
                C.append(dict(cond=cond, run=run, method=m, fpr_target=ft, fpr=a[y == 0].mean(), recall=a[y == 1].mean(),
                              precision=tp.sum() / max(a.sum(), 1), lead_ms=np.mean(ld[tp]) if tp.any() else np.nan,
                              actionable_recall=((first >= 0) & (first <= 1) & (y == 1)).sum() / max(y.sum(), 1)))
        print(cond, run, flush=True)
        pd.DataFrame(A).to_csv("results/report_toba_thr.csv", index=False)
        pd.DataFrame(C).to_csv("results/report_curves.csv", index=False)


if __name__ == "__main__":
    main()
