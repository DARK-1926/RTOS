"""Addendum G confirmatory analysis (FROZEN before any G13c/G14 data).
    python g_confirm.py            -> results/g_confirm_runs.csv, results/g_confirm_summary.txt
Runs: data/G13c/g13_s{141,142,143}_{calib,val}.csv (Autoware reference-system interference)
      data/G14/g14_s{151,152,153}_{calib,val}.csv  (real applications: compiler, gzip; validation adds xz = unseen type)
Primary method  KPT  = KP with a telemetry-only prior (features u = other processes' CPU share, r = runnable tasks,
                       w = own preemptions), 64 quantiles, in-frame KP updates; late checkpoints (2, 3) stacked with Toba-eq
                       by a 2-feature logistic model fitted on calibration frames only ("+stack").
Comparators     KP (history prior) + stack, EWMA-CP, Toba-eq.
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from cc_plaxity import auroc, ece, block_boot_diff
from g13_analysis import build, mat
from kp_methods import KalmanPlaxity, aci_alarms, aci_bound_upper, static_theta
from kpt_methods import KalmanPlaxityT
from sweep_analysis import toba_scores, EWMACP, sweep_metrics, GAMMA, LAG

RUNS = [("G13c", "data/G13c", "g13", s) for s in (141, 142, 143)] + [("G14", "data/G14", "g14", s) for s in (151, 152, 153)]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def stack(V, C, T, Tc, yc):
    """checkpoints 0-1 from the method; checkpoints 2 and 3 = calibration-fitted logistic stack of (method, Toba-eq)."""
    out = [V[:, 0], V[:, 1]]
    for k, tv, tc in [(2, T[:, 2], Tc[:, 2]), (3, np.maximum(T[:, 3], T[:, 4]), np.maximum(Tc[:, 3], Tc[:, 4]))]:
        m = LogisticRegression(C=1.0).fit(np.column_stack([lg(C[:, k]), lg(tc)]), yc)
        out.append(m.predict_proba(np.column_stack([lg(V[:, k]), lg(tv)]))[:, 1])
    return np.column_stack(out)


def since_switch(reg):
    ch = np.r_[True, reg[1:] != reg[:-1]]; s = np.zeros(len(reg), int); c = 0
    for i in range(len(reg)):
        c = 0 if ch[i] else c + 1; s[i] = c
    return s


def main():
    rows = []
    for cond, d, pre, s in RUNS:
        cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
        raw = pd.read_csv(f"{d}/{pre}_s{s}_val.csv")
        D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
        T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
        kpt = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")); kp = KalmanPlaxity(cal, nq=64); ew = EWMACP(cal)
        raw_s = {"KPT": (mat(kpt.predict(val, D)), mat(kpt.predict(cal, D))), "KP": (mat(kp.predict(val, D)), mat(kp.predict(cal, D))),
                 "EWMA-CP": (mat(ew.predict(val, D)), mat(ew.predict(cal, D)))}
        S = {m + "+stack": (stack(V, C, T, Tc, yc), stack(C, C, Tc, Tc, yc)) for m, (V, C) in raw_s.items()}
        S["Toba-eq"] = (T, Tc)
        k3 = (val.regime == "K3").values; sw = since_switch(val.regime.values) < 15; b = np.arange(len(val)) // 50
        say(f"--- {cond} seed {s}: n_val={len(val)} D={D} miss={y.mean():.3f} K3 miss={y[k3].mean():.3f} "
            f"KPT prior R2={kpt.r2_prior:.3f}; critical-path telemetry us: median {raw.us_tm.median():.2f}, p99 {raw.us_tm.quantile(.99):.2f}")
        for m, (V, C) in S.items():
            r = dict(cond=cond, seed=s, method=m, D=D, miss=y.mean(), **sweep_metrics(V, y, val, D),
                     ece_rel_K3=ece(V[k3, 0], y[k3]), auc_rel_switch=auroc(V[sw, 0], y[sw]), auc_rel_steady=auroc(V[~sw, 0], y[~sw]),
                     us_tm_p99=raw.us_tm.quantile(0.99))
            th0 = static_theta(C[:, :2], yc, 0.05); f, e, _ = aci_alarms(V[:, :2], y, 0.05, GAMMA, th0, LAG)
            r.update(aci_unw=e.mean(), aci_fpr=(f >= 0)[y == 0].mean(), aci_ub=0.05 + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0)),
                     worst100=pd.Series(e).rolling(100).mean().max())
            th0 = static_theta(C[:, :2], yc, 0.02); f, e, _ = aci_alarms(V[:, :2], y, 0.02, GAMMA, th0, LAG)
            r.update(aci_unw_02=e.mean(), aci_fpr_02=(f >= 0)[y == 0].mean(), aci_ub_02=0.02 + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0)))
            if m != "Toba-eq":
                dd = block_boot_diff(V[:, :2].max(1), T[:, :2].max(1), y, b, auroc); r.update(dact_vs_toba=dd[0], lo_vs_toba=dd[1])
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv("results/g_confirm_runs.csv", index=False)
    W = lambda c: R.pivot_table(index=["cond", "seed"], columns="method", values=c)
    a, sw_, ek, ay, wo, fp, tm = W("auc_act"), W("auc_rel_switch"), W("ece_rel_K3"), W("auc_any"), W("worst100"), W("aci_fpr"), W("us_tm_p99")
    n = len(a)
    g1 = (ek["KPT+stack"] < ek["KP+stack"]).sum(); g2 = (sw_["KPT+stack"] > sw_["KP+stack"]).sum()
    g3 = (a["KPT+stack"] >= a["KP+stack"] - 0.005).sum()
    kp = R[R.method == "KPT+stack"]; g4 = (kp.lo_vs_toba > 0).sum()
    g5 = (ay["KPT+stack"] >= ay["Toba-eq"] - 0.005).sum()
    g6 = int(((kp.aci_unw <= kp.aci_ub) & (kp.aci_unw_02 <= kp.aci_ub_02)).sum())
    g7 = (fp["KPT+stack"] < fp["Toba-eq"]).sum(); g8 = (wo["KPT+stack"] < wo["KP+stack"]).sum(); g9 = (tm["KPT+stack"] < 5).sum()
    say(f"\nG1 ECE on unseen K3, KPT < KP:                 {g1}/{n} (rule >= 5/6) -> {'PASS' if g1 >= 5 else 'FAIL'}")
    say(f"G2 release AUROC first 15 frames after switch: {g2}/{n} (rule >= 5/6) -> {'PASS' if g2 >= 5 else 'FAIL'}")
    say(f"G3 actionable AUROC KPT >= KP - 0.005:         {g3}/{n} (rule >= 5/6) -> {'PASS' if g3 >= 5 else 'FAIL'}")
    say(f"G4 actionable AUROC KPT > Toba, CI > 0:        {g4}/{n} (rule 6/6)  -> {'PASS' if g4 == 6 else 'FAIL'}")
    say(f"G5 any-time AUROC KPT >= Toba - 0.005:         {g5}/{n} (rule >= 5/6) -> {'PASS' if g5 >= 5 else 'FAIL'}")
    say(f"G6 ACI one-sided guarantee (both alphas):      {g6}/{n} (rule 6/6)  -> {'PASS' if g6 == 6 else 'FAIL'}")
    say(f"G7 ACI price FPR KPT < Toba (alpha 0.05):      {g7}/{n} (rule 6/6)  -> {'PASS' if g7 == 6 else 'FAIL'}")
    say(f"G8 worst 100-frame unwarned rate KPT < KP:     {g8}/{n} (rule >= 4/6) -> {'PASS' if g8 >= 4 else 'FAIL'}")
    say(f"G9 critical-path telemetry p99 < 5 us:         {g9}/{n} (rule 6/6)  -> {'PASS' if g9 == 6 else 'FAIL'}")
    pd.set_option("display.width", 250)
    say("\n" + R.groupby(["cond", "method"])[["auc_rel", "auc_act", "auc_any", "auc_rel_switch", "ece_rel_K3", "recall_fpr10",
                                             "lead_ms_fpr10", "aci_fpr", "aci_fpr_02", "worst100"]].mean().round(3).to_string())
    open("results/g_confirm_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
