"""Addendum J: final method (KPF+stack with BG-ACI) vs standard baselines (baselines_j.py).
    python j_confirm.py smoke     -> pilot seeds 131-132 only (code check; numbers not used for any decision)
    python j_confirm.py           -> results/j_confirm_runs.csv, results/j_confirm_summary.txt
PRIMARY runs = the 6 Addendum I runs (never used to design any method). SECONDARY = the 13 earlier runs
(G13c, G14, H13, H14, EuRoC 181; KPF was designed on these, so they are reported but not used for the J checks).
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import auroc, ece, block_boot_diff
from g13_analysis import build, mat
from g_confirm import stack, since_switch
from h_confirm import bg_aci, w100
from kp_methods import KalmanPlaxity, aci_alarms, aci_bound_upper, static_theta
from kpt_methods import KalmanPlaxityT
from kpf_methods import KalmanPlaxityF
from baselines_j import Budget, LRAll, GBDTAll
from sweep_analysis import toba_scores, sweep_metrics, GAMMA, LAG

PRIMARY = [("EUROC", "data/I_vo", "vo", s) for s in (182, 183, 184)] + [("LIDAR-APPS", "data/I_g14", "g14", s) for s in (191, 192, 193)]
SECONDARY = [("G13c", "data/G13c", "g13", s) for s in (141, 142, 143)] + [("G14", "data/G14", "g14", s) for s in (151, 152, 153)] + \
            [("H13", "data/H13", "g13", s) for s in (161, 162, 163)] + [("H14", "data/H14", "g14", s) for s in (171, 172, 173)] + \
            [("EUROC181", "data/H15", "vo", 181)]
SMOKE = [("pilot", "data/G13pilot", "g13", s) for s in (131, 132)]
BASE = ["Budget", "LR-all", "GBDT-all"]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def run_one(cond, d, pre, s, primary):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    S, cost = {}, {}
    kpf = KalmanPlaxityF(cal, nq=64)
    t = time.perf_counter(); Vf = mat(kpf.predict(val, D)); cost["KPF+stack"] = (time.perf_counter() - t) / len(val) * 1e6
    Cf = mat(kpf.predict(cal, D))
    S["KPF+stack"] = (stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc))
    S["Toba-eq"] = (T, Tc)
    for name, cls in [("Budget", Budget), ("LR-all", LRAll), ("GBDT-all", GBDTAll)]:
        m = cls(cal, D)
        t = time.perf_counter(); V = mat(m.predict(val)); cost[name] = (time.perf_counter() - t) / len(val) * 1e6
        S[name] = (V, m.cal_oof())
        if name == "GBDT-all":
            say(f"    GBDT chosen: {m.chosen}")
    reg = val.regime.values; k3 = reg == "K3"; sw = since_switch(reg) < 15; b = np.arange(len(val)) // 50
    say(f"--- {cond} seed {s} ({'primary' if primary else 'secondary'}): n_val={len(val)} D={D} miss={y.mean():.3f}")
    rows = []
    ref = S["KPF+stack"][0][:, :2].max(1)
    for m, (V, C) in S.items():
        r = dict(cond=cond, seed=s, primary=primary, method=m, D=D, miss=y.mean(), us_per_frame_py=cost.get(m, np.nan),
                 **sweep_metrics(V, y, val, D), ece_rel_K3=ece(V[k3, 0], y[k3]), ece_act_K3=ece(V[k3, :2].max(1), y[k3]),
                 auc_act_switch=auroc(V[sw, :2].max(1), y[sw]), auc_act_K3=auroc(V[k3, :2].max(1), y[k3]))
        for al, tag in [(0.05, ""), (0.02, "_02")]:
            th0 = static_theta(C[:, :2], yc, al); ub = al + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0))
            f, e, _ = aci_alarms(V[:, :2], y, al, GAMMA, th0, LAG); fb, eb = bg_aci(V[:, :2], y, al, GAMMA, th0, LAG)
            r.update({f"aci_unw{tag}": e.mean(), f"aci_fpr{tag}": (f >= 0)[y == 0].mean(), f"ub{tag}": ub,
                      f"bg_unw{tag}": eb.mean(), f"bg_fpr{tag}": (fb >= 0)[y == 0].mean()})
            if al == 0.05:
                r.update(worst100=w100(e), bg_worst100=w100(eb))
        if m != "KPF+stack":
            dd = block_boot_diff(ref, V[:, :2].max(1), y, b, auroc); r.update(kpf_minus=dd[0], kpf_lo=dd[1], kpf_hi=dd[2])
        rows.append(r)
    return rows


def checks(R):
    P = R[R.primary]
    pv = lambda c: P.pivot_table(index=["cond", "seed"], columns="method", values=c)
    a, ek, fp = pv("auc_act"), pv("ece_act_K3"), pv("bg_fpr")
    lo = P[P.method != "KPF+stack"].pivot_table(index=["cond", "seed"], columns="method", values="kpf_lo")
    ok = lambda m: int(((P[P.method == m].bg_unw <= P[P.method == m].ub) & (P[P.method == m].bg_unw_02 <= P[P.method == m].ub_02)).sum())
    return [("J1 act AUROC KPF > Budget, CI > 0", int((lo["Budget"] > 0).sum()), 6, 6),
            ("J2 act AUROC KPF > LR-all", int((a["KPF+stack"] > a["LR-all"]).sum()), 5, 6),
            ("J3 act AUROC KPF >= GBDT-all - 0.01 (non-inferior)", int((a["KPF+stack"] >= a["GBDT-all"] - 0.01).sum()), 5, 6),
            ("J4 BG-ACI FPR (alpha .05): KPF < GBDT-all", int((fp["KPF+stack"] < fp["GBDT-all"]).sum()), 5, 6),
            ("J5 act ECE on unseen K3: KPF < GBDT-all", int((ek["KPF+stack"] < ek["GBDT-all"]).sum()), 5, 6),
            ("J6 BG-ACI guarantee holds for every baseline", sum(ok(m) for m in BASE), 18, 18)]


def main(mode):
    runs = [(r, False) for r in SMOKE] if mode == "smoke" else [(r, True) for r in PRIMARY] + [(r, False) for r in SECONDARY]
    rows = []
    for run, pr in runs:
        rows += run_one(*run, pr)
    R = pd.DataFrame(rows); tag = "smoke" if mode == "smoke" else "confirm"
    R.to_csv(f"results/j_{tag}_runs.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["auc_act", "auc_act_switch", "auc_act_K3", "ece_act_K3", "lead_ms_fpr10", "bg_fpr", "bg_worst100", "us_per_frame_py"]
    if mode != "smoke":
        say("\n===== Addendum J (primary: 6 Addendum I runs) =====")
        for name, x, need, n in checks(R):
            say(f"{name:55s} {x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
        say("\nprimary per run (auc_act):\n" + R[R.primary].pivot_table(index=["cond", "seed"], columns="method", values="auc_act").round(3).to_string())
        say("\nsecondary (13 earlier runs) wins of KPF+stack on auc_act vs each baseline:")
        Sx = R[~R.primary].pivot_table(index=["cond", "seed"], columns="method", values="auc_act")
        for m in BASE + ["Toba-eq"]:
            say(f"  vs {m:9s} {int((Sx['KPF+stack'] > Sx[m]).sum())}/{len(Sx)}  mean diff {(Sx['KPF+stack'] - Sx[m]).mean():+.3f}")
    say("\nmeans by primary/method:\n" + R.groupby(["primary", "method"])[cols].mean().round(3).to_string())
    open(f"results/j_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "confirm")
