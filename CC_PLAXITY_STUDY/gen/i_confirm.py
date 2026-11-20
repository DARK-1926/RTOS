"""Addendum I confirmatory analysis (FROZEN before any Addendum I data).
    python i_confirm.py     -> results/i_confirm_runs.csv, results/i_confirm_summary.txt
Final method: KPF+stack (online-fused history/telemetry prior, kpf_methods.py) with burst-guarded ACI (h_confirm.bg_aci).
Runs (fresh seeds): EuRoC drone camera VO (vo, 182-184, 200 ms period) and lidar with real applications (g14, 191-193).
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import auroc, ece, block_boot_diff
from g13_analysis import build, mat
from g_confirm import stack, since_switch
from h_confirm import bg_aci, w100
from kp_methods import KalmanPlaxity, aci_alarms, aci_bound_upper, static_theta
from kpt_methods import KalmanPlaxityT
from kpf_methods import KalmanPlaxityF
from sweep_analysis import toba_scores, sweep_metrics, GAMMA, LAG

RUNS = [("EUROC", "data/I_vo", "vo", s) for s in (182, 183, 184)] + [("LIDAR-APPS", "data/I_g14", "g14", s) for s in (191, 192, 193)]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def run_one(cond, d, pre, s):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    M = {"KP": KalmanPlaxity(cal, nq=64), "KPT": KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")), "KPF": KalmanPlaxityF(cal, nq=64)}
    S = {m + "+stack": (stack(mat(k.predict(val, D)), mat(k.predict(cal, D)), T, Tc, yc), stack(mat(k.predict(cal, D)), mat(k.predict(cal, D)), Tc, Tc, yc))
         for m, k in M.items()}
    S["Toba-eq"] = (T, Tc)
    reg = val.regime.values; k3 = reg == "K3"; sw = since_switch(reg) < 15; b = np.arange(len(val)) // 50
    say(f"--- {cond} seed {s}: n_val={len(val)} D={D} miss={y.mean():.3f} K3 miss={y[k3].mean():.3f} KPF rho={M['KPF'].rho}")
    rows = []
    for m, (V, C) in S.items():
        r = dict(cond=cond, seed=s, method=m, D=D, miss=y.mean(), **sweep_metrics(V, y, val, D),
                 ece_rel_K3=ece(V[k3, 0], y[k3]), auc_rel_switch=auroc(V[sw, 0], y[sw]))
        for al, tag in [(0.05, ""), (0.02, "_02")]:
            th0 = static_theta(C[:, :2], yc, al); ub = al + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0))
            f, e, _ = aci_alarms(V[:, :2], y, al, GAMMA, th0, LAG); fb, eb = bg_aci(V[:, :2], y, al, GAMMA, th0, LAG)
            r.update({f"aci_unw{tag}": e.mean(), f"aci_fpr{tag}": (f >= 0)[y == 0].mean(), f"ub{tag}": ub,
                      f"bg_unw{tag}": eb.mean(), f"bg_fpr{tag}": (fb >= 0)[y == 0].mean()})
            if al == 0.05:
                r.update(worst100=w100(e), bg_worst100=w100(eb))
        if m != "Toba-eq":
            dd = block_boot_diff(V[:, :2].max(1), T[:, :2].max(1), y, b, auroc); r.update(lo_vs_toba=dd[1])
        rows.append(r)
    return rows


def main():
    rows = []
    for run in RUNS:
        rows += run_one(*run)
    R = pd.DataFrame(rows); R.to_csv("results/i_confirm_runs.csv", index=False)
    P = lambda c: R.pivot_table(index=["cond", "seed"], columns="method", values=c)
    a, ek, fp = P("auc_act"), P("ece_rel_K3"), P("bg_fpr")
    f = R[R.method == "KPF+stack"].set_index(["cond", "seed"]); dr = a.index.get_level_values(0) == "EUROC"
    best = np.maximum(a["KP+stack"], a["KPT+stack"])
    chk = [("I1 actionable AUROC KPF >= max(KP, KPT) - 0.01", int((a["KPF+stack"] >= best - 0.01).sum()), 5, 6),
           ("I2 actionable AUROC KPF > Toba-eq, CI > 0", int((f.lo_vs_toba > 0).sum()), 6, 6),
           ("I3 drone: actionable AUROC KPF > KPT", int((a["KPF+stack"] > a["KPT+stack"])[dr].sum()), 3, 3),
           ("I4 ECE on unseen K3: KPF < KP", int((ek["KPF+stack"] < ek["KP+stack"]).sum()), 5, 6),
           ("I5 BG-ACI one-sided guarantee, both alpha", int(((f.bg_unw <= f.ub) & (f.bg_unw_02 <= f.ub_02)).sum()), 6, 6),
           ("I6 worst100: BG-ACI < plain ACI", int((f.bg_worst100 < f.worst100).sum()), 5, 6),
           ("I7 BG-ACI FPR: KPF < Toba-eq", int((fp["KPF+stack"] < fp["Toba-eq"]).sum()), 6, 6)]
    say("\n===== Addendum I =====")
    for name, x, need, n in chk:
        say(f"{name:50s} {x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
    pd.set_option("display.width", 250)
    say("\n" + R.pivot_table(index=["cond", "seed"], columns="method", values=["auc_act", "ece_rel_K3", "bg_fpr"]).round(3).to_string())
    say("\nmeans:\n" + R.groupby(["cond", "method"])[["auc_rel", "auc_act", "auc_any", "auc_rel_switch", "ece_rel_K3", "recall_fpr10",
                                                    "lead_ms_fpr10", "bg_fpr", "bg_worst100", "worst100"]].mean().round(3).to_string())
    open("results/i_confirm_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
