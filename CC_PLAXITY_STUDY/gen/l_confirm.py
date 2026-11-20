"""Addendum L: second platform (Raspberry Pi 5, Cortex-A76) replication of Addenda I and J with the FROZEN per-run code.
    python l_confirm.py      -> results/l_confirm_runs_I.csv, results/l_confirm_runs_J.csv, results/l_confirm_summary.txt
Runs: EuRoC drone VO (vo, seeds 201-203, 200 ms) and lidar with real applications (g14, seeds 211-213, 100 ms), recorded on the Pi 5.
Per-run metrics come from i_confirm.run_one and j_confirm.run_one unchanged; the check rules are copied verbatim from those files.
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
import i_confirm, j_confirm

RUNS = [("PI5-EUROC", "data/L_vo", "vo", s) for s in (201, 202, 203)] + [("PI5-LIDAR-APPS", "data/L_g14", "g14", s) for s in (211, 212, 213)]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def main():
    rI = []
    for run in RUNS:
        rI += i_confirm.run_one(*run)
    R = pd.DataFrame(rI); R.to_csv("results/l_confirm_runs_I.csv", index=False)
    P = lambda c: R.pivot_table(index=["cond", "seed"], columns="method", values=c)
    a, ek, fp = P("auc_act"), P("ece_rel_K3"), P("bg_fpr")
    f = R[R.method == "KPF+stack"].set_index(["cond", "seed"]); dr = a.index.get_level_values(0) == "PI5-EUROC"
    best = np.maximum(a["KP+stack"], a["KPT+stack"])
    chk = [("L-I1 actionable AUROC KPF >= max(KP, KPT) - 0.01", int((a["KPF+stack"] >= best - 0.01).sum()), 5, 6),
           ("L-I2 actionable AUROC KPF > Toba-eq, CI > 0", int((f.lo_vs_toba > 0).sum()), 6, 6),
           ("L-I3 drone: actionable AUROC KPF > KPT", int((a["KPF+stack"] > a["KPT+stack"])[dr].sum()), 3, 3),
           ("L-I4 ECE on unseen K3: KPF < KP", int((ek["KPF+stack"] < ek["KP+stack"]).sum()), 5, 6),
           ("L-I5 BG-ACI one-sided guarantee, both alpha", int(((f.bg_unw <= f.ub) & (f.bg_unw_02 <= f.ub_02)).sum()), 6, 6),
           ("L-I6 worst100: BG-ACI < plain ACI", int((f.bg_worst100 < f.worst100).sum()), 5, 6),
           ("L-I7 BG-ACI FPR: KPF < Toba-eq", int((fp["KPF+stack"] < fp["Toba-eq"]).sum()), 6, 6)]
    rJ = []
    for run in RUNS:
        rJ += j_confirm.run_one(*run, True)
    RJ = pd.DataFrame(rJ); RJ.to_csv("results/l_confirm_runs_J.csv", index=False)
    chk += [("L-" + n, x, need, m) for n, x, need, m in j_confirm.checks(RJ)]
    say("\n===== Addendum L (Raspberry Pi 5) =====")
    for name, x, need, n in chk:
        say(f"{name:58s} {x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
    pd.set_option("display.width", 250)
    say("\n" + R.pivot_table(index=["cond", "seed"], columns="method", values=["auc_act", "ece_rel_K3", "bg_fpr"]).round(3).to_string())
    cols = ["auc_act", "auc_act_switch", "ece_act_K3", "lead_ms_fpr10", "bg_fpr", "bg_worst100", "us_per_frame_py"]
    say("\nbaselines, means by cond/method:\n" + RJ.groupby(["cond", "method"])[cols].mean().round(3).to_string())
    open("results/l_confirm_summary.txt", "w").write("\n".join(i_confirm.OUT + j_confirm.OUT + OUT))


if __name__ == "__main__":
    main()
