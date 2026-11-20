"""Addendum N: trace-driven control case study on the Pi 5 runs (frozen before any Pi confirmatory data was pulled).
    python n_confirm.py   -> results/n_confirm_runs.csv, results/n_confirm_sweep.csv, results/n_confirm_summary.txt
Primary setting (fixed a priori, before the sensitivity sweep): w0 = 3 rad/s, fallback noise ratio 10, alpha 0.05, 20 noise seeds."""
import sys, os; sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from g13_analysis import build, mat
from g_confirm import stack
from h_confirm import bg_aci
from kp_methods import static_theta
from kpf_methods import KalmanPlaxityF
from baselines_j import LRAll
from sweep_analysis import toba_scores, GAMMA, LAG
from wh_methods import wh_aci
from control_study import simulate, modes
from l_confirm import RUNS
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def policies(d, pre, s):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    kpf = KalmanPlaxityF(cal, nq=64); Vf, Cf = mat(kpf.predict(val, D)), mat(kpf.predict(cal, D)); lr = LRAll(cal, D)
    S = {"KPF": (stack(Vf, Cf, T, Tc, yc), stack(Cf, Cf, Tc, Tc, yc)), "LR": (mat(lr.predict(val)), lr.cal_oof()), "Toba": (T, Tc)}
    pol = {"none": modes(y, kind="none"), "always-fallback": modes(y, kind="always"), "oracle": modes(y, kind="oracle")}
    for m_, (V, C) in S.items():
        th0 = static_theta(C[:, :2], yc, 0.05)
        pol[m_ + "+BG"] = modes(y, bg_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG)[0])
        for mm, K in [(4, 20), (2, 20)]:
            pol[f"{m_}+WH({mm},{K})"] = modes(y, wh_aci(V[:, :2], y, 0.05, GAMMA, th0, LAG, m=mm, K=K)[0])
    return pol, (0.2 if pre == "vo" else 0.1), y


def main():
    rows, sw = [], []
    for cond, d, pre, s in RUNS:
        pol, h, y = policies(d, pre, s)
        for p, md in pol.items():
            rows.append(dict(cond=cond, seed=s, h=h, policy=p, miss=y.mean(), fallback=(md == 1).mean(), hold=(md == 2).mean(), **simulate(md, h)))
        for w0 in (2.0, 3.0, 4.0, 5.0):
            for ratio in (3.0, 10.0, 30.0):
                for p in ["none", "always-fallback", "oracle", "Toba+BG", "KPF+BG", "KPF+WH(4,20)", "KPF+WH(2,20)"]:
                    sw.append(dict(cond=cond, seed=s, h=h, w0=w0, ratio=ratio, policy=p, **simulate(pol[p], h, w0=w0, ratio=ratio, nseed=5)))
        say(f"--- {cond} seed {s}: miss={y.mean():.3f}")
    R = pd.DataFrame(rows); R.to_csv("results/n_confirm_runs.csv", index=False); S = pd.DataFrame(sw); S.to_csv("results/n_confirm_sweep.csv", index=False)
    J = R.pivot_table(index=["cond", "seed"], columns="policy", values="J"); C = R.pivot_table(index=["cond", "seed"], columns="policy", values="crash_per_1000")
    chk = [("N1 control cost: KPF+BG < no monitor", int((J["KPF+BG"] < J["none"]).sum()), 6, 6),
           ("N2 control cost: KPF+BG < Toba+BG", int((J["KPF+BG"] < J["Toba+BG"]).sum()), 5, 6),
           ("N3 control cost: KPF+BG < always-fallback", int((J["KPF+BG"] < J["always-fallback"]).sum()), 5, 6),
           ("N4 crashes: no monitor > KPF+BG", int((C["none"] > C["KPF+BG"]).sum()), 5, 6),
           ("N5 window guarantee (4,20) costs <= 5% control cost", int((J["KPF+WH(4,20)"] <= 1.05 * J["KPF+BG"]).sum()), 5, 6)]
    say("\n===== Addendum N (Pi 5 runs; w0 = 3, ratio 10) =====")
    for name, x, need, n in chk:
        say(f"{name:55s} {x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}")
    pd.set_option("display.width", 250)
    say("\nprimary setting, means:\n" + R.groupby(["h", "policy"])[["fallback", "hold", "J", "crash_per_1000", "p99_angle"]].mean().round(4).to_string())
    for c in ["J", "crash_per_1000"]:
        say(f"\nsensitivity sweep, {c} (all settings):\n" + S.pivot_table(index=["h", "w0", "ratio"], columns="policy", values=c).round(3).to_string())
    open("results/n_confirm_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
