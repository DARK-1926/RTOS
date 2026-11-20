"""Addendum Q confirmatory analysis: closed loop v3 (lidar_ros_cl3) on the Raspberry Pi 5, two loads, plus fallback stress runs.
    python q_confirm.py                      -> data/Q, seeds 241 242 243   (confirmatory)
    python q_confirm.py <subdir> <seeds...>  -> pilot / smoke (never used for claims)
Per-run definitions are p_confirm.run() (Addendum P) unchanged; stress runs: every frame at level-2 strength (random p = 1,
CC_FB_DIV = 8) under constant contention K3 (and K4, exploratory); late = not finished or finished after the seed's deadline.
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import beta
import p_confirm
from p_confirm import run, POL, LOADS, M
from o_confirm import params
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def stress(sub, s, tag):
    a = pd.read_csv(f"data/{sub}/cl_s{s}_val_{tag}.csv"); a = a[a.regime != "K-1"].iloc[20:]
    r = dict(seed=s, tag=tag, n=len(a), not_done=int((a.done == 0).sum()), degraded_frac=(a.degraded > 0).mean(),
             e5_max=a.e5[a.done == 1].max(), e5_p999=a.e5[a.done == 1].quantile(0.999), e5_p50=a.e5[a.done == 1].median())
    for L in LOADS:
        D = params(f"data/{sub}/params_s{s}_{L}.txt")["D"]; late = int(((a.done == 0) | (a.e5 > D)).sum())
        r[f"D_{L}"] = D; r[f"late_{L}"] = late; r[f"late_frac_{L}"] = late / len(a); r[f"cp95_{L}"] = cp_upper(late, len(a))
    return r


def main(sub, seeds, tag):
    R = pd.DataFrame([run(sub, s, L, p) for s in seeds for L in LOADS for p in POL]); R.to_csv(f"results/q_{tag}_runs.csv", index=False)
    S = pd.DataFrame([stress(sub, s, t) for s in seeds for t in ("stressK3", "stressK4")]); S.to_csv(f"results/q_{tag}_stress.csv", index=False)
    n = len(seeds); I = lambda L, c: R[R.load == L].pivot_table(index="seed", columns="policy", values=c)
    H, Lt = (lambda c: I("heavy", c)), (lambda c: I("light", c))
    d = R[R.policy == "kpmd"]; dl, dh = d[d.load == "light"], d[d.load == "heavy"]
    g = R[R.policy.isin(["kpm", "kpmwh", "kpmd"])]
    wl = R.pivot_table(index=["seed", "load"], columns="policy", values="maxwin_late")
    s3 = S[S.tag == "stressK3"]
    k3n, k3l = int(s3.n.sum()), int(s3.late_light.sum())
    s3p = s3.late_light.sum() / max(k3n, 1)
    chk = [("Q1 heavy: level-2 frames of kpmd, pooled over seeds, late <= 2%", int(dh.late_l2.sum() <= 0.02 * dh.n_l2.sum()), 1, 1),
           ("Q1b at least 100 level-2 frames in total, heavy kpmd", int(dh.n_l2.sum() >= 100), 1, 1),
           ("Q2 theorem, exact form: worst 20-window of late non-level-2 frames <= 4, kpmd, both loads", int((d.maxwin_late_nl2 <= M).sum()), 2 * n, 2 * n),
           ("Q3 heavy: worst 20-window of ALL late frames <= 4, kpmd", int((dh.maxwin_late <= M).sum()), n - 1, n),
           ("Q3L light: worst 20-window of ALL late frames <= 4, kpmd", int((dl.maxwin_late <= M).sum()), n, n),
           ("Q4 prediction, stress K3: level-2 strength NOT on time at the heavy deadline (late > 0.5%)", int((s3.late_frac_heavy > 0.005).sum()), n - 1, n),
           ("Q5 heavy: worst window of all late frames, kpmd <= kpmwh", int((wl.xs("heavy", level="load")["kpmd"] <= wl.xs("heavy", level="load")["kpmwh"]).sum()), n - 1, n),
           ("Q6 latch-up isolated, heavy: kpmn degrades fewer K0 frames than kp", int((H("degraded_K0")["kpmn"] < H("degraded_K0")["kp"]).sum()), n, n),
           ("Q7 no latch, heavy: kpmn K0 degraded <= max(1.5 x none would-alarm, 0.02)", int((H("degraded_K0")["kpmn"] <= np.maximum(1.5 * H("alarm_K0")["none"], 0.02)).sum()), n, n),
           ("Q8 long-run guarantee, kpm kpmwh kpmd, both loads", int((g.unwarned <= g.ub).sum()), 6 * n, 6 * n),
           ("Q9a light: kpm degrades <= 25% of frames", int((Lt("degraded")["kpm"] <= 0.25).sum()), n, n),
           ("Q9b light: kpm misses <= random (same budget)", int((Lt("miss")["kpm"] <= Lt("miss")["random"]).sum()), n - 1, n),
           ("Q10 inline cost p99 < 100 us, kpmd", int((d.us_p99 < 100).sum()), 2 * n, 2 * n),
           ("Q11 stress K3, pooled: level-2 strength late at the light deadline <= 1.5%", int(s3p <= 0.015), 1, 1)]
    say(f"\n===== Addendum Q ({sub}, seeds {seeds}) =====")
    for name, x, need, m in chk:
        say(f"{name:86s} {x}/{m} (rule {'>=' if need < m else ''}{need}/{m}) -> {'PASS' if x >= need else 'FAIL'}")
    say(f"stress K3 pooled: {k3l} late of {k3n} at the light deadline; 95% upper bound on the late rate {cp_upper(k3l, k3n):.4%}")
    say(f"stress K3 pooled: {int(s3.late_heavy.sum())} late of {k3n} at the heavy deadline ({s3.late_heavy.sum() / max(k3n, 1):.2%})")
    pd.set_option("display.width", 260)
    cols = ["n", "not_done", "D", "miss", "degraded", "degraded_K0", "alarm_K0", "unwarned", "ub", "maxwin_unw", "maxwin_late",
            "maxwin_late_nl2", "n_l2", "late_l2", "miss_released_degraded", "us_p99"]
    say("\nper seed, load and policy:\n" + R.set_index(["seed", "load", "policy"])[cols].round(3).to_string())
    for L in LOADS:
        say(f"\nmeans, {L}:\n" + R[R.load == L].groupby("policy")[cols[3:]].mean().round(3).loc[POL].to_string())
    say("\nstress runs (every frame at level-2 strength, constant contention; K4 is exploratory):\n" + S.round(4).to_string(index=False))
    open(f"results/q_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], [int(x) for x in sys.argv[2:]], "pilot")
    else:
        main("Q", [241, 242, 243], "confirm")
