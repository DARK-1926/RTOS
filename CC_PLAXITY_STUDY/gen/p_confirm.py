"""Addendum P confirmatory analysis: closed loop v3 on the Arduino UNO Q (lidar_ros_cl3), two loads.
    python p_confirm.py                      -> data/P, seeds 231 232 233   (confirmatory)
    python p_confirm.py <subdir> <seeds...>  -> pilot / smoke (never used for claims)
Per-run definitions follow o_confirm.py (Addendum O) exactly; new quantities:
  late         every released frame in release order: late = not finished, or finished after D (conservative)
  l2           frame ran at level 2 (degraded == 3, policy kpmd only)
  maxwin_late  worst 20-frame window of late frames;  maxwin_late_nl2  same, counting only late frames NOT at level 2
"""
import sys
import numpy as np, pandas as pd
from kp_methods import aci_bound_upper
from o_confirm import params
POL = ["none", "kp", "kpmn", "kpm", "kpmwh", "kpmd", "toba", "random"]; LOADS = ["heavy", "light"]; LAG = 1; M, K = 4, 20
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def win(x, k=K):
    x = np.asarray(x, float); return int(np.convolve(x, np.ones(k), "valid").max()) if len(x) >= k else int(x.sum())


def run(sub, s, L, p):
    prm = params(f"data/{sub}/params_s{s}_{L}.txt"); D = prm["D"]
    a = pd.read_csv(f"data/{sub}/cl_s{s}_val_{L}_{p}.csv")
    late_all = ((a.done == 0) | (a.e5 > D)).values; l2_all = (a.degraded == 3).values
    ea = ((a.done == 1) & (a.e5 > D) & (a.alarm < 0)).values
    r = a[(a.done == 1) & (a.regime != "K-1")].iloc[20:]
    miss = (r.e5 > D).values; unw = miss & (r.alarm < 0).values; k0 = (r.regime == "K0").values; l2 = (r.degraded == 3).values
    return dict(seed=s, load=L, policy=p, n=len(r), not_done=int((a.done == 0).sum()), D=D, miss=miss.mean(), miss_K3=miss[(r.regime == "K3").values].mean(),
                degraded=(r.degraded > 0).mean(), degraded_K0=(r.degraded > 0)[k0].mean(), alarm=(r.alarm >= 0).mean(), alarm_K0=(r.alarm >= 0)[k0].mean(),
                unwarned=unw.mean(), ub=prm["alpha"] + aci_bound_upper(len(r), prm["gamma"], LAG, max(prm["theta0"], 0)),
                maxwin_unw=win(ea), maxwin_late=win(late_all), maxwin_late_nl2=win(late_all & ~l2_all),
                n_l2=int(l2_all.sum()), late_l2=int((late_all & l2_all).sum()), l2_frac=l2.mean(),
                miss_released_degraded=miss[(r.degraded == 1).values].mean() if (r.degraded == 1).any() else np.nan,
                us_p50=r.us_kp.median(), us_p99=r.us_kp.quantile(0.99))


def main(sub, seeds, tag):
    R = pd.DataFrame([run(sub, s, L, p) for s in seeds for L in LOADS for p in POL]); R.to_csv(f"results/p_{tag}_runs.csv", index=False)
    n = len(seeds); I = lambda L, c: R[R.load == L].pivot_table(index="seed", columns="policy", values=c)
    H, Lt = (lambda c: I("heavy", c)), (lambda c: I("light", c))
    d = R[R.policy == "kpmd"]
    p1 = int(((d.late_l2 <= 0.005 * d.n_l2)).sum())
    th, tk = H("degraded")["toba"], H("degraded")["kpm"]; rt, rk = th.max() - th.min(), tk.max() - tk.min()
    g = R[R.policy.isin(["kpm", "kpmwh", "kpmd"])]
    lt = R.pivot_table(index=["seed", "load"], columns="policy", values="maxwin_late")
    tot = R.assign(nl=R.miss * R.n).pivot_table(index=["seed", "load"], columns="policy", values="nl")
    chk = [("P1 level-2 frames finish on time (late <= 0.5%), kpmd", p1, 2 * n, 2 * n),
           ("P1b at least 30 level-2 frames in total", int(d.n_l2.sum() >= 30), 1, 1),
           ("P2 theorem: worst 20-window of late non-level-2 frames <= 4", int((d.maxwin_late_nl2 <= M).sum()), 2 * n, 2 * n),
           ("P3 worst 20-window of ALL late frames <= 4, kpmd", int((d.maxwin_late <= M).sum()), 2 * n - 1, 2 * n),
           ("P4 isolation, heavy: kpmn degrades fewer K0 frames than kp", int((H("degraded_K0")["kpmn"] < H("degraded_K0")["kp"]).sum()), n, n),
           ("P5 kpmn does not latch, heavy: K0 degraded <= 1.5 x none would-alarm", int((H("degraded_K0")["kpmn"] <= 1.5 * H("alarm_K0")["none"]).sum()), n, n),
           ("P6 light: kpm degrades <= 25% of frames", int((Lt("degraded")["kpm"] <= 0.25).sum()), n, n),
           ("P7a light: kpm misses < none", int((Lt("miss")["kpm"] < Lt("miss")["none"]).sum()), n, n),
           ("P7b light: kpm misses <= random (same budget)", int((Lt("miss")["kpm"] <= Lt("miss")["random"]).sum()), n - 1, n),
           ("P8 heavy: toba degraded range >= 0.4 and kpm range < half of it", int(rt >= 0.4 and rk < 0.5 * rt), 1, 1),
           ("P9 long-run guarantee, kpm kpmwh kpmd, both loads", int((g.unwarned <= g.ub).sum()), 6 * n, 6 * n),
           ("P10 inline cost p99 < 200 us, kpmd", int((d.us_p99 < 200).sum()), 2 * n, 2 * n),
           ("P11 total late frames: kpmd <= kpmwh", int((tot["kpmd"] <= tot["kpmwh"]).sum()), 2 * n - 2, 2 * n)]
    say(f"\n===== Addendum P ({sub}, seeds {seeds}) =====")
    for name, x, need, m in chk:
        say(f"{name:68s} {x}/{m} (rule {'>=' if need < m else ''}{need}/{m}) -> {'PASS' if x >= need else 'FAIL'}")
    say(f"toba degraded range (heavy) {rt:.3f}; kpm range {rk:.3f}")
    pd.set_option("display.width", 260)
    cols = ["n", "not_done", "D", "miss", "miss_K3", "degraded", "degraded_K0", "alarm_K0", "unwarned", "ub", "maxwin_unw", "maxwin_late",
            "maxwin_late_nl2", "n_l2", "late_l2", "miss_released_degraded", "us_p99"]
    say("\nper seed, load and policy:\n" + R.set_index(["seed", "load", "policy"])[cols].round(3).to_string())
    for L in LOADS:
        say(f"\nmeans, {L}:\n" + R[R.load == L].groupby("policy")[cols[3:]].mean().round(3).loc[POL].to_string())
    open(f"results/p_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], [int(x) for x in sys.argv[2:]], "pilot")
    else:
        main("P", [231, 232, 233], "confirm")
