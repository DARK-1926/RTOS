"""Addendum O confirmatory analysis: closed loop v2 on the Pi 5.   python o_confirm.py [subdir seeds...]
Default: data/O seeds 221 222 223 -> results/o_confirm_runs.csv, results/o_confirm_summary.txt"""
import sys, os
import numpy as np, pandas as pd
from kp_methods import aci_bound_upper
POL = ["none", "kp", "kpm", "kpmwh", "toba", "random"]; LAG = 1
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def params(fn):
    h = [float(x) for x in open(fn).read().split("\n")[0].split()]
    return dict(D=h[4], theta0=h[5], alpha=h[6], gamma=h[7])


def main(sub, seeds, tag):
    rows = []
    for s in seeds:
        prm = params(f"data/{sub}/params_s{s}.txt"); D = prm["D"]
        for p in POL:
            a = pd.read_csv(f"data/{sub}/cl_s{s}_val_{p}.csv")
            ea = ((a.done == 1) & (a.e5 > D) & (a.alarm < 0)).values.astype(float)            # every released frame, in order
            w20 = int(np.convolve(ea, np.ones(20), "valid").max())
            r = a[(a.done == 1) & (a.regime != "K-1")].iloc[20:]
            miss = (r.e5 > D).values; unw = miss & (r.alarm < 0).values; k0 = (r.regime == "K0").values
            rows.append(dict(seed=s, policy=p, n=len(r), not_done=int((a.done == 0).sum()), D=D, miss=miss.mean(), miss_K3=miss[(r.regime == "K3").values].mean(),
                             degraded=(r.degraded > 0).mean(), degraded_K0=(r.degraded > 0)[k0].mean(), alarm=(r.alarm >= 0).mean(), alarm_K0=(r.alarm >= 0)[k0].mean(),
                             unwarned=unw.mean(), ub=prm["alpha"] + aci_bound_upper(len(r), prm["gamma"], LAG, max(prm["theta0"], 0)),
                             max_win20=w20, forced=r.forced.mean(), us_p50=r.us_kp.median(), us_p99=r.us_kp.quantile(0.99)))
    R = pd.DataFrame(rows); R.to_csv(f"results/o_{tag}_runs.csv", index=False)
    P = lambda c: R.pivot_table(index="seed", columns="policy", values=c)
    dg, al, ms, uw, ub, w, us = P("degraded"), P("alarm"), P("miss"), P("unwarned"), P("ub"), P("max_win20"), P("us_p99")
    n = len(seeds)
    chk = [("O1 latch-up: legacy kp degraded >= 1.5 x none would-alarm", int((dg["kp"] >= 1.5 * al["none"]).sum()), max(n - 1, 1), n),
           ("O2 fix: kpm degrades fewer frames than legacy kp", int((dg["kpm"] < dg["kp"]).sum()), n, n),
           ("O3 no latch: kpm degraded <= 1.5 x none would-alarm", int((dg["kpm"] <= 1.5 * al["none"]).sum()), n, n),
           ("O4 kpm degrades fewer frames than toba", int((dg["kpm"] < dg["toba"]).sum()), n, n),
           ("O5a delivered misses: kpm < none", int((ms["kpm"] < ms["none"]).sum()), n, n),
           ("O5b delivered misses: kpm <= random (same budget)", int((ms["kpm"] <= ms["random"]).sum()), max(n - 1, 1), n),
           ("O6 long-run guarantee, kpm and kpmwh", int((uw["kpm"] <= ub["kpm"]).sum() + (uw["kpmwh"] <= ub["kpmwh"]).sum()), 2 * n, 2 * n),
           ("O7 live (4,20) window guarantee under kpmwh", int((w["kpmwh"] <= 4).sum()), n, n),
           ("O8 inline cost p99 < 100 us under kpmwh", int((us["kpmwh"] < 100).sum()), n, n)]
    say(f"\n===== Addendum O ({sub}, seeds {seeds}) =====")
    for name, x, need, m in chk:
        say(f"{name:58s} {x}/{m} (rule {'>=' if need < m else ''}{need}/{m}) -> {'PASS' if x >= need else 'FAIL'}")
    pd.set_option("display.width", 250)
    say("\nper seed and policy:\n" + R.set_index(["seed", "policy"])[["n", "not_done", "miss", "miss_K3", "degraded", "degraded_K0", "alarm_K0", "unwarned", "ub", "max_win20", "forced", "us_p50", "us_p99"]].round(3).to_string())
    say("\nmeans:\n" + R.groupby("policy")[["miss", "miss_K3", "degraded", "degraded_K0", "unwarned", "max_win20", "forced", "us_p99"]].mean().round(3).loc[POL].to_string())
    open(f"results/o_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], [int(x) for x in sys.argv[2:]], "pilot")
    else:
        main("O", [221, 222, 223], "confirm")
