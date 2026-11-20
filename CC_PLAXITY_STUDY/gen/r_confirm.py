"""Addendum R confirmatory analysis: fallback (level-2 strength) stress on the Pi 5 under constant K3, two background-load arms:
base (interferers SCHED_OTHER, as Addendum Q) and idle (interferers SCHED_IDLE, unprivileged). Same frame filter as q_confirm.stress().
    python r_confirm.py                  -> data/R, seeds 251 252 253 (confirmatory)
    python r_confirm.py <subdir> <seeds> -> pilot (never used for claims)
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import beta
OUT = []
DL = (6.0, 10.0)        # the two Pi 5 deadlines of Addendum Q


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def cp_upper(k, n, conf=0.95):
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def arm(sub, s, a_):
    a = pd.read_csv(f"data/{sub}/cl_s{s}_val_stress_{a_}.csv"); a = a[a.regime != "K-1"].iloc[20:]
    r = dict(seed=s, arm=a_, n=len(a), not_done=int((a.done == 0).sum()), e5_p50=a.e5[a.done == 1].median(),
             e5_p99=a.e5[a.done == 1].quantile(0.99), e5_max=a.e5[a.done == 1].max())
    for D in DL:
        late = ((a.done == 0) | (a.e5 > D)).values
        r[f"late_{D:g}"] = int(late.sum()); r[f"frac_{D:g}"] = late.mean(); r[f"cp95_{D:g}"] = cp_upper(int(late.sum()), len(a))
        r[f"win20_{D:g}"] = int(np.convolve(late.astype(float), np.ones(20), "valid").max())
    return r


def main(sub, seeds, tag):
    R = pd.DataFrame([arm(sub, s, a) for s in seeds for a in ("base", "idle")]); R.to_csv(f"results/r_{tag}_runs.csv", index=False)
    n = len(seeds); B, I = R[R.arm == "base"].set_index("seed"), R[R.arm == "idle"].set_index("seed")
    pool = lambda X, D: (int(X[f"late_{D:g}"].sum()), int(X.n.sum()))
    ib6, ib10 = pool(I, 6), pool(I, 10); bb10 = pool(B, 10)
    chk = [("R1 replication, base: level 2 late at 6 ms > 0.5%, per seed", int((B.frac_6 > 0.005).sum()), n - 1, n),
           ("R2 idle priority for the background load reduces late level-2 frames at 6 ms, per seed", int((I.frac_6 < B.frac_6).sum()), n, n),
           ("R3 prediction: even so, A fails at 6 ms (idle arm, pooled late > 0.5%)", int(ib6[0] > 0.005 * ib6[1]), 1, 1),
           ("R4 idle arm, 10 ms: pooled late <= 0.15% (the Addendum Q base rate)", int(ib10[0] <= 0.0015 * ib10[1]), 1, 1)]
    say(f"\n===== Addendum R ({sub}, seeds {seeds}) =====")
    for name, x, need, m in chk:
        say(f"{name:92s} {x}/{m} (rule {'>=' if need < m else ''}{need}/{m}) -> {'PASS' if x >= need else 'FAIL'}")
    for lab, X in (("base", B), ("idle", I)):
        for D in DL:
            k, nn = pool(X, D)
            say(f"{lab} pooled, {D:g} ms: {k} late of {nn} ({k / max(nn, 1):.3%}); 95% upper bound {cp_upper(k, nn):.3%}; "
                f"worst late per 20-window {int(X[f'win20_{D:g}'].max())}")
    pd.set_option("display.width", 250)
    say("\nper seed and arm:\n" + R.round(4).to_string(index=False))
    open(f"results/r_{tag}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], [int(x) for x in sys.argv[2:]], "pilot")
    else:
        main("R", [251, 252, 253], "confirm")
