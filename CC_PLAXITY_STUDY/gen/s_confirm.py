"""Addendum S confirmatory analysis (UNO Q): (a) replication of the Addendum P closed loop on 7 fresh seeds with the frozen P rules,
(b) fallback stress: every frame at level-2 strength under constant K3, (c) pooled 10-seed description (P + S), not a rule.
    python s_confirm.py                   -> data/S, seeds 261..267 (confirmatory), stress seeds 261 262 263
    python s_confirm.py <subdir> <seeds>  -> pilot (never used for claims)
"""
import sys, numpy as np, pandas as pd
import p_confirm as P
from scipy.stats import beta
DL = (27.0, 43.0)   # the UNO Q heavy and light deadlines of Addendum P (identical in every P seed)


def cp_upper(k, n): return 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))


def stress(sub, seeds):
    rows = []
    for s in seeds:
        a = pd.read_csv(f"data/{sub}/cl_s{s}_val_stressK3.csv"); a = a[a.regime != "K-1"].iloc[20:]
        r = dict(seed=s, n=len(a), not_done=int((a.done == 0).sum()), e5_p50=a.e5[a.done == 1].median(), e5_max=a.e5[a.done == 1].max())
        for D in DL:
            late = ((a.done == 0) | (a.e5 > D)).values; r[f"late_{D:g}"] = int(late.sum())
            r[f"win20_{D:g}"] = int(np.convolve(late.astype(float), np.ones(20), "valid").max())
        rows.append(r)
    return pd.DataFrame(rows)


def main(sub, seeds, sseeds, tag):
    P.OUT.clear(); P.main(sub, seeds, f"S_{tag}"); out = list(P.OUT)
    S = stress(sub, sseeds); k27, k43, n = int(S["late_27"].sum()), int(S["late_43"].sum()), int(S.n.sum())
    rule = k27 <= 0.005 * n
    out += [f"\n===== Addendum S stress (UNO Q, constant K3, every frame at level-2 strength; seeds {sseeds}) =====",
            f"S12 pooled late at 27 ms <= 0.5% of frames: {k27}/{n} ({k27 / n:.3%}), 95% upper {cp_upper(k27, n):.3%} -> {'PASS' if rule else 'FAIL'}",
            f"(descriptive) late at 43 ms: {k43}/{n}, 95% upper {cp_upper(k43, n):.3%}; worst 20-window at 27 ms {int(S['win20_27'].max())}",
            "per seed:\n" + S.round(3).to_string(index=False)]
    if tag == "confirm":   # pooled description, P and S together (not a rule)
        R = pd.concat([pd.read_csv("results/p_confirm_runs.csv"), pd.read_csv(f"results/p_S_{tag}_runs.csv")])
        d = R[R.policy == "kpmd"]; w = R.groupby(["load", "policy"]).maxwin_late.agg(["mean", "min", "max"]).round(2)
        out += [f"\n===== pooled 10 seeds (P 231-233 + S 261-267), descriptive =====",
                f"kpmd Theorem 4.6 held {int((d.maxwin_late_nl2 <= 4).sum())}/{len(d)}; plain form held {int((d.maxwin_late <= 4).sum())}/{len(d)}; "
                f"level-2 late {int(d.late_l2.sum())}/{int(d.n_l2.sum())} (95% upper {cp_upper(int(d.late_l2.sum()), int(d.n_l2.sum())):.3%})",
                "worst 20-window of all late frames by load and policy (mean, min, max over seeds):\n" + w.to_string(),
                "miss rate by load and policy (mean over seeds):\n" + R.groupby(["load", "policy"]).miss.mean().round(4).to_string()]
    print("\n".join(out[len(P.OUT):])); open(f"results/s_{tag}_summary.txt", "w").write("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sd = [int(x) for x in sys.argv[2:]]; main(sys.argv[1], sd, sd, "pilot")
    else:
        main("S", list(range(261, 268)), [261, 262, 263], "confirm")
