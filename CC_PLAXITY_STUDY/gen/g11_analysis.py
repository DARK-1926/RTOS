"""Addendum E confirmatory analysis: G11 (ROS 2 LiDAR pipeline under Autoware reference-system interference).
Run from gen/:   python g11_analysis.py      -> results/g11_sweep.csv, g11_aci.csv, g11_summary.txt

Conversion of the ROS 2 timestamps (ms after release t0) to the standard trace format (frozen in Addendum E):
    dur_node1 = s2                       (release -> N2 callback start: load/crop + publish + DDS + executor)
    fork      = min(s3, s4)
    dur_node2 = fork - s2                (voxel + publish + transport to the first fork branch)
    dur_node3 = e3 - fork,  dur_node4 = e4 - fork
    dur_node5 = e5 - max(e3, e4)         (transport of the later branch + fusion)
    end_to_end = e5                      (so n1 + n2 + F + n5 = e2e exactly; checkpoints are the true elapsed times)
Frames that never completed (done = 0) count as misses for every method (none expected).  Frames released
during a regime transition (K = -1) are removed together with the history link (has_prev recomputed).
Calibration = g11_s{seed}_calib.csv, validation = g11_s{seed}_val.csv, concatenated with val ids offset by 100000.
"""
import glob, os, io, contextlib
import numpy as np, pandas as pd
from g_analysis import load
from toba_eq_alarms import attach, evaluate
from sweep_analysis import all_scores, sweep_metrics, ALPHAS, GAMMA, LAG
from kp_methods import aci_alarms, aci_bound, static_theta
from cc_plaxity import block_boot_diff, auroc

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def convert(fn, offset):
    r = pd.read_csv(fn)
    fork = np.minimum(r.s3, r.s4)
    d = pd.DataFrame({"global_frame_id": r.frame + offset, "regime": r.regime, "split": r.split, "bag_frame_idx": r.scan,
                      "dur_node1_ms": r.s2, "dur_node2_ms": fork - r.s2, "dur_node3_ms": r.e3 - fork,
                      "dur_node4_ms": r.e4 - fork, "dur_node5_ms": r.e5 - np.maximum(r.e3, r.e4),
                      "end_to_end_ms": r.e5, "done": r.done})
    return d


def build(seed, root="data/G11"):
    c = convert(f"{root}/g11_s{seed}_calib.csv", 0); v = convert(f"{root}/g11_s{seed}_val.csv", 100000)
    d = pd.concat([c, v])
    nd = int((d.done == 0).sum()); ntr = int((d.regime == "K-1").sum())
    d = d[(d.done == 1) & (d.regime != "K-1")]
    fn = f"{root}/combined/open_s{seed}.csv"; os.makedirs(os.path.dirname(fn), exist_ok=True)
    d.drop(columns="done").to_csv(fn, index=False)
    return fn, nd, ntr


def main():
    rows, arows, h = [], [], []
    for f in sorted(glob.glob("data/G11/g11_s*_calib.csv")):
        s = int(os.path.basename(f).split("_")[1][1:])
        fn, nd, ntr = build(s)
        cal, val = load(fn); raw = pd.read_csv(fn)
        cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
        val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
        D = float(np.round(np.quantile(cal.e2e, 0.85)))
        y = (val.e2e > D).values.astype(int); ycal = (cal.e2e > D).values.astype(int)
        say(f"--- G11 seed {s}: calib {len(cal)} val {len(val)} frames, D = {D} ms, val miss {y.mean():.3f}, "
            f"K3 miss {y[val.regime.values == 'K3'].mean():.3f}, not-completed {nd}, transition frames dropped {ntr}")
        say("    per-K median e2e (val):", val.groupby("regime").e2e.median().round(2).to_dict())
        P, calP, cost, nanfrac = all_scores(cal, val, D)
        blocks = np.arange(len(val)) // 50
        for m, S in P.items():
            r = dict(seed=s, D=D, method=m, miss_rate=y.mean(), nan_frac=nanfrac[m], **sweep_metrics(S, y, val, D))
            if m != "Toba-eq":
                d = block_boot_diff(S[:, 0], P["Toba-eq"][:, 0], y, blocks, auroc)
                r.update(dAUCrel_vs_toba=d[0], lo_vs_toba=d[1], hi_vs_toba=d[2])
                d = block_boot_diff(S.max(1), P["Toba-eq"].max(1), y, blocks, auroc)
                r.update(dAUCany_vs_toba=d[0], lo_any_vs_toba=d[1])
            rows.append(r)
            if m in calP:
                for a, (tgt, SS, CC_) in [(a, t) for a in ALPHAS for t in
                                          (("actionable", S[:, :2], calP[m][:, :2]), ("any", S, calP[m]))]:
                    th0 = static_theta(CC_, ycal, a); st = SS.max(1) > th0
                    f_, e, _ = aci_alarms(SS, y, a, GAMMA, th0, LAG)
                    arows.append(dict(seed=s, method=m, target=tgt, alpha=a, T=len(y), static_unwarned=np.mean((y == 1) & ~st),
                                      static_fpr=st[y == 0].mean(), aci_unwarned=e.mean(), aci_fpr=(f_ >= 0)[y == 0].mean(),
                                      aci_bound=aci_bound(len(y), GAMMA, LAG)))
        al = evaluate(cal, val, D, "K0", f"G11_s{s}"); [a.update(seed=s) for a in al]; h += al
    R = pd.DataFrame(rows); A = pd.DataFrame(arows); H = pd.DataFrame(h)
    R.to_csv("results/g11_sweep.csv", index=False); A.to_csv("results/g11_aci.csv", index=False)
    H.to_csv("results/g11_alarms.csv", index=False)
    pd.set_option("display.width", 250)
    say("\n== threshold-free metrics ==\n" + R.round(3).to_string())
    say("\n== ACI ==\n" + A.round(4).to_string())
    say("\n== Toba-threshold alarms (job-count policy) ==\n" +
        H[H.policy == "var_job_0.8_1.0"][["seed", "method", "recall", "fpr", "precision", "lead_ms_tp"]].round(3).to_string())
    open("results/g11_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
