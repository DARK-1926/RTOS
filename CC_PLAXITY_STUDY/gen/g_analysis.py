"""G-study analysis (PROTOCOL_GENERALIZATION.md). Run from CC_PLAXITY_STUDY/gen:
    python g_analysis.py
Primary H1/H3 use cc_plaxity.run_unoq unchanged (one call per condition folder). This file adds:
  - GBM-deadline and NaiveScale baselines (H3-style ECE on unseen K3 at release, plus release AUROC),
  - secondary deadlines (calib p75 / p95),
  - per-K latency/miss tables.
The body mirrors run_unoq line by line so that D, the splits, warm-up and the models are identical.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(".."))
from cc_plaxity import prep, CCPlaxity, IndepPlaxity, CP_DONE, ece, auroc, block_boot_diff, run_unoq, feats
from cc_transfer_lr import LRdeadline
from sklearn.ensemble import HistGradientBoostingClassifier

CONDS = {"G1": "data/G1", "G2": "data/G2", "G3": "data/G3", "G4x": "data/G4x", "G8a": "data/G8a", "G8b": "data/G8b"}


class GBMdeadline(LRdeadline):
    """Same features and deadline set as LRdeadline; only the learner differs."""
    def __init__(self, calib, Ds):
        self.med = calib[["e2e", "n1", "n2"]].median()
        self.m = {}
        for k in CP_DONE:
            X = np.vstack([self._x(calib, k, D) for D in Ds])
            y = np.concatenate([(calib.e2e > D).values.astype(int) for D in Ds])
            self.m[k] = HistGradientBoostingClassifier(random_state=0).fit(X, y)


def naive_scale(cal, val, D):
    e = np.sort(cal.e2e.values); med = np.median(e)
    s = (val.prev_e2e / med).fillna(1.0).values
    return np.array([1.0 - np.searchsorted(e, D / si, side="right") / len(e) for si in s])


def load(fn):
    d = pd.read_csv(fn).iloc[20:]
    o = pd.DataFrame({"n1": d.dur_node1_ms, "n2": d.dur_node2_ms,
                      "F": np.maximum(d.dur_node3_ms, d.dur_node4_ms), "n5": d.dur_node5_ms,
                      "e2e": d.end_to_end_ms, "regime": d.regime, "split": d.split,
                      "order": d.global_frame_id, "cluster": d.bag_frame_idx}).reset_index(drop=True)
    o = prep(o)
    return o[o.split == "calib"].reset_index(drop=True), o[o.split == "val"].reset_index(drop=True)


def one(fn, q):
    cal, val = load(fn)
    D = float(np.round(np.quantile(cal.e2e, q)))
    y = (val.e2e > D).astype(int).values
    cc = CCPlaxity(cal, "K0"); ind = IndepPlaxity(cal)
    p_cc, p_ind = cc.predict(val, D, True), ind.predict(val, D)
    Ds = list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95])))
    p_lr = LRdeadline(cal, Ds).predict(val, D)
    p_gb = GBMdeadline(cal, Ds).predict(val, D)
    p_ns = naive_scale(cal, val, D)
    blocks = np.arange(len(val)) // 50
    u = (val.regime == "K3").values
    h1 = block_boot_diff(p_cc[0], p_ind[0], y, blocks, auroc)
    r = dict(file=os.path.basename(fn), q=q, D=D, miss_rate=y.mean(), K3_n=int(u.sum()), K3_miss=y[u].mean(),
             H1_auroc_diff=h1[0], H1_lo=h1[1], H1_pass=h1[1] > 0,
             auroc_cc=auroc(p_cc[0], y), auroc_lr=auroc(p_lr[0], y), auroc_gbm=auroc(p_gb[0], y),
             auroc_naive=auroc(p_ns, y))
    for n, p in (("cc", p_cc[0]), ("lr", p_lr[0]), ("gbm", p_gb[0]), ("naive", p_ns)):
        r[f"eceK3_{n}"] = ece(p[u], y[u])
    r["H3_pass"] = r["eceK3_cc"] < r["eceK3_lr"]
    r["G5_cc_beats_gbm"] = r["eceK3_cc"] < r["eceK3_gbm"]
    r["G6_cc_beats_naive_ece"] = r["eceK3_cc"] < r["eceK3_naive"]
    r["G6_cc_beats_naive_auroc"] = r["auroc_cc"] > r["auroc_naive"]
    return r


def main():
    os.makedirs("results", exist_ok=True)
    rows, lat = [], []
    for c, dd in CONDS.items():
        fs = sorted(glob.glob(f"{dd}/*.csv"))
        if not fs:
            continue
        os.makedirs(f"results/{c}", exist_ok=True)
        print(f"===== {c}: primary (frozen run_unoq)"); run_unoq(dd, f"results/{c}")
        for fn in fs:
            for q in (0.85, 0.75, 0.95):
                r = one(fn, q); r["cond"] = c; rows.append(r)
            d = pd.read_csv(fn).iloc[20:]
            Dp = float(np.round(d[d.split == "calib"].end_to_end_ms.quantile(0.85)))
            v = d[d.split == "val"]
            for k, g in v.groupby("regime"):
                lat.append(dict(cond=c, file=os.path.basename(fn), K=k, n=len(g), mean_e2e=g.end_to_end_ms.mean(),
                                mean_n1=g.dur_node1_ms.mean(), miss_rate=(g.end_to_end_ms > Dp).mean()))
    R = pd.DataFrame(rows); L = pd.DataFrame(lat)
    R.to_csv("results/g_all_runs.csv", index=False); L.to_csv("results/g_latency_by_K.csv", index=False)
    pd.set_option("display.width", 250)
    P = R[R.q == 0.85]
    print("\n===== per run (D = p85)")
    print(P[["cond", "file", "D", "K3_miss", "H1_auroc_diff", "H1_pass", "eceK3_cc", "eceK3_lr", "eceK3_gbm",
             "eceK3_naive", "auroc_cc", "auroc_naive", "H3_pass"]].round(3).to_string(index=False))
    print("\n===== pass counts (D = p85)")
    print(P.groupby("cond")[["H1_pass", "H3_pass", "G5_cc_beats_gbm", "G6_cc_beats_naive_ece",
                             "G6_cc_beats_naive_auroc"]].agg(["sum", "size"]).to_string())
    g = P[P.cond.isin(["G1", "G2", "G3"])]
    print(f"\nG5 (CC beats GBM on K3 ECE across G1-G3): {int(g.G5_cc_beats_gbm.sum())}/{len(g)}  (pass rule >= 8/11)")
    print("\n===== secondary deadlines (pass counts)")
    print(R[R.q != 0.85].groupby(["cond", "q"])[["H1_pass", "H3_pass"]].agg(["sum", "size"]).to_string())
    print("\n===== mean e2e / miss by K (D = p85)")
    print(L.groupby(["cond", "K"])[["mean_e2e", "mean_n1", "miss_rate"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
