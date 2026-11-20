"""Reviewer-hardening analysis (Addendum D). Run from CC_PLAXITY_STUDY/gen:
    python ext_analysis.py existing     -> results/ext_existing.csv  (EXPLORATORY: all 36 earlier hardware runs)
    python ext_analysis.py <cond> <dir> -> results/ext_<cond>.csv    (used for the fresh confirmatory runs)
Every run is evaluated twice: 'standard' (calib/val as recorded) and, for real-LiDAR runs, 'scene_disjoint'
(3 folds over the 10 nuScenes scenes: models are fit on calib frames of the other scenes only, and scored on val
frames of the held-out scenes; history/forward filtering still runs over the full val sequence, because the
previous frame is test-time information, not training data).
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(".."))
from cc_plaxity import CCPlaxity, IndepPlaxity, ece, auroc, block_boot_diff
from cc_transfer_lr import LRdeadline
from g_analysis import GBMdeadline, load
from ext_methods import AR1, EWMA, LRIso, NaiveCP, cc_sc

SCENES = pd.read_csv("scan_scene_map.csv")
SC_OF = dict(zip(SCENES.scan_idx, SCENES.scene))
ORDER = list(dict.fromkeys(SCENES.scene))
FOLDS = [[ORDER[i] for i in (0, 3, 6, 9)], [ORDER[i] for i in (1, 4, 7)], [ORDER[i] for i in (2, 5, 8)]]

EXISTING = {"ArmA": "../board_armA_schedutil", "ArmB": "../board_armB_spin", "ArmC": "../board_armC_perfgov",
            "G1": "data/G1", "G2": "data/G2", "G3": "data/G3", "G4x": "data/G4x", "G8a": "data/G8a", "G8b": "data/G8b"}
LIDAR = {"G1", "G4x", "G8a", "G8b"}
METHODS = ["cc", "ccsc", "indep", "lr", "lriso", "gbm", "naive", "ar1", "ewma"]


def relink(df):
    """Recompute chunk/has_prev after row filtering (so the HMM never links non-consecutive frames)."""
    df = df.reset_index(drop=True).copy()
    brk = (df.order.diff() != 1)
    df["has_prev"] = ~brk; df["chunk"] = brk.cumsum()
    return df


def fit_predict(cal, val, D):
    """Release-time (cp0) probabilities on val for every method, fit on cal."""
    Ds = list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95])))
    cc = CCPlaxity(cal, "K0"); nv = NaiveCP(cal)
    p_sc, flag = cc_sc(cc, nv, val, D)
    P = {"cc": cc.predict(val, D, True)[0], "ccsc": p_sc[0], "indep": IndepPlaxity(cal).predict(val, D)[0],
         "lr": LRdeadline(cal, Ds).predict(val, D)[0], "lriso": LRIso(cal, Ds, D).predict(val)[0],
         "gbm": GBMdeadline(cal, Ds).predict(val, D)[0], "naive": nv.predict(val, D)[0],
         "ar1": AR1(cal).predict(val, D), "ewma": EWMA(cal).predict(val, D)}
    return P, flag, dict(rho=cc.rho)


def score(fn, cond, mode, q=0.85):
    cal, val = load(fn)
    D = float(np.round(np.quantile(cal.e2e, q)))
    if mode == "standard":
        P, flag, info = fit_predict(cal, val, D)
        keep = np.ones(len(val), bool)
    else:
        sc_cal = cal.cluster.map(SC_OF).values; sc_val = val.cluster.map(SC_OF).values
        P = {m: np.full(len(val), np.nan) for m in METHODS}; flag = np.zeros(len(val), bool); info = {}
        for hold in FOLDS:
            c2 = relink(cal[~np.isin(sc_cal, hold)])
            Pf, ff, info = fit_predict(c2, val, D)
            te = np.isin(sc_val, hold)
            for m in METHODS:
                P[m][te] = Pf[m][te]
            flag[te] = ff[te]
        keep = ~np.isnan(P["cc"])
    y = (val.e2e > D).values.astype(int)
    u = (val.regime == "K3").values; ind = ~u
    blocks = np.arange(len(val)) // 50
    r = dict(cond=cond, file=os.path.basename(fn), mode=mode, q=q, D=D, n_val=int(keep.sum()), miss=y.mean(),
             K3_miss=y[u].mean(), sc_switch_K3=flag[u].mean(), sc_switch_in=flag[ind].mean(), **info)
    for m in METHODS:
        r[f"auroc_{m}"] = auroc(P[m], y); r[f"eceK3_{m}"] = ece(P[m][u], y[u]); r[f"eceIn_{m}"] = ece(P[m][ind], y[ind])
        r[f"brier_{m}"] = np.mean((P[m] - y) ** 2)
    for m in ["indep", "ar1", "ewma", "naive", "lriso"]:
        d, lo, hi = block_boot_diff(P["cc"], P[m], y, blocks, auroc, B=400)
        r[f"dAUC_cc_{m}"], r[f"dAUC_cc_{m}_lo"], r[f"dAUC_cc_{m}_hi"] = d, lo, hi
    for m in ["ar1", "ewma", "naive"]:
        d, lo, hi = block_boot_diff(P["ccsc"], P[m], y, blocks, auroc, B=400)
        r[f"dAUC_ccsc_{m}"], r[f"dAUC_ccsc_{m}_lo"], r[f"dAUC_ccsc_{m}_hi"] = d, lo, hi
    return r


def main():
    os.makedirs("results", exist_ok=True)
    if sys.argv[1] == "existing":
        conds, out = EXISTING, "results/ext_existing.csv"
    else:
        conds, out = {sys.argv[1]: sys.argv[2]}, f"results/ext_{sys.argv[1]}.csv"
    rows = []
    for c, dd in conds.items():
        for fn in sorted(glob.glob(f"{dd}/*.csv")):
            modes = ["standard"] + (["scene_disjoint"] if c in LIDAR or c.startswith("G10") else [])
            for mode in modes:
                r = score(fn, c, mode); rows.append(r)
                print(c, os.path.basename(fn), mode, "AUC cc/ar1/ewma/naive %.3f/%.3f/%.3f/%.3f" %
                      (r["auroc_cc"], r["auroc_ar1"], r["auroc_ewma"], r["auroc_naive"]),
                      "K3ECE cc/ccsc/lr/lriso/naive/ar1 %.3f/%.3f/%.3f/%.3f/%.3f/%.3f" %
                      (r["eceK3_cc"], r["eceK3_ccsc"], r["eceK3_lr"], r["eceK3_lriso"], r["eceK3_naive"], r["eceK3_ar1"]),
                      flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
