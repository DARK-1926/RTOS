"""Alarm results with the EQUATION-LEVEL Toba-2026 baseline (TECS 2026 Eqs. 8-18).
Run from CC_PLAXITY_STUDY/gen:  python toba_eq_alarms.py  -> results/toba_eq_alarms.csv

Toba-eq: D:/RTKK/FINAL_SUBMISSION_BUNDLE/detectors/toba_baseline.py (copied to gen/toba_baseline.py), i.e.
  exit-node init (Eq. 9), convolutional subtraction (Eq. 10), independent fork MIN rule (Eqs. 11-12),
  plaxity-CDF (Eq. 8), 0.1 ms bins, checks at the start of every node. Our DAG maps onto its 6-node chain as
  N1 -> N2 -> {N3 || N4} -> N5 -> N6, with N6 = measured overhead (e2e minus the structural sum).
  Pooled calibration PMFs (as Toba profile offline).
Thresholds (paper Sec. 4.2): variable thresholds, th_min = 0.8, th_max = 1.0, workload = mean execution time
  (Eq. 14, the paper's recommended setting) and = job count (Eq. 13); plus a global 0.95 threshold.
Every other method gets the SAME per-node thresholds theta_v applied to its own P(meet) = 1 - P(miss) at the
  start of N1, N2, N3/N4 and N5 (CC has no check at N6; Toba keeps its extra N6 check).
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.abspath(".."))
from cc_plaxity import CCPlaxity, LR, load_lidar, load_pilot, load_hclock, prep, CP_DONE
from cc_transfer_lr import LRdeadline
from g_analysis import load
from ext_methods import NaiveCP, cc_sc
from toba_baseline import TobaPlaxityBaseline

RUNS = {"ArmA": "../board_armA_schedutil", "ArmB": "../board_armB_spin", "ArmC": "../board_armC_perfgov",
        "G1": "data/G1", "G2": "data/G2", "G3": "data/G3", "G4x": "data/G4x", "G8a": "data/G8a", "G8b": "data/G8b"}
STAGE0 = [("lidar", load_lidar, [35, 50, 80]), ("pilot", load_pilot, [30, 35]), ("hclock", load_hclock, [28, 32])]
POLICIES = {"var_job_0.8_1.0": dict(threshold_mode="linear_job", th_min=0.8, th_max=1.0),
            "var_time_0.8_1.0": dict(threshold_mode="linear_time", th_min=0.8, th_max=1.0),
            "global_0.95": dict(threshold_mode="fixed", fixed_threshold=0.95)}


def attach(o, raw, order_col, c3, c4):
    m = raw.set_index(order_col)
    return o.assign(n3=m.loc[o.order.values, c3].values, n4=m.loc[o.order.values, c4].values)


def tdf(df):
    """columns expected by toba_baseline (node 6 = overhead)."""
    return pd.DataFrame({"dur_node1_ms": df.n1, "dur_node2_ms": df.n2, "dur_node3_ms": df.n3, "dur_node4_ms": df.n4,
                         "dur_node5_ms": df.n5, "dur_node6_ms": df.ovh})


def alarms_generic(P, df, D, th):
    """P: dict cp->P(miss). th: toba thresholds by node. Checks at start of N1 (cp0), N2 (cp1), N3/4 (cp2), N5 (cp3)."""
    node_th = {0: th[1], 1: th[2], 2: min(th[3], th[4]), 3: th[5]}
    n = len(df); first = np.full(n, -1)
    for k in sorted(CP_DONE):
        hit = ((1 - P[k]) < node_th[k]) & (first < 0); first[hit] = k
    ast = df[["e0", "e1", "e2", "e3"]].values
    t = np.where(first >= 0, ast[np.arange(n), np.maximum(first, 0)], np.nan)
    return first, D - t


def alarms_toba(tb, df, D):
    n = len(df); first = np.full(n, -1); t = np.full(n, np.nan)
    ast = np.column_stack([df.e0, df.e1, df.e2, df.e2, df.e3, df.e3 + df.n5])
    for v in range(1, 7):
        cdf = tb.plaxity_pmfs[v].to_plaxity_cdf()
        b = np.clip(np.round(np.maximum(ast[:, v - 1], 0) * tb.scale).astype(int), 0, None)
        pm = np.where(b < len(cdf), cdf[np.minimum(b, len(cdf) - 1)], 0.0)
        hit = (pm < tb.thresholds[v]) & (first < 0)
        first[hit] = v; t[hit] = ast[hit, v - 1]
    return first, D - t


def summ(name, first, lead, y, **kw):
    a = first >= 0; tp = a & (y == 1)
    early = tp & (first >= 0) & (first <= (2 if name == "Toba-eq" else 1))   # alarm before N2 finishes starting
    return dict(method=name, recall=a[y == 1].mean(), fpr=a[y == 0].mean(),
                precision=tp.sum() / max(a.sum(), 1), lead_ms_tp=np.nanmean(lead[tp]) if tp.any() else np.nan,
                late_alarms=(lead[tp] < 0).mean() if tp.any() else np.nan,
                tp_alarm_by_N2start=early.sum() / max(tp.sum(), 1), **kw)


def evaluate(cal, val, D, nominal, tag):
    y = (val.e2e > D).values.astype(int)
    cc = CCPlaxity(cal, nominal); nv = NaiveCP(cal)
    Ds = list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95])))
    preds = {"CC-plaxity": cc.predict(val, D, True), "CC-SC": cc_sc(cc, nv, val, D)[0],
             "LR-deadline": LRdeadline(cal, Ds).predict(val, D), "NaiveScale": nv.predict(val, D)}
    rows = []
    for pol, cfg in POLICIES.items():
        tb = TobaPlaxityBaseline(scale=10, **cfg).fit(tdf(cal), deadline_ms=D)
        f, l = alarms_toba(tb, val, D); rows.append(summ("Toba-eq", f, l, y, policy=pol, D=D, run=tag))
        for name, P in preds.items():
            f, l = alarms_generic(P, val, D, tb.thresholds); rows.append(summ(name, f, l, y, policy=pol, D=D, run=tag))
    return rows


def main():
    rows = []
    for ds, loader, Ds in STAGE0:
        src = "../input_traces" if os.path.isdir("../input_traces") else "../../RTOS-project/results"
        o, nominal = loader(src)
        raw = pd.read_csv(f"{src}/" + {"lidar": "multinode_real_lidar_traces.csv", "pilot": "pilot_finer_contention_traces.csv",
                                         "hclock": "h_clock_trials.csv"}[ds])
        if ds == "hclock":
            raw = raw.sort_values(["session_id", "trial_idx"]).reset_index(drop=True); raw["order"] = np.arange(len(raw))
            o = attach(o, raw, "order", "d3", "d4")
        else:
            o = attach(o, raw, {"lidar": "global_frame_id", "pilot": "pilot_frame_id"}[ds], "dur_node3_ms", "dur_node4_ms")
        o = prep(o)
        cal = o[o.split == "calib"].reset_index(drop=True); val = o[o.split == "val"].reset_index(drop=True)
        for D in Ds:
            r = evaluate(cal, val, D, nominal, f"stage0_{ds}"); [x.update(cond="stage0_" + ds) for x in r]; rows += r
            print(ds, D, flush=True)
    for c, dd in RUNS.items():
        for fn in sorted(glob.glob(f"{dd}/*.csv")):
            cal, val = load(fn); raw = pd.read_csv(fn)
            cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
            val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms"); D = float(np.round(np.quantile(cal.e2e, 0.85)))
            r = evaluate(cal, val, D, "K0", os.path.basename(fn)); [x.update(cond=c) for x in r]; rows += r
            print(c, os.path.basename(fn), flush=True)
        pd.DataFrame(rows).to_csv("results/toba_eq_alarms.csv", index=False)
    R = pd.DataFrame(rows); R.to_csv("results/toba_eq_alarms.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[R.policy == "var_job_0.8_1.0"].groupby(["cond", "method"])[["recall", "fpr", "precision", "lead_ms_tp",
          "late_alarms", "tp_alarm_by_N2start"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
