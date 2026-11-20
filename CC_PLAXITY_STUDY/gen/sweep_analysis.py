"""Round E: threshold-free comparison + ACI guarantee check on ALL existing runs (exploratory for these runs;
the confirmatory test is Addendum E on fresh ROS 2 / new-dataset runs).  Run from gen/:
    python sweep_analysis.py            -> results/sweep_runs.csv, results/aci_runs.csv
Every method is scored with the SAME rule: alarm at the first checkpoint whose P(miss) exceeds one global
threshold; sweeping that threshold gives the full ROC (no method is judged at someone else's thresholds).
Checkpoints: release (N1 start), N2 start, N3||N4 start, N5 start; Toba-eq additionally keeps its N6 check.
"""
import glob, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.abspath(".."))
from cc_plaxity import CCPlaxity, load_lidar, load_pilot, load_hclock, prep, CP_DONE
from g_analysis import load
from ext_methods import NaiveCP, LRIso, cc_sc, EWMA
from toba_baseline import TobaPlaxityBaseline
from toba_eq_alarms import attach, tdf, RUNS, STAGE0
from kp_methods import KalmanPlaxity, aci_alarms, aci_bound, static_theta, _ewma

ALPHAS, GAMMA, LAG = [0.02, 0.05], 0.01, 1


class EWMACP(NaiveCP):
    """NaiveScale at every checkpoint, but the scale is exp(EWMA of past log(e2e/med)) (lambda as in EWMA)."""
    def __init__(self, cal):
        super().__init__(cal); self.lam = EWMA(cal).lam

    def scale(self, df):
        m = _ewma(np.log(df.e2e.values / self.med), df.has_prev.values, self.lam)
        return np.where(np.isnan(m), 1.0, np.exp(m))


def toba_scores(cal, val, D):
    tb = TobaPlaxityBaseline(scale=10, threshold_mode="fixed", fixed_threshold=0.95).fit(tdf(cal), deadline_ms=D)
    ast = np.column_stack([val.e0, val.e1, val.e2, val.e2, val.e3, val.e3 + val.n5])
    S = np.zeros((len(val), 6))
    for v in range(1, 7):
        cdf = tb.plaxity_pmfs[v].to_plaxity_cdf()
        b = np.clip(np.round(np.maximum(ast[:, v - 1], 0) * tb.scale).astype(int), 0, None)
        S[:, v - 1] = 1 - np.where(b < len(cdf), cdf[np.minimum(b, len(cdf) - 1)], 0.0)
    # checkpoints: 0=N1, 1=N2, 2=max(N3,N4), 3=N5, 4=N6 (Toba-only extra check)
    return np.column_stack([S[:, 0], S[:, 1], np.maximum(S[:, 2], S[:, 3]), S[:, 4], S[:, 5]])


def all_scores(cal, val, D, nominal="K0"):
    Ds = list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95])))
    cc = CCPlaxity(cal, nominal); nv = NaiveCP(cal)
    out, cost, nanfrac = {}, {}, {}

    def put(name, f):
        t = time.perf_counter(); P = f(); cost[name] = (time.perf_counter() - t) / len(val) * 1e6
        S = P if isinstance(P, np.ndarray) else np.column_stack([P[k] for k in sorted(CP_DONE)])
        nanfrac[name] = float(np.isnan(S).mean())
        for j in range(S.shape[1]):            # a monitor that cannot update keeps its previous checkpoint value
            bad = np.isnan(S[:, j])
            S[bad, j] = S[bad, j - 1] if j > 0 else 0.5
        out[name] = S
    put("Toba-eq", lambda: toba_scores(cal, val, D))
    put("CC", lambda: cc.predict(val, D, True))
    put("CC-SC", lambda: cc_sc(cc, nv, val, D)[0])
    put("NaiveCP", lambda: nv.predict(val, D))
    put("EWMA-CP", lambda: EWMACP(cal).predict(val, D))
    put("LR-iso", lambda: LRIso(cal, Ds, D).predict(val))
    kp = KalmanPlaxity(cal)
    put("KP", lambda: kp.predict(val, D))
    calP = {"KP": kp.predict(cal, D), "EWMA-CP": EWMACP(cal).predict(cal, D), "NaiveCP": nv.predict(cal, D),
            "CC": cc.predict(cal, D, True), "Toba-eq": toba_scores(cal, cal, D)}
    calP = {k: v if isinstance(v, np.ndarray) else np.column_stack([v[j] for j in sorted(CP_DONE)]) for k, v in calP.items()}
    calP = {k: np.nan_to_num(v, nan=0.5) for k, v in calP.items()}
    return out, calP, cost, nanfrac


def safe_auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def sweep_metrics(S, y, val, D, fpr_target=0.10):
    smax = S.max(1)
    neg = np.sort(smax[y == 0])
    thr = neg[int(np.ceil((1 - fpr_target) * len(neg))) - 1] if len(neg) else 1.0     # FPR <= target
    alarm = S > thr
    first = np.where(alarm.any(1), alarm.argmax(1), -1)
    times = np.column_stack([val.e0, val.e1, val.e2, val.e3, val.e3 + val.n5])[:, :S.shape[1]]
    tp = (first >= 0) & (y == 1)
    lead = D - times[np.arange(len(y)), np.maximum(first, 0)]
    return dict(auc_rel=safe_auc(y, S[:, 0]), auc_act=safe_auc(y, S[:, :2].max(1)), auc_any=safe_auc(y, smax),
                ap_any=average_precision_score(y, smax) if y.sum() else np.nan,
                recall_fpr10=tp.sum() / max(y.sum(), 1), lead_ms_fpr10=np.mean(lead[tp]) if tp.any() else np.nan,
                rel_share_fpr10=np.mean(first[tp] == 0) if tp.any() else np.nan)


def runs():
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
            yield "stage0_" + ds, f"stage0_{ds}_D{D}", cal, val, float(D), nominal
    dirs = dict(RUNS); dirs.update({"G10a": "data/G10a/combined", "G10b": "data/G10b/combined", "G10s": "data/G10s"})
    for c, dd in dirs.items():
        for fn in sorted(glob.glob(f"{dd}/*.csv")):
            cal, val = load(fn); raw = pd.read_csv(fn)
            cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
            val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
            yield c, os.path.basename(fn), cal, val, float(np.round(np.quantile(cal.e2e, 0.85))), "K0"


def main():
    rows, arows = [], []
    for cond, run, cal, val, D, nominal in runs():
        y = (val.e2e > D).values.astype(int); ycal = (cal.e2e > D).values.astype(int)
        P, calP, cost, nanfrac = all_scores(cal, val, D, nominal)
        for m, S in P.items():
            rows.append(dict(cond=cond, run=run, D=D, method=m, miss_rate=y.mean(), us_per_frame_py=cost[m], nan_frac=nanfrac[m],
                             **sweep_metrics(S, y, val, D)))
        for m, S in P.items():
            if m not in calP:
                continue
            for a in ALPHAS:
                th0 = static_theta(calP[m], ycal, a)
                st = S.max(1) > th0
                f, e, th = aci_alarms(S, y, a, GAMMA, th0, LAG)
                arows.append(dict(cond=cond, run=run, method=m, alpha=a, T=len(y), miss_rate=y.mean(),
                                  static_unwarned=np.mean((y == 1) & ~st), static_fpr=st[y == 0].mean(),
                                  static_alarm=st.mean(),
                                  aci_unwarned=e.mean(), aci_fpr=(f >= 0)[y == 0].mean(), aci_alarm=(f >= 0).mean(),
                                  aci_bound=aci_bound(len(y), GAMMA, LAG)))
        print(cond, run, flush=True)
        pd.DataFrame(rows).to_csv("results/sweep_runs.csv", index=False)
        pd.DataFrame(arows).to_csv("results/aci_runs.csv", index=False)


if __name__ == "__main__":
    main()
