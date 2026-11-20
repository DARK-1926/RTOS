"""Round G analysis for telemetry runs (lidar_ros_tm). EXPLORATORY on the pilot seeds; the same code is frozen for the
confirmatory seeds (Addendum G).   python g13_analysis.py <data_dir> <seed> [<seed> ...]   -> <data_dir>/g13_metrics.csv
Methods: Toba-eq, EWMA-CP, KP, KP-T (KP with telemetry prior), KP-T(tel only), and each '+TL' hybrid
(the method at checkpoints 0-1, Toba-eq at the later checkpoints).
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import prep, block_boot_diff, auroc, ece, CP_DONE
from g11_analysis import convert
from toba_eq_alarms import tdf
from sweep_analysis import toba_scores, EWMACP, sweep_metrics, GAMMA, LAG
from kp_methods import KalmanPlaxity, aci_alarms, static_theta
from kpt_methods import KalmanPlaxityT, telemetry_features


def build(fn, split, offset):
    raw = pd.read_csv(fn)
    d = convert(fn, offset)
    tm = telemetry_features(raw)
    d = pd.concat([d.reset_index(drop=True), tm], axis=1)
    d = d[(d.done == 1) & (d.regime != "K-1")].iloc[20:]
    o = pd.DataFrame({"n1": d.dur_node1_ms, "n2": d.dur_node2_ms, "F": np.maximum(d.dur_node3_ms, d.dur_node4_ms),
                      "n5": d.dur_node5_ms, "e2e": d.end_to_end_ms, "regime": d.regime, "split": split,
                      "order": d.global_frame_id, "cluster": d.bag_frame_idx, "n3": d.dur_node3_ms, "n4": d.dur_node4_ms,
                      "tm_u": d.tm_u, "tm_r": d.tm_r, "tm_w": d.tm_w}).reset_index(drop=True)
    return prep(o)


def mat(P):
    return np.column_stack([P[k] for k in sorted(CP_DONE)])


def main(ddir, seeds):
    rows = []
    for s in seeds:
        cal = build(f"{ddir}/g13_s{s}_calib.csv", "calib", 0); val = build(f"{ddir}/g13_s{s}_val.csv", "val", 100000)
        D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
        kp, kpt = KalmanPlaxity(cal, nq=64), KalmanPlaxityT(cal, nq=64)
        kpo = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w"))
        T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
        ew = EWMACP(cal)
        S = {"Toba-eq": (T, Tc), "EWMA-CP": (mat(ew.predict(val, D)), mat(ew.predict(cal, D))),
             "KP": (mat(kp.predict(val, D)), mat(kp.predict(cal, D))), "KP-T": (mat(kpt.predict(val, D)), mat(kpt.predict(cal, D))),
             "KP-T(tel only)": (mat(kpo.predict(val, D)), mat(kpo.predict(cal, D)))}
        for m in ["EWMA-CP", "KP", "KP-T", "KP-T(tel only)"]:
            a, b = S[m]
            S[m + "+TL"] = (np.column_stack([a[:, 0], a[:, 1], T[:, 2], T[:, 3], T[:, 4]]),
                            np.column_stack([b[:, 0], b[:, 1], Tc[:, 2], Tc[:, 3], Tc[:, 4]]))
        k3 = (val.regime == "K3").values; blocks = np.arange(len(val)) // 50
        print(f"seed {s}: D={D} miss={y.mean():.3f} KP-T prior R2={kpt.r2_prior:.3f} coef={np.round(kpt.coef, 3)} "
              f"tel-only R2={kpo.r2_prior:.3f}", flush=True)
        for m, (V, C) in S.items():
            r = dict(seed=s, method=m, D=D, miss=y.mean(), **sweep_metrics(V, y, val, D),
                     ece_rel_K3=ece(V[k3, 0], y[k3]) if k3.any() else np.nan, ece_rel=ece(V[:, 0], y))
            if m != "KP":
                d = block_boot_diff(V[:, :2].max(1), S["KP"][0][:, :2].max(1), y, blocks, auroc)
                r.update(dact_vs_KP=d[0], lo_vs_KP=d[1], hi_vs_KP=d[2])
            for al in [0.02, 0.05]:
                th0 = static_theta(C[:, :2], yc, al); f, e, _ = aci_alarms(V[:, :2], y, al, GAMMA, th0, LAG)
                r[f"aci_fpr_{al}"] = (f >= 0)[y == 0].mean(); r[f"aci_unw_{al}"] = e.mean()
            rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(f"{ddir}/g13_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.groupby("method")[["auc_rel", "auc_act", "auc_any", "recall_fpr10", "lead_ms_fpr10", "aci_fpr_0.02", "aci_fpr_0.05",
                               "ece_rel_K3", "dact_vs_KP"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
