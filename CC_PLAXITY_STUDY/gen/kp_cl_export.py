"""Addendum F: fit the inline monitors on a G12 calibration run (policy none) and export them for lidar_ros_cl.
    python kp_cl_export.py <g12_sSEED_calib.csv> <out_params.txt>
Fits (calibration frames only, same conversion as G11):
  KP-64   KalmanPlaxity(nq=64); D = calib p85
  ACI     theta0 = static_theta on the ACTIONABLE checkpoints (0, 1) for alpha = 0.05; gamma = 0.01
  Toba-eq TobaPlaxityBaseline(scale=10, linear_job, th_min 0.8, th_max 1.0), node 6 = overhead
Format (whitespace text): lam q q0 me2e D theta0 alpha gamma | med[5] | b[5] | v[5] | NQ | B[4][NQ] | thr[1..6] |
                          for v = 1..6: len cdf[len]
"""
import sys, os
import numpy as np, pandas as pd
from g11_analysis import convert
from g_analysis import load
from toba_eq_alarms import attach, tdf
from toba_baseline import TobaPlaxityBaseline
from kp_methods import KalmanPlaxity, static_theta

ALPHA, GAMMA, NQ64 = 0.05, 0.01, 64


def fit(calfile):
    d = convert(calfile, 0)
    d = d[(d.done == 1) & (d.regime != "K-1")].drop(columns="done")
    tmp = calfile.replace(".csv", "_conv.csv"); d.to_csv(tmp, index=False)
    cal, _ = load(tmp); raw = pd.read_csv(tmp)
    cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
    D = float(np.round(np.quantile(cal.e2e, 0.85)))
    kp = KalmanPlaxity(cal, nq=NQ64)
    P = kp.predict(cal, D); S = np.column_stack([P[0], P[1]])
    th0 = static_theta(S, (cal.e2e > D).values.astype(int), ALPHA)
    tb = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.8, th_max=1.0).fit(tdf(cal), deadline_ms=D)
    return cal, D, kp, th0, tb


def main(calfile, out):
    cal, D, kp, th0, tb = fit(calfile)
    f = lambda a: " ".join(repr(float(x)) for x in a)
    L = [f([kp.lam, kp.q, kp.q0, kp.me2e, D, th0, ALPHA, GAMMA]), f(kp.med), f(kp.b), f(kp.v), str(NQ64)]
    L += [f(kp.B[k]) for k in range(4)]
    L.append(f([tb.thresholds[v] for v in range(1, 7)]))
    for v in range(1, 7):
        c = tb.plaxity_pmfs[v].to_plaxity_cdf(); L.append(f"{len(c)} " + f(c))
    open(out, "w").write("\n".join(L) + "\n")
    print(f"D={D} theta0={th0:.4f} lam={kp.lam} toba_thr={[round(tb.thresholds[v], 3) for v in range(1, 7)]} n_cal={len(cal)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
