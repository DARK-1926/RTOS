"""Addendum P: export inline-monitor parameters for lidar_ros_cl3 at a chosen deadline quantile.
    python kp_cl3_export.py <calib_none.csv> <calib_random.csv> <out_params.txt> <deadline>
<deadline> is a calibration quantile (e.g. 0.85), or "m<factor>" = round(factor x calibration quantile 0.85) (e.g. m1.6).
Same fit as kp_cl_export.fit (frozen, Addendum F) except that the deadline is the given quantile of calibration end-to-end
latency instead of the fixed 0.85; plus the ME2E_D line exactly as kp_cl2_export (Addendum O)."""
import sys
import numpy as np, pandas as pd
from g11_analysis import convert
from g_analysis import load
from toba_eq_alarms import attach, tdf
from toba_baseline import TobaPlaxityBaseline
from kp_methods import KalmanPlaxity, static_theta

ALPHA, GAMMA, NQ64 = 0.05, 0.01, 64


def fit(calfile, q):
    d = convert(calfile, 0)
    d = d[(d.done == 1) & (d.regime != "K-1")].drop(columns="done")
    tmp = calfile.replace(".csv", "_conv.csv"); d.to_csv(tmp, index=False)
    cal, _ = load(tmp); raw = pd.read_csv(tmp)
    cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
    D = float(np.round(np.quantile(cal.e2e, q))) if not isinstance(q, str) else float(np.round(float(q[1:]) * np.quantile(cal.e2e, 0.85)))
    kp = KalmanPlaxity(cal, nq=NQ64)
    P = kp.predict(cal, D); S = np.column_stack([P[0], P[1]])
    th0 = static_theta(S, (cal.e2e > D).values.astype(int), ALPHA)
    tb = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.8, th_max=1.0).fit(tdf(cal), deadline_ms=D)
    return cal, D, kp, th0, tb


def main(cal_none, cal_rand, out, q):
    q = q if str(q).startswith("m") else float(q); cal, D, kp, th0, tb = fit(cal_none, q)
    f = lambda a: " ".join(repr(float(x)) for x in a)
    L = [f([kp.lam, kp.q, kp.q0, kp.me2e, D, th0, ALPHA, GAMMA]), f(kp.med), f(kp.b), f(kp.v), str(NQ64)]
    L += [f(kp.B[k]) for k in range(4)]
    L.append(f([tb.thresholds[v] for v in range(1, 7)]))
    for v in range(1, 7):
        c = tb.plaxity_pmfs[v].to_plaxity_cdf(); L.append(f"{len(c)} " + f(c))
    r = pd.read_csv(cal_rand); r = r[(r.done == 1) & (r.regime != "K-1")].iloc[20:]
    ratio = float(r.e5[r.degraded == 1].median() / r.e5[r.degraded == 0].median())
    L.append(repr(float(kp.me2e * ratio)))
    open(out, "w").write("\n".join(L) + "\n")
    print(f"q={q} D={D} theta0={th0:.4f} ME2E_D={kp.me2e * ratio:.4f} ratio={ratio:.4f} n_cal={len(cal)}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
