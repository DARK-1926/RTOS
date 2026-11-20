"""Fit every monitor on a lidar_cl calibration CSV and export the binary tables read by lidar_cl.c.
    python export_tables.py <calib.csv> <tables.bin> [q=0.85]
Fitting is exactly the Python analysis code (CCPlaxity, NaiveCP, toba_baseline), so C and Python agree.
Also: python export_tables.py --check <calib.csv> <val_none.csv>   compares C-logged release probabilities with Python.
"""
import struct, sys
import numpy as np, pandas as pd
from g_analysis import load
from cc_plaxity import CCPlaxity, EVID, NODES, CP_DONE, prep
from ext_methods import NaiveCP, cc_sc
from toba_baseline import TobaPlaxityBaseline


def calib_frame(fn):
    raw = pd.read_csv(fn)
    cal, _ = load(fn)
    m = raw.set_index("global_frame_id")
    return cal.assign(n3=m.loc[cal.order.values, "dur_node3_ms"].values, n4=m.loc[cal.order.values, "dur_node4_ms"].values)


def fit(fn, q=0.85):
    cal = calib_frame(fn)
    D = float(np.round(np.quantile(cal.e2e, q)))
    cc = CCPlaxity(cal, "K0"); nv = NaiveCP(cal)
    tb = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.8, th_max=1.0).fit(
        pd.DataFrame({"dur_node1_ms": cal.n1, "dur_node2_ms": cal.n2, "dur_node3_ms": cal.n3, "dur_node4_ms": cal.n4,
                      "dur_node5_ms": cal.n5, "dur_node6_ms": cal.ovh}), deadline_ms=D)
    return cal, D, cc, nv, tb


def export(fn, out, q=0.85):
    cal, D, cc, nv, tb = fit(fn, q)
    zg, _ = cc.kde[EVID[0]].__defaults__
    fgs = [cc.kde[c].__defaults__[1] for c in EVID]
    G, M = cc.R[0].shape; MN = len(nv.R[0])
    cdfs = [tb.plaxity_pmfs[v].to_plaxity_cdf().astype(np.float32) for v in range(1, 7)]
    with open(out, "wb") as f:
        f.write(struct.pack("<10i", G, M, MN, len(zg), *[len(c) for c in cdfs]))
        f.write(struct.pack("<8d", D, cc.rho, nv.med, zg[0], zg[-1], 0, 0, 0))
        f.write(np.array([cc.beta[c] for c in NODES], "<f8").tobytes())
        from cc_plaxity import GRID
        f.write(np.asarray(GRID, "<f8").tobytes()); f.write(np.asarray(cc.pi, "<f8").tobytes())
        f.write(np.array([tb.thresholds[v] for v in range(1, 7)], "<f8").tobytes())
        for fgc in fgs:
            f.write(np.asarray(fgc, "<f8").tobytes())
        for k in CP_DONE:
            f.write(np.asarray(cc.R[k], "<f4").tobytes())
        for k in CP_DONE:
            f.write(np.asarray(nv.R[k], "<f4").tobytes())
        for c in cdfs:
            f.write(c.astype("<f4").tobytes())
    print(f"D={D} rho={cc.rho} G={G} M={M} MN={MN} thr={[round(tb.thresholds[v], 4) for v in range(1, 7)]}")


def check(cfn, vfn):
    cal, D, cc, nv, tb = fit(cfn)
    v = pd.read_csv(vfn)
    o = pd.DataFrame({"n1": v.dur_node1_ms, "n2": v.dur_node2_ms, "F": np.maximum(v.dur_node3_ms, v.dur_node4_ms),
                      "n5": v.dur_node5_ms, "e2e": v.end_to_end_ms, "regime": v.regime, "split": v.split,
                      "order": v.global_frame_id, "cluster": v.bag_frame_idx})
    o = prep(o)
    p_cc = cc.predict(o, D, True)[0]; p_sc, flag = cc_sc(cc, nv, o, D); p_nv = nv.predict(o, D)[0]
    for name, a, b in [("cc", p_cc, v.p0_cc), ("ccsc", p_sc[0], v.p0_ccsc), ("naive", p_nv, v.p0_naive)]:
        print(name, "max |C - Python| =", np.nanmax(np.abs(a - b.values)), " mean =", np.nanmean(np.abs(a - b.values)))
    print("sc flag agreement:", (flag == v.sc_flag.values.astype(bool)).mean())


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2], sys.argv[3])
    else:
        export(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.85)
