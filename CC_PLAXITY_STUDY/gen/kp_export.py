"""Export KP parameters + a frame stream for kp_monitor.c, and check C == Python.
    python kp_export.py export <trace.csv> <outprefix>     -> <outprefix>_params.txt, _frames.csv, _D.txt, _py.csv
    python kp_export.py check  <outprefix> <c_out.csv>
Trace: a G10-style combined CSV (calib + val); KP is fit on calib and replayed on val.
"""
import sys
import numpy as np, pandas as pd
from g_analysis import load
from kp_methods import KalmanPlaxity, NQ
from cc_plaxity import NODES


def export(fn, pre):
    cal, val = load(fn)
    D = float(np.round(np.quantile(cal.e2e, 0.85)))
    kp = KalmanPlaxity(cal)
    with open(pre + "_params.txt", "w") as f:
        f.write(" ".join(repr(float(x)) for x in (kp.lam, kp.q, kp.q0, kp.me2e)) + "\n")
        for arr in (kp.med, kp.b, kp.v):
            f.write(" ".join(repr(float(a)) for a in arr) + "\n")
        f.write(f"{NQ}\n")
        for k in range(4):
            f.write(" ".join(repr(float(a)) for a in kp.B[k]) + "\n")
    val[["n1", "n2", "F", "n5", "e2e"]].assign(hp=val.has_prev.astype(int)).to_csv(pre + "_frames.csv", index=False,
                                                                                    header=False, float_format="%.6f")
    open(pre + "_D.txt", "w").write(f"{D}\n")
    P = kp.predict(val, D)
    pd.DataFrame({f"p{k}": P[k] for k in range(4)}).to_csv(pre + "_py.csv", index=False)
    print("D", D, "n", len(val))


def check(pre, cfile):
    a = pd.read_csv(pre + "_py.csv"); c = pd.read_csv(cfile)
    d = (a.values - c[["p0", "p1", "p2", "p3"]].values)
    print("max |C - Python| =", np.abs(d).max(), " us median", c.us.median(), " p99", c.us.quantile(0.99), " max", c.us.max())


if __name__ == "__main__":
    export(sys.argv[2], sys.argv[3]) if sys.argv[1] == "export" else check(sys.argv[2], sys.argv[3])
