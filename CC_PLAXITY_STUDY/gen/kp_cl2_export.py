"""Addendum O: export inline-monitor parameters for lidar_ros_cl2 = kp_cl_export (unchanged) + one extra line ME2E_D.
    python kp_cl2_export.py <calib_none.csv> <calib_random.csv> <out_params.txt>
ME2E_D = me2e * (median e2e of release-degraded frames / median e2e of undegraded frames), both medians from the SAME
random-degradation calibration run (p = 0.5), so the ratio is not confounded by the interference schedule."""
import sys
import numpy as np, pandas as pd
import kp_cl_export


def main(cal_none, cal_rand, out):
    kp_cl_export.main(cal_none, out)
    r = pd.read_csv(cal_rand); r = r[(r.done == 1) & (r.regime != "K-1")].iloc[20:]
    ratio = float(r.e5[r.degraded == 1].median() / r.e5[r.degraded == 0].median())
    me2e = float(open(out).read().split("\n")[0].split()[3])
    open(out, "a").write(repr(me2e * ratio) + "\n")
    print(f"ME2E_D={me2e * ratio:.4f} ratio={ratio:.4f} n_degr={(r.degraded == 1).sum()} n_undegr={(r.degraded == 0).sum()}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
