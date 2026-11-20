"""Addendum F confirmatory analysis: G12, ROS 2 closed loop with inline KP-64 + ACI (seeds 121-123).
Run from gen/:  python g12_analysis.py   -> results/g12_closedloop.csv, g12_summary.txt
Per seed: params file data/G12/params_s{seed}.txt (exported from that seed's calibration run by kp_cl_export.py),
validation runs data/G12/g12_s{seed}_val_{none,kp,toba,random}.csv (same seeded interference schedule).
Frames: drop the first 20, frames not completed (counted, expected 0) and transition frames (K-1).
"""
import numpy as np, pandas as pd
from kp_methods import KalmanPlaxity, aci_bound
from kp_cl_export import fit
from g11_analysis import convert

POL = ["none", "kp", "toba", "random"]
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def params(fn):
    L = open(fn).read().split("\n"); h = [float(x) for x in L[0].split()]
    return dict(D=h[4], theta0=h[5], alpha=h[6], gamma=h[7])


def cpp_vs_python(seed, prm):
    """Replay KP-64 in Python on the 'none' run (all completed frames, one chain, as the inline monitor saw them)."""
    cal, D, kp, th0, tb = fit(f"data/G12/g12_s{seed}_calib_none.csv")
    r = pd.read_csv(f"data/G12/g12_s{seed}_val_none.csv"); r = r[r.done == 1].reset_index(drop=True)
    fork = np.minimum(r.s3, r.s4); join = np.maximum(r.e3, r.e4)
    df = pd.DataFrame({"n1": r.s2, "n2": fork - r.s2, "F": join - fork, "n5": r.e5 - join, "ovh": 1e-3, "e2e": r.e5,
                       "has_prev": np.r_[False, np.ones(len(r) - 1, bool)], "e0": 0.0, "e1": r.s2, "e2": fork, "e3": join})
    for c in ["n1", "n2", "F", "n5"]:
        df[c] = np.clip(df[c], 1e-3, None)
    P = kp.predict(df, prm["D"])
    d = np.abs(np.column_stack([P[k] for k in range(4)]) - r[["p0", "p1", "p2", "p3"]].values)
    return float(np.nanmax(d)), float(np.nanmedian(d))


def main():
    q = pd.read_csv("data/G10a/quality.csv").set_index("scan_idx")
    rows = []
    for seed in [121, 122, 123]:
        prm = params(f"data/G12/params_s{seed}.txt"); D = prm["D"]
        mx, md = cpp_vs_python(seed, prm)
        say(f"--- G12 seed {seed}: D = {D} ms, theta0 = {prm['theta0']:.4f}; C++ vs Python KP-64 on the none run: "
            f"max |diff| {mx:.2e}, median {md:.2e}")
        for p in POL:
            r = pd.read_csv(f"data/G12/g12_s{seed}_val_{p}.csv").iloc[20:]
            nd = int((r.done == 0).sum()); r = r[(r.done == 1) & (r.regime != "K-1")]
            k3 = (r.regime == "K3").values; miss = (r.e5 > D).values; al = (r.alarm >= 0).values
            unw = miss & ~al
            sd = np.where(r.degraded > 0, q.loc[r.scan, "sector_diff_m1"].values, 0)
            rows.append(dict(seed=seed, policy=p, D=D, n=len(r), not_completed=nd, miss=miss.mean(), miss_K3=miss[k3].mean(),
                             miss_K0_2=miss[~k3].mean(), degraded=(r.degraded > 0).mean(), alarm=al.mean(),
                             unwarned_actionable=unw.mean(), aci_bound=aci_bound(len(r), prm["gamma"], 1),
                             sector_change_mean=sd.mean(), us_kp_med=r.us_kp.median(), us_kp_p99=r.us_kp.quantile(0.99),
                             us_kp_p99_K3=np.quantile(r.us_kp[k3], 0.99) if k3.any() else np.nan, us_kp_max=r.us_kp.max(),
                             us_tb_med=r.us_tb.median(), e2e_med=r.e5.median(), e2e_p99=r.e5.quantile(0.99)))
    R = pd.DataFrame(rows); R.to_csv("results/g12_closedloop.csv", index=False)
    pd.set_option("display.width", 250)
    say("\n" + R.round(4).to_string())
    W = R.pivot(index="seed", columns="policy")
    kp = R[R.policy == "kp"].set_index("seed")
    f1 = (kp.us_kp_p99_K3 < 100).sum()
    f2 = ((kp.unwarned_actionable - 0.05).abs() <= kp.aci_bound).sum()
    f3 = (W.miss.kp < W.miss.none).sum(); f4 = (W.miss.kp < W.miss.random).sum(); f5 = (W.degraded.kp < W.degraded.toba).sum()
    say(f"\nF1 inline KP-64 p99 (K3 frames, kp run) < 100 us: {f1}/3 -> {'PASS' if f1 == 3 else 'FAIL'}")
    say(f"F2 ACI guarantee (|unwarned actionable - 0.05| <= bound), kp run: {f2}/3 -> {'PASS' if f2 == 3 else 'FAIL'}")
    say(f"F3 miss(kp) < miss(none): {f3}/3 -> {'PASS' if f3 == 3 else 'FAIL'}")
    say(f"F4 miss(kp) < miss(random, same budget): {f4}/3 -> {'PASS' if f4 >= 2 else 'FAIL'}")
    say(f"F5 degraded(kp) < degraded(toba): {f5}/3 -> {'PASS' if f5 == 3 else 'FAIL'}")
    open("results/g12_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
