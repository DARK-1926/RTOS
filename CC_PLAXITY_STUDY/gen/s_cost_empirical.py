"""Cost theorem on the logged kpmwh runs (Addenda P, Q): forced F vs (K-1)U + (L-1)N, L = observed max lag; and forced rate vs alpha."""
import glob, numpy as np, pandas as pd
from o_confirm import params
M, K, P = 4, 20, 100.0; out = []
for fn in sorted(glob.glob("data/P/cl_s23?_val_*_kpmwh.csv") + glob.glob("data/Q/cl_s24?_val_*_kpmwh.csv")):
    fn = fn.replace("\\", "/"); sub, b = fn.split("/")[1], fn.split("/")[-1]; s, load = int(b.split("_")[1][1:]), b.split("_")[3]
    pr = params(f"data/{sub}/params_s{s}_{load}.txt"); D = pr["D"]
    a = pd.read_csv(fn).sort_values("frame"); n = len(a); done = a.done.values == 1; e5 = a.e5.values
    unw = a.alarm.values < 0; F = int((a.forced.values == 1).sum()); U = int((unw & (~done | (e5 > D))).sum()); N = int(unw.sum())
    lag = int(np.ceil(np.nanmax(np.where(done, e5, np.nan)) / P)); bound = ((K - 1) * U + (lag - 1) * N) / M
    out.append(f"{sub} s{s} {load:5s} n={n} alpha={pr['alpha']} L={lag} forced F={F} ({F/n:.3%})  U={U} ({U/n:.3%})  bound on F={bound:.0f}  "
               f"(K-1)U/m={(K-1)*U/M:.0f}  holds={F <= bound}")
open("results/s_cost_empirical.txt", "w").write("\n".join(out)); print("\n".join(out))
