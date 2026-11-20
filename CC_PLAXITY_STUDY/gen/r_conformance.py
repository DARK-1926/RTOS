"""Implementation conformance: replay the logged closed-loop runs (Addenda P and Q) and check that every release decision the
deployed C++ monitor made under the window rules equals the rule as stated in the paper.

  rule W  (policy kpmd):  forced to level 2  <=>  #{j in [f-K+1, f-1]: j known late, or j running and not at level 2} >= m
  rule Wu (policy kpmwh): forced (level 1)   <=>  #{j in [f-K+1, f-1]: j known unwarned miss, or j unwarned and running} >= m

Frame j is "known" at the release of f if it finished before f was released. Releases are every P = 100 ms; the release time of j is
taken as j * P (the logged columns are relative to each frame's own release). A frame that never finished stays "running".
Usage: python r_conformance.py   -> results/r_conformance.txt
"""
import glob, numpy as np, pandas as pd
from o_confirm import params
M, K, P = 4, 20, 100.0
out, tot_dec, tot_mis, tot_amb = [], 0, 0, 0
for fn in sorted(glob.glob("data/P/cl_s23?_val_*_kpm*.csv") + glob.glob("data/Q/cl_s24?_val_*_kpm*.csv")):
    pol = fn.rsplit("_", 1)[1][:-4]
    if pol not in ("kpmd", "kpmwh"):
        continue
    fn = fn.replace("\\", "/"); sub, base = fn.split("/")[1], fn.split("/")[-1]
    seed, load = int(base.split("_")[1][1:]), base.split("_")[3]
    D = params(f"data/{sub}/params_s{seed}_{load}.txt")["D"]
    a = pd.read_csv(fn).sort_values("frame").reset_index(drop=True)
    n = len(a); done = a.done.values == 1; e5 = a.e5.values; lvl2 = a.degraded.values == 3
    late = done & (e5 > D); unw = (a.alarm.values < 0)
    fin = np.where(done, np.arange(n) * P + e5, np.inf)                     # absolute finish time
    mism = amb = 0
    for f in range(n):
        t = f * P; c = 0
        for j in range(max(0, f - (K - 1)), f):
            known = fin[j] <= t
            if abs(fin[j] - t) < 5.0: amb += 1                               # finish within 5 ms of a release: timing ambiguous
            if pol == "kpmd":
                c += (known and late[j]) or (not known and not lvl2[j])
            else:
                c += (known and late[j] and unw[j]) or (not known and unw[j])
        pred = c >= M
        logged = a.forced.values[f] == (2 if pol == "kpmd" else 1)
        mism += pred != logged
    tot_dec += n; tot_mis += mism; tot_amb += amb
    out.append(f"{sub} seed {seed} {load:5s} {pol:6s} frames {n:5d}  forced {int((a.forced.values > 0).sum()):4d}  "
               f"mismatches {mism}  ambiguous-timing pairs {amb}  max e5 {np.nanmax(e5):.1f} ms")
out.append(f"TOTAL release decisions {tot_dec}, mismatches {tot_mis}, ambiguous-timing pairs {tot_amb}")
open("results/r_conformance.txt", "w").write("\n".join(out)); print("\n".join(out))
