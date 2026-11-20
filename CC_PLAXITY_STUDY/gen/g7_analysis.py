"""G7 (Addendum A): CC vs CC-R (eps 0.05; sensitivity 0.01, 0.10) vs NaiveScale, at D = calib p85, release checkpoint.
Usage: python g7_analysis.py <csv> [<csv> ...]"""
import sys
import numpy as np, pandas as pd
import g_analysis as g
from cc_plaxity import CCPlaxity, ece, auroc
from cc_robust import CCRobust
rows = []
for fn in sys.argv[1:]:
    cal, val = g.load(fn); D = float(np.round(np.quantile(cal.e2e, 0.85)))
    y = (val.e2e > D).astype(int).values; u = (val.regime == "K3").values; ind = ~u
    r = dict(file=fn.split("/")[-1], D=D, K3_miss=y[u].mean())
    m = {"cc": CCPlaxity(cal, "K0")}
    for e in (0.01, 0.05, 0.10):
        m[f"ccR{e:g}"] = CCRobust(cal, "K0", eps=e)
    for n, mod in m.items():
        p = mod.predict(val, D, True)[0]
        r[f"eceK3_{n}"] = ece(p[u], y[u]); r[f"eceIn_{n}"] = ece(p[ind], y[ind]); r[f"auroc_{n}"] = auroc(p, y)
        r[f"rho_{n}"] = mod.rho
    pn = g.naive_scale(cal, val, D)
    r["eceK3_naive"] = ece(pn[u], y[u]); r["eceIn_naive"] = ece(pn[ind], y[ind]); r["auroc_naive"] = auroc(pn, y)
    r["G7a"] = r["eceK3_ccR0.05"] < r["eceK3_cc"]
    r["G7b"] = r["eceIn_ccR0.05"] <= r["eceIn_cc"] + 0.02
    rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
cols = ["file", "K3_miss", "eceK3_cc", "eceK3_ccR0.05", "eceK3_naive", "eceIn_cc", "eceIn_ccR0.05", "eceIn_naive",
        "auroc_cc", "auroc_ccR0.05", "auroc_naive", "rho_cc", "rho_ccR0.05", "G7a", "G7b"]
print(R[cols].round(3).to_string(index=False))
print("sensitivity eceK3:", R[[c for c in R if c.startswith("eceK3_ccR")]].round(3).to_dict("list"))
R.to_csv(sys.stdout if False else "results/g7_last.csv", index=False)
