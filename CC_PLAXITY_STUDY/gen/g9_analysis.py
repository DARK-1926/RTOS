"""G9 (Addendum C): CC vs CC-E vs LR on G8a seeds 53-55 (plus primary G8 metrics). Usage: python g9_analysis.py <csv>..."""
import sys
import numpy as np, pandas as pd
import g_analysis as g
from cc_plaxity import CCPlaxity, ece, auroc
from cc_fullevid import CCFullEvidence
from cc_transfer_lr import LRdeadline
rows = []
for fn in sys.argv[1:]:
    cal, val = g.load(fn); D = float(np.round(np.quantile(cal.e2e, 0.85)))
    y = (val.e2e > D).astype(int).values; u = (val.regime == "K3").values
    pc = CCPlaxity(cal, "K0").predict(val, D)[0]; pe = CCFullEvidence(cal, "K0").predict(val, D)[0]
    pl = LRdeadline(cal, list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95])))).predict(val, D)[0]
    r = dict(file=fn.split("/")[-1], K3_n=int(u.sum()), K3_miss=y[u].mean(), eceK3_cc=ece(pc[u], y[u]), eceK3_ccE=ece(pe[u], y[u]),
             eceK3_lr=ece(pl[u], y[u]), auroc_cc=auroc(pc, y), auroc_ccE=auroc(pe, y))
    r["G9a"] = r["eceK3_ccE"] < r["eceK3_cc"]; r["G9b"] = r["eceK3_ccE"] < r["eceK3_lr"]; r["G9c"] = r["auroc_ccE"] >= r["auroc_cc"] - 0.01
    rows.append(r)
R = pd.DataFrame(rows); pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
R.to_csv("results/g9_last.csv", index=False)
