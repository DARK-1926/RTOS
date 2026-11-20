"""EXPLORATORY (designed after seeing primary results): does CC-plaxity's structure transfer
(a) to unseen deadlines and (b) to unseen (stronger) contention levels, versus a learned model?"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from cc_plaxity import (load_lidar, load_pilot, load_hclock, prep, CCPlaxity, IndepPlaxity, CP_DONE,
                        feats, auroc, ece)

dd, out = sys.argv[1], sys.argv[2]


from cc_transfer_lr import LRdeadline


def score(name, sc, y, tag, rows, D):
    for k in [0, 1, 2]:
        rows.append(dict(test=tag, method=name, D=D, cp=k, auroc=auroc(sc[k], y),
                         brier=np.mean((sc[k] - y) ** 2), ece=ece(sc[k], y), base=y.mean()))


rows = []
# (a) deadline transfer on lidar and pilot: train LR on two deadlines, test on held-out ones
for ds, loader, Dtr, Dte in [("lidar", load_lidar, [35, 80], [45, 50, 65]), ("pilot", load_pilot, [28, 36], [30, 32, 34]),
                             ("hclock", load_hclock, [26, 34], [28, 30, 32])]:
    o, nom = loader(dd); o = prep(o)
    cal = o[o.split == "calib"].reset_index(drop=True); val = o[o.split == "val"].reset_index(drop=True)
    cc = CCPlaxity(cal, nom); lrd = LRdeadline(cal, Dtr)
    for D in Dte:
        y = (val.e2e > D).values.astype(int)
        score("CC-plaxity", cc.predict(val, D), y, f"deadline-transfer:{ds}", rows, D)
        score("Deadline-conditional LR", lrd.predict(val, D), y, f"deadline-transfer:{ds}", rows, D)

# (b) contention extrapolation: fit on low-contention regimes, test on unseen heavier ones
for ds, loader, train_regs, D_list in [("pilot", load_pilot, ["C0", "C1", "C2"], [30, 35]),
                                       ("lidar", load_lidar, ["idle", "moderate"], [50, 80])]:
    o, nom = loader(dd); o = prep(o)
    cal = o[(o.split == "calib") & o.regime.isin(train_regs)].reset_index(drop=True)
    val = o[(o.split == "val") & ~o.regime.isin(train_regs)].reset_index(drop=True)
    cc = CCPlaxity(cal, nom)
    tp = IndepPlaxity(cal)
    lrd = LRdeadline(cal, sorted(set(np.quantile(cal.e2e, [0.7, 0.8, 0.9, 0.95]).round(1))))
    for D in D_list:
        y = (val.e2e > D).values.astype(int)
        tag = f"contention-extrapolation:{ds}"
        score("CC-plaxity", cc.predict(val, D), y, tag, rows, D)
        score("Toba-pooled (indep)", tp.predict(val, D), y, tag, rows, D)
        score("Deadline-conditional LR", lrd.predict(val, D), y, tag, rows, D)
        # fixed-D LR trained on low-contention calib (may be degenerate if no misses)
        yc = (cal.e2e > D).values.astype(int)
        if 5 < yc.sum() < len(yc) - 5:
            med = cal[["e2e", "n1", "n2"]].median()
            sc = {k: LogisticRegression(max_iter=3000).fit(feats(cal, k, med), yc).predict_proba(feats(val, k, med))[:, 1]
                  for k in CP_DONE}
            score("Fixed-D LR", sc, y, tag, rows, D)

R = pd.DataFrame(rows)
R.to_csv(f"{out}/cc_transfer_exploratory.csv", index=False)
pd.set_option("display.width", 250)
print(R.pivot_table(index=["test", "D", "method"], columns="cp", values=["auroc", "brier", "ece"]).round(3).to_string())
