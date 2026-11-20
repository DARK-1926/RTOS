"""Figure for Addendum D (G10): python g10_figure.py -> results/fig_g10_hardening.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("results/g10_openloop.csv"); C = pd.read_csv("results/g10_closedloop.csv"); O = pd.read_csv("results/g10_overhead.csv")
X = pd.read_csv("results/ext_existing.csv"); X = X[X["mode"] == "standard"]
fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
allr = pd.concat([X.assign(set="earlier 36 runs"), R.assign(set="fresh G10 runs")])
for i, (m, lab) in enumerate([("cc", "CC-plaxity"), ("ccsc", "CC-SC"), ("ewma", "EWMA"), ("ar1", "AR(1)"), ("lriso", "LR-iso")]):
    for j, (st, g) in enumerate(allr.groupby("set", sort=False)):
        ax[0].scatter(np.full(len(g), i * 2 + j * 0.6) + np.random.uniform(-.15, .15, len(g)), g[f"auroc_{m}"], s=12,
                      color=f"C{j}", label=st if i == 0 else None)
ax[0].axhline(0.5, color="k", lw=0.8, ls="--"); ax[0].text(0, 0.51, "Toba-2026 (constant at release)", fontsize=7)
ax[0].set_xticks([i * 2 + .3 for i in range(5)]); ax[0].set_xticklabels(["CC", "CC-SC", "EWMA", "AR(1)", "LR-iso"])
ax[0].set(title="Release-time AUROC: all history methods beat Toba;\nCC is not better than simple EWMA", ylabel="AUROC at frame release"); ax[0].legend(fontsize=7)
mk = {"none": "o", "toba": "s", "cc": "^", "ccsc": "D", "naive": "x"}
for p, g in C.groupby("policy"):
    for c, gg in g.groupby("cond"):
        ax[1].scatter(gg.degraded, gg.miss, marker=mk[p], color="C0" if c == "G10a" else "C1", s=40,
                      label=f"{p} ({c})")
ax[1].set(xlabel="fraction of frames degraded", ylabel="deadline-miss rate (val)",
          title="Closed loop (G10): Toba-eq thresholds degrade almost every frame;\nCC/CC-SC degrade selectively")
ax[1].legend(fontsize=6, ncol=2)
t = O[O.monitor.isin(["us_cc", "us_ccsc_extra", "us_naive", "us_toba", "us_total"])]
pv = t.groupby(["cond", "monitor"])["median"].median().unstack(0)
p9 = t.groupby(["cond", "monitor"])["p99"].max().unstack(0)
xx = np.arange(len(pv))
for j, c in enumerate(pv.columns):
    ax[2].bar(xx + j * 0.38, pv[c], 0.38, label=f"{c} median ({'UNO Q' if c == 'G10a' else 'laptop'})", color=f"C{j}")
    ax[2].scatter(xx + j * 0.38, p9[c], color="k", marker="_", s=200)
ax[2].axhline(500, color="r", ls="--", lw=0.8); ax[2].text(0, 520, "D5 rule: p99 < 500 us (1% of 50 ms)", fontsize=7, color="r")
ax[2].set_xticks(xx + .19); ax[2].set_xticklabels([m.replace("us_", "") for m in pv.index])
ax[2].set(ylabel="us per frame", title="Inline monitor cost in C (bars = median, ticks = worst p99)"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/fig_g10_hardening.png", dpi=150); print("ok")
