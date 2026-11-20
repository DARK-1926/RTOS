"""Figure for the G-study: reads results/g_all_runs.csv and results/g_latency_by_K.csv (from g_analysis.py)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = pd.read_csv("results/g_all_runs.csv"); R = R[R.q == 0.85]
L = pd.read_csv("results/g_latency_by_K.csv")
names = {"G1": "G1 real LiDAR\n(UNO Q, mem)", "G2": "G2 synthetic\n(UNO Q, CPU hogs)", "G3": "G3 synthetic\n(laptop x86)",
         "G4x": "G4x real LiDAR\n(UNO Q, schedutil)", "G8a": "G8a C LiDAR\n(UNO Q, perf gov)",
         "G8b": "G8b C LiDAR\n(laptop x86)"}
conds = [c for c in names if c in set(R.cond)]
fig, ax = plt.subplots(1, 3, figsize=(18, 4.4))
x = np.arange(len(conds)); w = 0.2
for i, (m, lab) in enumerate([("cc", "CC-plaxity"), ("lr", "LR-deadline"), ("gbm", "GBM-deadline"), ("naive", "NaiveScale")]):
    g = R.groupby("cond")[f"eceK3_{m}"]
    ax[0].bar(x + (i - 1.5) * w, g.mean().reindex(conds), w, yerr=g.std().reindex(conds), capsize=2, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([names[c] for c in conds], fontsize=8)
ax[0].set(title="Calibration on UNSEEN heavier contention (K3)", ylabel="ECE at release (lower is better)")
ax[0].legend(fontsize=8)
for c in conds:
    v = R[R.cond == c].H1_auroc_diff.values
    ax[1].scatter(np.full(len(v), conds.index(c)), v, s=30)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(x); ax[1].set_xticklabels([names[c] for c in conds], fontsize=8)
ax[1].set(title="H1: release AUROC gain over Toba-2026 independence", ylabel="AUROC(CC) - AUROC(indep)")
for c in conds:
    g = L[L.cond == c].groupby("K").mean_e2e.mean()
    base = g.iloc[0]
    ax[2].plot(range(len(g)), g.values / base, "o-", label=names[c].replace("\n", " "))
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["K0", "K1", "K2", "K3 (unseen)"])
ax[2].set(title="Latency vs hidden contention (normalised to K0)", ylabel="mean e2e / mean e2e at K0")
ax[2].axhline(1, color="k", lw=0.6); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/fig_g1_generalization.png", dpi=150)
print("wrote results/fig_g1_generalization.png")
