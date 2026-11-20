"""Round E figure (exploratory 54 runs + G11 when present). Run from gen/: python e_figure.py -> results/fig_e_round.png"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"KP": "#2a78d6", "Toba-eq": "#eb6834", "EWMA-CP": "#1baf7a", "CC": "#eda100"}
MK = {"KP": "o", "Toba-eq": "s", "EWMA-CP": "^", "CC": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

S = pd.read_csv("results/sweep_runs.csv")
A = pd.read_csv("results/aci_actionable_runs.csv")
L = pd.read_csv("results/info_limit_all.csv") if os.path.exists("results/info_limit_all.csv") else pd.read_csv("results/info_limit.csv")
g11 = os.path.exists("results/g11_sweep.csv")
if g11:
    G = pd.read_csv("results/g11_sweep.csv").assign(cond="G11 (ROS 2)", run=lambda d: "g11_s" + d.seed.astype(str))
    S = pd.concat([S, G[S.columns.intersection(G.columns)]])
    GA = pd.read_csv("results/g11_aci.csv"); GA = GA[GA.target == "actionable"].assign(run=lambda d: "g11_s" + d.seed.astype(str))
    A = pd.concat([A, GA[["run", "method", "alpha", "aci_fpr", "aci_unwarned"]]])

fig, ax = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
# (a) actionable AUROC per condition
order = S[S.method == "KP"].groupby("cond").auc_act.mean().sort_values().index.tolist()
y = np.arange(len(order))
for j, m in enumerate(["Toba-eq", "CC", "EWMA-CP", "KP"]):
    v = S[S.method == m].groupby("cond").auc_act.mean().reindex(order)
    ax[0].scatter(v.values, y + (j - 1.5) * 0.14, s=26, marker=MK[m], color=COL[m], label=m, zorder=3,
                  edgecolor="white", linewidth=0.6)
ax[0].set_yticks(y); ax[0].set_yticklabels(order); ax[0].set_xlabel("Actionable AUROC (alarm by N2 start)")
ax[0].grid(axis="x", color=GRID, lw=0.8); ax[0].set_axisbelow(True)
ax[0].legend(frameon=False, loc="lower right", fontsize=8)
ax[0].set_title("a  Early warning, threshold-free", loc="left", color=INK, fontsize=10)
# (b) price of the guarantee
P = A[A.alpha == 0.05].pivot_table(index="run", columns="method", values="aci_fpr")
ax[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
ax[1].scatter(P["Toba-eq"], P["KP"], s=22, color=COL["KP"], edgecolor="white", linewidth=0.6, zorder=3)
n = int((P.KP < P["Toba-eq"]).sum())
ax[1].text(0.62, 0.08, f"KP cheaper\nin {n}/{len(P)} runs", color=INK, fontsize=8)
ax[1].set_xlabel("False-alarm rate, Toba-eq + ACI"); ax[1].set_ylabel("False-alarm rate, KP + ACI")
ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
ax[1].set_title("b  Guaranteed 5% unwarned misses", loc="left", color=INK, fontsize=10)
# (c) information limit
ax[2].plot([0.5, 1], [0.5, 1], color=INK2, lw=1, ls="--")
ax[2].scatter(L.AUC_star_gauss, L.KP, s=22, color=COL["KP"], edgecolor="white", linewidth=0.6, zorder=3, label="KP")
ax[2].scatter(L.AUC_star_gauss, L["Toba-eq"], s=22, marker="s", color=COL["Toba-eq"], edgecolor="white", linewidth=0.6,
              zorder=3, label="Toba-eq")
ax[2].set_xlabel("Gaussian AUC* from R² of the regime label"); ax[2].set_ylabel("Achieved release AUROC")
ax[2].set_xlim(0.5, 1); ax[2].set_ylim(0.45, 1); ax[2].legend(frameon=False, loc="upper left", fontsize=8)
ax[2].set_title("c  Contention R² predicts release AUROC", loc="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig("results/fig_e_round.png", dpi=180)
print("saved", "with G11" if g11 else "without G11")
