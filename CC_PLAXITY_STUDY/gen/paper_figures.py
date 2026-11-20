"""Figures for the manuscript (vector PDF, column width). Run from CC_PLAXITY_STUDY/gen: python paper_figures.py -> PAPER/tecs/figs/"""
import numpy as np, pandas as pd, matplotlib, glob
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from o_confirm import params
import os
OUT = os.environ.get("FIG_OUT", "../../PAPER/tecs/figs/")
INK, MUTE, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C = {"Toba-eq": "#d9622b", "KP": "#2f6fb5", "EWMA-CP": "#1a9e77", "LR-iso": "#8a63b8"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "lines.linewidth": 1.3, "legend.fontsize": 6.8})
W = 3.45

# ---- Fig. release blindness
S = pd.read_csv("results/sweep_runs.csv")
G = pd.read_csv("results/g11_sweep.csv").assign(cond="G11 (ROS 2)", run=lambda d: "g11_s" + d.seed.astype(str))
S = pd.concat([S, G[S.columns.intersection(G.columns)]])
S["cond"] = S.cond.replace({"stage0_lidar": "S0-lidar", "stage0_pilot": "S0-pilot", "stage0_hclock": "S0-hclock"})
order = S[S.method == "KP"].groupby("cond").auc_rel.mean().sort_values().index.tolist(); y = np.arange(len(order))
f, ax = plt.subplots(figsize=(W, 2.5))
ax.axvline(0.5, color=C["Toba-eq"], lw=1.2, ls="--")
ax.text(0.51, len(order) - 1.2, "Toba (any PMFs):\nexactly 0.5", color=C["Toba-eq"], fontsize=6.8, va="top")
for j, (m, lab, mk) in enumerate([("EWMA-CP", "EWMA history", "^"), ("KP", "KP (ours)", "o")]):
    v = S[S.method == m].groupby("cond").auc_rel.mean().reindex(order)
    ax.scatter(v, y + (j - .5) * .25, s=14, marker=mk, color=C[m], label=lab, zorder=3, lw=0)
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.5); ax.set_xlim(0.45, 1.0); ax.set_xlabel("AUROC at frame release (before any node runs)")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right")
f.savefig(OUT + "release_auroc.pdf", bbox_inches="tight"); plt.close(f)

# ---- Fig. trade-off curves (actionable recall, lead time)
Cu = pd.read_csv("results/report_curves.csv")
g = Cu.groupby(["method", "fpr_target"])[["fpr", "lead_ms", "actionable_recall"]].mean().reset_index()
f, axs = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"wspace": 0.55})
lab = {"Toba-eq": "Toba", "KP": "KP (ours)", "EWMA-CP": "EWMA", "LR-iso": "LR"}
for ax, (c, yl) in zip(axs, [("actionable_recall", "Misses warned by N2 start"), ("lead_ms", "Lead time of true alarms (ms)")]):
    for m in ["Toba-eq", "LR-iso", "EWMA-CP", "KP"]:
        d = g[g.method == m]; ax.plot(d.fpr, d[c], color=C[m], label=lab[m], lw=1.6 if m == "KP" else 1.1)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel(yl); ax.grid(color=GRID); ax.set_axisbelow(True); ax.set_xlim(0, 0.6)
axs[0].legend(frameon=False, loc="lower right")
f.savefig(OUT + "tradeoff.pdf", bbox_inches="tight"); plt.close(f)

# ---- Fig. closed-loop worst windows (P: UNO Q, Q: Pi 5)
pol = ["none", "toba", "random", "kp", "kpm", "kpmwh", "kpmd"]
pl = {"none": "No action", "toba": "Toba", "random": "Random (same budget)", "kp": "Old monitor (latches)", "kpm": "Fixed monitor",
      "kpmwh": "+ window rule, unwarned", "kpmd": "+ window rule, all late (W)"}
sets = [("p", "heavy", "UNO Q 27 ms"), ("p", "light", "UNO Q 43 ms"), ("q", "heavy", "Pi 5 6 ms"), ("q", "light", "Pi 5 10 ms")]
f, axs = plt.subplots(1, 4, figsize=(7.1, 1.85), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, (r, L, t) in zip(axs, sets):
    R = pd.read_csv(f"results/{r}_confirm_runs.csv"); R = R[R.load == L]
    for i, p in enumerate(pol):
        v = R[R.policy == p]
        col = "#2f6fb5" if p in ("kpm", "kpmwh", "kpmd") else "#9aa0a8"
        ax.scatter(v.maxwin_late, [i] * len(v), s=11, color=col, zorder=3, lw=0, alpha=.85)
        if p == "kpmd":
            ax.scatter(v.maxwin_late_nl2, [i + .32] * len(v), s=11, marker="x", color="#d9622b", zorder=3, lw=.9)
    ax.axvline(4, color=INK, lw=.8, ls=(0, (3, 2))); ax.set_title(t, fontsize=7.5); ax.set_xlim(0, 21)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.set_xlabel("worst late frames / 20")
axs[0].set_yticks(range(len(pol))); axs[0].set_yticklabels([pl[p] for p in pol]); axs[0].invert_yaxis()
axs[3].text(11, 5.6, "x = late frames\nnot at level 2", color="#d9622b", fontsize=6.2, va="center")
f.savefig(OUT + "closed_loop_windows.pdf", bbox_inches="tight"); plt.close(f)

# ---- Fig. fallback timing: CCDF of level-2-strength latency
f, ax = plt.subplots(figsize=(W, 1.9))
def ccdf(x, **k):
    x = np.sort(x); ax.step(x, 1 - np.arange(len(x)) / len(x), where="post", **k)
x = []
for fn in glob.glob("data/Q/cl_s24?_val_stressK3.csv"):
    a = pd.read_csv(fn); a = a[a.regime != "K-1"].iloc[20:]; x.append(a.e5.where(a.done == 1, 99).values)
ccdf(np.concatenate(x), color="#d9622b", label="Pi 5: every frame at level 2,\nconstant K3 (8,913 frames)")
y = []
for fn in glob.glob("data/P/cl_s23?_val_*_kpmd.csv"):
    a = pd.read_csv(fn); y.append(a.e5[(a.degraded == 3) & (a.done == 1)].values)
ccdf(np.concatenate(y), color="#2f6fb5", label="UNO Q: level-2 frames\nin the loop (790)")
for d, t in [(6, "Pi D=6"), (10, "Pi D=10"), (27, "UNO Q D=27")]:
    ax.axvline(d, color=MUTE, lw=.7, ls=":"); ax.text(d, 1.6, t, fontsize=5.8, color=MUTE, ha="center", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2); ax.set_xlim(0.8, 40)
ax.set_xlabel("End-to-end latency (ms)"); ax.set_ylabel("P(latency > x)"); ax.grid(color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.42), fontsize=6.0)
f.savefig(OUT + "fallback_ccdf.pdf", bbox_inches="tight"); plt.close(f)
print("ok")
