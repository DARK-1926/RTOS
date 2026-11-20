"""Figures for the complete report. Run from gen/ after report_curves.py (and g12_analysis.py for R8).
    python report_figures.py   -> ../../CC_PLAXITY_PAPER_PACKAGE/figures/R*.png
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "../../CC_PLAXITY_PAPER_PACKAGE/figures"; os.makedirs(OUT, exist_ok=True)
M = ["Toba-eq", "CC", "EWMA-CP", "LR-iso", "KP"]
COL = {"KP": "#2a78d6", "Toba-eq": "#eb6834", "EWMA-CP": "#1baf7a", "CC": "#eda100", "LR-iso": "#e87ba4"}
MK = {"KP": "o", "Toba-eq": "s", "EWMA-CP": "^", "CC": "D", "LR-iso": "v"}
LBL = {"KP": "KP (ours)", "Toba-eq": "Toba 2026", "EWMA-CP": "EWMA (simple history)", "CC": "CC-plaxity (ours, v1)", "LR-iso": "Logistic reg. + isotonic"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "lines.linewidth": 2, "lines.solid_capstyle": "round"})


def title(ax, s):
    ax.set_title(s, loc="left", color=INK, fontsize=10)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}", dpi=170); plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- R1 system overview
def r1():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
    def box(x, y, w, h, t, fc="#eef3fb", ec=COL["KP"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5, color=INK)
    box(0.2, 1.6, 1.5, 0.8, "N1 sensor\nload + crop")
    box(2.3, 1.6, 1.5, 0.8, "N2 voxel\n0.2 m grid")
    box(4.4, 2.3, 1.5, 0.8, "N3 ground\nobstacle grid"); box(4.4, 0.9, 1.5, 0.8, "N4 BEV\nheight map")
    box(6.5, 1.6, 1.5, 0.8, "N5 fuse\nfree sectors")
    for (a, b) in [((1.7, 2.0), (2.3, 2.0)), ((3.8, 2.0), (4.4, 2.7)), ((3.8, 2.0), (4.4, 1.3)), ((5.9, 2.7), (6.5, 2.0)), ((5.9, 1.3), (6.5, 2.0))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    for x, t in [(0.2, "cp0\nrelease"), (2.3, "cp1\nN2 start"), (4.4, "cp2\nfork start"), (6.5, "cp3\nN5 start")]:
        ax.annotate(t, xy=(x, 2.45), xytext=(x - 0.05, 3.25), fontsize=7.5, color=INK2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, ls=":"))
    ax.add_patch(FancyBboxPatch((0.15, 0.15), 3.75, 0.55, boxstyle="round,pad=0.02", fc="#fdf0ea", ec=COL["Toba-eq"], lw=1))
    ax.text(2.02, 0.42, "actionable window: an alarm here can still halve the work", ha="center", fontsize=7.5, color=INK)
    box(8.5, 2.2, 2.3, 1.1, "Monitor (KP + ACI)\nP(miss) at cp0..cp3\nalarm if > θ_t", fc="#eef3fb")
    box(8.5, 0.6, 2.3, 1.1, "Contention (shared,\npersistent): memory hogs or\nAutoware reference system", fc="#f3f2ee", ec=INK2)
    ax.annotate("", xy=(8.0, 2.0), xytext=(8.5, 2.7), arrowprops=dict(arrowstyle="->", color=COL["KP"], lw=1.2))
    ax.annotate("", xy=(7.3, 1.55), xytext=(8.5, 1.15), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0, ls="--"))
    title(ax, "R1  The LiDAR DAG, the four checkpoints, the monitor and the contention source")
    save(fig, "R1_system_overview.png")


# ---------------------------------------------------------------- R2 release-time blindness
def r2(S):
    order = S[S.method == "KP"].groupby("cond").auc_rel.mean().sort_values().index.tolist(); y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axvline(0.5, color=COL["Toba-eq"], lw=1.5, ls="--"); ax.text(0.507, -0.4, "Toba 2026 = 0.5 exactly\n(Theorem 1)", fontsize=8, color=INK, va="bottom")
    for j, m in enumerate(["CC", "EWMA-CP", "KP"]):
        v = S[S.method == m].groupby("cond").auc_rel.mean().reindex(order)
        ax.scatter(v, y + (j - 1) * 0.18, s=34, marker=MK[m], color=COL[m], edgecolor=SURF, linewidth=1.5, label=LBL[m], zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlim(0.45, 1.0); ax.set_xlabel("AUROC at frame release (before any node runs)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, loc="lower right")
    title(ax, "R2  At frame release Toba cannot rank frames; any cross-frame state can")
    save(fig, "R2_release_blindness.png")


# ---------------------------------------------------------------- R3 at Toba's own thresholds
def r3(A):
    g = A.groupby("method")[["fpr", "precision", "recall", "lead_ms", "actionable_recall"]].mean().reindex(M)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (c, lab) in zip(axs, [("fpr", "False-positive rate (lower is better)"), ("precision", "Precision (higher is better)"),
                                  ("recall", "Recall (higher is better)"), ("actionable_recall", "Misses warned by N2 start")]):
        x = np.arange(len(M)); ax.bar(x, g[c], color=[COL[m] for m in M], width=0.7, edgecolor=SURF, linewidth=2)
        for i, v in enumerate(g[c]):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([LBL[m].split(" (")[0] for m in M], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_title(lab, fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle(f"R3  Every method judged at Toba's recommended job-count thresholds (th 0.8 to 1.0), mean of {A.run.nunique()} hardware runs",
                 x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, "R3_at_toba_thresholds.png")


# ---------------------------------------------------------------- R4 the trade-off curves (FPR vs precision / recall / lead)
def r4(C, tag, sub):
    g = C.groupby(["method", "fpr_target"])[["fpr", "precision", "recall", "lead_ms", "actionable_recall"]].mean().reset_index()
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (c, lab) in zip(axs, [("precision", "Precision"), ("recall", "Recall (any alarm before the deadline)"),
                                  ("actionable_recall", "Actionable recall (warned by N2 start)"), ("lead_ms", "Mean lead time of true alarms (ms)")]):
        for m in M:
            d = g[g.method == m]
            ax.plot(d.fpr, d[c], color=COL[m], marker=MK[m], ms=5, mec=SURF, mew=1.2, label=LBL[m])
        ax.set_xlabel("Realized false-positive rate (threshold swept)"); ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.grid(color=GRID); ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=5, loc="lower center", fontsize=8.5)
    fig.suptitle(f"R4{sub}  Trade-off curves, same alarm rule for every method ({tag})", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/R4{sub}_tradeoff_curves.png", dpi=170); plt.close(fig); print("saved R4")


# ---------------------------------------------------------------- R5 the guarantee
def r5(AA, G):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    a = pd.concat([AA[["run", "method", "alpha", "aci_unwarned", "static_unwarned", "aci_fpr", "aci_bound"]],
                   G[G.target == "actionable"][["seed", "method", "alpha", "aci_unwarned", "static_unwarned", "aci_fpr", "aci_bound"]]
                   .rename(columns={"seed": "run"})])
    ax = axs[0]
    for j, (col, lab, c) in enumerate([("static_unwarned", "fixed threshold tuned on calibration", INK2), ("aci_unwarned", "ACI (ours)", COL["KP"])]):
        for k, al in enumerate([0.02, 0.05]):
            v = a[a.alpha == al][col].values; xx = k * 2 + j * 0.6 + np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
            ax.scatter(xx, v, s=8, color=c, alpha=0.6, label=lab if k == 0 else None)
        ax.hlines([0.02, 0.05], [-0.4, 1.6], [1.0, 3.0], color=COL["Toba-eq"], lw=1.5)
    ax.set_xticks([0.3, 2.3]); ax.set_xticklabels(["target α = 0.02", "target α = 0.05"])
    ax.set_ylabel("Unwarned actionable miss rate (per frame)"); ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    title(ax, f"R5a  ACI hits its target; fixed thresholds drift ({len(a) // 1} method-run cases)")
    ax = axs[1]
    p = a.pivot_table(index=["run", "alpha"], columns="method", values="aci_fpr").reset_index()
    for al, mk in [(0.02, "o"), (0.05, "^")]:
        d = p[p.alpha == al]; ax.scatter(d["Toba-eq"], d["KP"], s=22, marker=mk, color=COL["KP"], edgecolor=SURF, linewidth=1.2, label=f"α = {al}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = int((p.KP < p["Toba-eq"]).sum()); ax.text(0.55, 0.08, f"KP needs fewer false alarms\nin {n}/{len(p)} cases", fontsize=8, color=INK)
    ax.set_xlabel("False-positive rate needed with Toba's score"); ax.set_ylabel("False-positive rate needed with KP")
    ax.legend(frameon=False, loc="upper left", fontsize=8); title(ax, "R5b  Price of the guarantee: KP vs Toba")
    save(fig, "R5_guarantee.png")


# ---------------------------------------------------------------- R6 overhead
def r6(g12=None):
    o = pd.read_csv("results/g10_overhead.csv"); o = o[o.cond == "G10a"]
    st = lambda m: (o[o.monitor == m]["median"].median(), o[o.monitor == m]["p99"].max())   # median of run medians, worst run p99
    t, n, c = st("us_toba"), st("us_naive"), st("us_cc")
    rows = [("Toba-eq (C, board, G10a)", t[0], t[1], COL["Toba-eq"]), ("NaiveScale history (C, board, G10a)", n[0], n[1], COL["EWMA-CP"]),
            ("KP-64 inline in ROS 2 (pilot, K3)", 34.4, 62.9, COL["KP"]), ("KP-256 offline C (E5, K3)", 118, 175, "#9cc0ec"),
            ("CC-plaxity (C, board, G10a)", c[0], c[1], COL["CC"])]
    if g12 is not None:
        k = g12[g12.policy == "kp"]; rows[2] = ("KP-64 inline in ROS 2 (G12, worst K3 p99)", k.us_kp_med.median(), k.us_kp_p99_K3.max(), COL["KP"])
    fig, ax = plt.subplots(figsize=(8.5, 3.4)); y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.6, edgecolor=SURF, linewidth=2)
    ax.scatter([r[2] for r in rows], y, marker="|", s=200, color=INK, zorder=3)
    for i, r in enumerate(rows):
        ax.text(r[2] * 1.15, i, f"{r[1]:.0f} µs median, {r[2]:.0f} µs p99", va="center", fontsize=8, color=INK)
    ax.axvline(100, color=INK2, ls=":", lw=1); ax.text(102, len(rows) - 0.45, "100 µs rule", fontsize=7.5, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(2, 20000); ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Monitor cost per frame on the Cortex-A53 (µs, log scale; bar = median, tick = p99)")
    title(ax, "R6  Monitor overhead per frame (period 50–100 ms)")
    save(fig, "R6_overhead.png")


# ---------------------------------------------------------------- R7 closed loop (G12)
def r7(R):
    pol = ["none", "random", "toba", "kp"]; lab = {"none": "no action", "random": "random (same budget)", "toba": "Toba 2026 alarms", "kp": "KP + ACI (ours)"}
    pc = {"none": "#c3c2b7", "random": INK2, "toba": COL["Toba-eq"], "kp": COL["KP"]}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (c, t) in zip(axs, [("miss", "Deadline-miss rate (lower is better)"), ("miss_K3", "Miss rate under unseen K3 contention"),
                                ("degraded", "Frames degraded (lower is better)")]):
        for j, p in enumerate(pol):
            v = R[R.policy == p].sort_values("seed")[c].values; x = np.arange(len(v)) + (j - 1.5) * 0.2
            ax.bar(x, v, width=0.19, color=pc[p], edgecolor=SURF, linewidth=1.5, label=lab[p])
        ax.set_xticks(range(R.seed.nunique())); ax.set_xticklabels([f"seed {s}" for s in sorted(R.seed.unique())])
        ax.set_title(t, fontsize=9, color=INK, loc="left"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.suptitle("R7  Closed loop on ROS 2 (G12): the alarm halves the points while the frame can still be saved", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.savefig(f"{OUT}/R7_closed_loop.png", dpi=170); plt.close(fig); print("saved R7_closed_loop.png")


if __name__ == "__main__":
    S = pd.read_csv("results/sweep_runs.csv")
    G = pd.read_csv("results/g11_sweep.csv").assign(cond="G11 (ROS 2)", run=lambda d: "g11_s" + d.seed.astype(str))
    S = pd.concat([S, G[S.columns.intersection(G.columns)]])
    A = pd.read_csv("results/report_toba_thr.csv"); C = pd.read_csv("results/report_curves.csv")
    r1(); r2(S); r3(A)
    r4(C, f"mean of {C.run.nunique()} hardware runs", "a"); r4(C[C.cond == "G11"], "real ROS 2 + Autoware interference, G11", "b")
    r5(pd.read_csv("results/aci_actionable_runs.csv"), pd.read_csv("results/g11_aci.csv"))
    g12 = pd.read_csv("results/g12_closedloop.csv") if os.path.exists("results/g12_closedloop.csv") else None
    r6(g12)
    if g12 is not None:
        r7(g12)
