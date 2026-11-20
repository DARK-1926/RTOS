import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"CC-plaxity": "#2a78d6", "CC-within (no temporal)": "#eb6834",
       "Toba-pooled (indep)": "#1baf7a", "Logistic regression": "#eda100"}
MK = {"CC-plaxity": "o", "CC-within (no temporal)": "s", "Toba-pooled (indep)": "^", "Logistic regression": "D"}
LAB = {"CC-plaxity": "CC-plaxity (ours)", "CC-within (no temporal)": "CC, within-frame only",
       "Toba-pooled (indep)": "Toba-2026 independence", "Logistic regression": "Logistic regression"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

m = pd.read_csv(f"{OUT}/cc_metrics_by_checkpoint.csv")
m = m[m.cp.astype(str).isin(["0", "1", "2"])].copy(); m["cp"] = m.cp.astype(int)
sel = [("lidar", 50, "Real LiDAR pipeline, D = 50 ms"), ("pilot", 30, "Contention sweep C0–C5, D = 30 ms"),
       ("hclock", 28, "DVFS / clock sessions, D = 28 ms")]
fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
for ax, (ds, D, title) in zip(axs[:3], sel):
    x = m[(m.dataset == ds) & (m.D == D)]
    for meth in COL:
        g = x[x.method == meth].sort_values("cp")
        ax.plot(g.cp, g.auroc, color=COL[meth], marker=MK[meth], lw=2, ms=7,
                mec="#fcfcfb", mew=1.5, label=LAB[meth])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["frame\nrelease", "after N1", "after N2"])
    ax.set_ylim(0.45, 1.02); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
axs[0].set_ylabel("AUROC (validation split)")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")

# panel 4: coupling vs contention level (pilot calib)
lv = ["C0", "C1", "C2", "C3", "C4", "C5"]; rho = [0.14, 0.19, 0.28, 0.47, 0.41, 0.42]
ax = axs[3]
ax.bar(lv, rho, color="#2a78d6", width=0.6)
for i, r in enumerate(rho):
    ax.text(i, r + 0.015, f"{r:.2f}", ha="center", color=INK, fontsize=8)
ax.set_ylim(0, 0.6); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Within-level coupling grows with contention", fontsize=9.5, color=INK, loc="left")
ax.set_ylabel("Spearman ρ (N1 time vs remaining)"); ax.set_xlabel("contention level (calibration split)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cc1_auroc_by_checkpoint.png", dpi=180)

# figure 2: calibration under unseen contention
t = pd.read_csv(f"{OUT}/cc_transfer_exploratory.csv")
t = t[t.test.str.startswith("contention") & (t.cp == 0)]
meths = ["CC-plaxity", "Deadline-conditional LR", "Fixed-D LR", "Toba-pooled (indep)"]
col2 = {"CC-plaxity": "#2a78d6", "Deadline-conditional LR": "#eb6834", "Fixed-D LR": "#eda100", "Toba-pooled (indep)": "#1baf7a"}
cases = [(r.test.split(":")[1], r.D) for r in t[["test", "D"]].drop_duplicates().itertuples()]
fig, ax = plt.subplots(figsize=(8.5, 3.3))
w = 0.2
for j, meth in enumerate(meths):
    vals = [t[(t.test.str.endswith(c)) & (t.D == D) & (t.method == meth)].ece.squeeze() if
            len(t[(t.test.str.endswith(c)) & (t.D == D) & (t.method == meth)]) else np.nan for c, D in cases]
    xs = np.arange(len(cases)) + (j - 1.5) * w
    ax.bar(xs, vals, width=w - 0.02, color=col2[meth], label=meth.replace("Toba-pooled (indep)", "Toba-2026 independence"))
    for xx, v in zip(xs, vals):
        if not np.isnan(v):
            ax.text(xx, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(range(len(cases)))
ax.set_xticklabels([f"{'LiDAR' if c=='lidar' else 'Pilot'}: train low → test heavy\nD = {D} ms" for c, D in cases])
ax.set_ylabel("ECE at frame release (lower = better)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Exploratory: calibration when contention exceeds anything seen in calibration", fontsize=9.5, loc="left", color=INK)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_cc2_contention_extrapolation_ece.png", dpi=180)
print("ok")
