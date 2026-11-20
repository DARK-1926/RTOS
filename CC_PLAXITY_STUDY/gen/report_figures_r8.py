"""Figures R8 (closed loop v2, Addendum O) and R9 (control case study, Addendum N) for COMPLETE_REPORT. python report_figures_r8.py"""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "../../CC_PLAXITY_PAPER_PACKAGE/figures/"
INK, MUTE, OURS, REST = "#1f2328", "#6b7280", "#2f6fb5", "#b8bcc4"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK, "axes.labelcolor": INK})


def bars(ax, names, vals, ours, title, fmt, line=None):
    c = [OURS if n in ours else REST for n in names]
    b = ax.barh(range(len(names)), vals, color=c, height=0.62)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_title(title, loc="left", fontsize=9.5)
    top = max(vals) if max(vals) > 0 else 1
    for r, v in zip(b, vals):
        ax.text(v + 0.02 * top, r.get_y() + r.get_height() / 2, fmt.format(v), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.22); ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False)
    if line is not None:
        ax.axvline(line[0], color=INK, lw=1, ls=(0, (3, 2))); ax.text(line[0], len(names) - 0.42, " " + line[1], fontsize=8, color=INK, va="center")


R = pd.read_csv("results/o_confirm_runs.csv")
m = R.groupby("policy").mean(numeric_only=True)
order = ["none", "kp", "kpm", "kpmwh", "toba", "random"]
lab = {"none": "No action", "kp": "Old monitor", "kpm": "Fixed monitor (ours)", "kpmwh": "Fixed + window rule (ours)", "toba": "Toba", "random": "Random, same budget"}
names = [lab[p] for p in order]; ours = {lab["kpm"], lab["kpmwh"]}
f, ax = plt.subplots(1, 4, figsize=(13, 2.9), gridspec_kw={"wspace": 0.75})
bars(ax[0], names, [100 * m.loc[p, "miss"] for p in order], ours, "Missed deadlines (%)", "{:.1f}")
bars(ax[1], names, [100 * m.loc[p, "degraded"] for p in order], ours, "Frames degraded (%)", "{:.1f}")
bars(ax[2], names, [100 * m.loc[p, "degraded_K0"] for p in order], ours, "Degraded when nothing is wrong (%)", "{:.1f}")
bars(ax[3], names, [m.loc[p, "max_win20"] for p in order], ours, "Worst 20-frame window (unwarned misses)", "{:.1f}", line=(4, "bound = 4"))
for a in ax[1:]:
    a.set_yticklabels([])
f.suptitle("Closed loop on Raspberry Pi 5, live ROS 2, mean of 3 seeds (Addendum O)", x=0.01, ha="left", fontsize=10.5, y=1.04)
f.savefig(OUT + "R8_closed_loop_v2.png", dpi=170, bbox_inches="tight"); plt.close(f)

N = pd.read_csv("results/n_confirm_runs.csv")
print(N.columns.tolist()); print(N.head(3).to_string())

order = ["none", "always-fallback", "Toba+BG", "LR+BG", "KPF+BG", "KPF+WH(4,20)", "oracle"]
lab = {"none": "No monitor", "always-fallback": "Always fallback", "Toba+BG": "Toba + guarantee", "LR+BG": "Logistic regression + guarantee",
       "KPF+BG": "Ours + guarantee", "KPF+WH(4,20)": "Ours + window rule", "oracle": "Oracle (perfect warning)"}
ours = {lab["KPF+BG"], lab["KPF+WH(4,20)"]}
f, ax = plt.subplots(1, 2, figsize=(9.5, 3.1), gridspec_kw={"wspace": 0.15})
for a, (h, t) in zip(ax, [(0.1, "Lidar pipeline, 100 ms period"), (0.2, "Drone pipeline, 200 ms period")]):
    g = N[N.h == h].groupby("policy").J.mean()
    bars(a, [lab[p] for p in order], [g[p] for p in order], ours, t, "{:.3f}")
ax[1].set_yticklabels([])
f.suptitle("Trace-driven control case study (simulated pendulum, real Pi 5 timing traces, Addendum N): control cost J, lower is better", x=0.01, ha="left", fontsize=10.5, y=1.04)
f.savefig(OUT + "R9_control_study.png", dpi=170, bbox_inches="tight"); plt.close(f)
