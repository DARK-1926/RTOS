"""Figure R11 (closed loop v3 on the Raspberry Pi 5, Addendum Q) for COMPLETE_REPORT. python report_figures_r11.py"""
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


R = pd.read_csv("results/q_confirm_runs.csv")
order = ["none", "kp", "kpmn", "kpm", "kpmwh", "kpmd", "toba", "random"]
lab = {"none": "No action", "kp": "Old monitor", "kpmn": "Fixed update only (ours)", "kpm": "Fixed monitor (ours)",
       "kpmwh": "+ window rule, unwarned", "kpmd": "+ window rule, all late + L2",
       "toba": "Toba", "random": "Random, same budget"}
names = [lab[p] for p in order]; ours = {lab[p] for p in ["kpmn", "kpm", "kpmwh", "kpmd"]}
f, ax = plt.subplots(2, 4, figsize=(10.5, 5.6), gridspec_kw={"wspace": 0.3, "hspace": 0.5})
for row, L in enumerate(["heavy", "light"]):
    m = R[R.load == L].groupby("policy").mean(numeric_only=True)
    t = f"{L.capitalize()} (D = {int(m.D.iloc[0])} ms): "
    bars(ax[row, 0], names, [100 * m.loc[p, "miss"] for p in order], ours, t + "missed (%)", "{:.1f}")
    bars(ax[row, 1], names, [100 * m.loc[p, "degraded"] for p in order], ours, "Frames degraded (%)", "{:.1f}")
    bars(ax[row, 2], names, [R[(R.load == L) & (R.policy == p)].maxwin_late.max() for p in order], ours,
         "Worst 20-window, all late (max)", "{:.0f}", line=(4, "bound = 4"))
    bars(ax[row, 3], names, [R[(R.load == L) & (R.policy == p)].maxwin_late_nl2.max() for p in order], ours,
         "Same, late frames not at L2 (max)", "{:.0f}", line=(4, "bound = 4"))
    for a in ax[row, 1:]:
        a.set_yticklabels([])
f.suptitle("Closed loop v3 on the Raspberry Pi 5, live ROS 2, 3 seeds per load (Addendum Q). Blue = ours; L2 = level-2 fallback, late 5.9% here",
           x=0.01, ha="left", fontsize=10.5, y=0.99)
f.savefig(OUT + "R11_closed_loop_v3_pi.png", dpi=170, bbox_inches="tight"); plt.close(f)
print("ok")
