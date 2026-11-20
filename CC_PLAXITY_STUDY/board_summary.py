"""Summarise the two UNO Q arms (written after Arm A, before Arm B finished; descriptive only).

Inputs : results_armA_schedutil/unoq_confirmatory_results.csv, results_armB_spin/unoq_confirmatory_results.csv
         board_armA_schedutil/*.csv, board_armB_spin/*.csv
Outputs: results/board_two_arm_summary.csv, results/board_miss_by_K.csv, results/fig_cc3_board_two_arms.png
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ARMS = {"A: schedutil (shipping DVFS)": ("board_armA_schedutil", "results_armA_schedutil"),
        "B: clock held at 2.016 GHz": ("board_armB_spin", "results_armB_spin")}

rows, miss = [], []
for arm, (bd, rd) in ARMS.items():
    R = pd.read_csv(f"{rd}/unoq_confirmatory_results.csv")
    for h in ("H1", "H2", "H3"):
        rows.append(dict(arm=arm, hypothesis=h, seeds=len(R), passed=int(R[f"{h}_pass"].sum())))
    rows.append(dict(arm=arm, hypothesis="kill_rule_triggered", seeds=len(R), passed=int((~R.H3_pass).sum() >= 3)))
    for fn in sorted(glob.glob(f"{bd}/*.csv")):
        d = pd.read_csv(fn).iloc[20:]                         # same warm-up drop as the frozen analysis
        D = float(np.round(d[d.split == "calib"].end_to_end_ms.quantile(0.85)))
        v = d[d.split == "val"]
        for k, g in v.groupby("regime"):
            miss.append(dict(arm=arm, file=os.path.basename(fn), D=D, K=k, n=len(g),
                             mean_e2e=g.end_to_end_ms.mean(), mean_n1=g.dur_node1_ms.mean(),
                             miss_rate=(g.end_to_end_ms > D).mean()))
S = pd.DataFrame(rows); M = pd.DataFrame(miss)
os.makedirs("results", exist_ok=True)
S.to_csv("results/board_two_arm_summary.csv", index=False)
M.to_csv("results/board_miss_by_K.csv", index=False)
print(S.to_string(index=False)); print()
print(M.groupby(["arm", "K"])[["mean_e2e", "mean_n1", "miss_rate"]].mean().round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for i, (arm, _) in enumerate(ARMS.items()):
    g = M[M.arm == arm].groupby("K")
    ax[0].plot(g.mean_e2e.mean().index, g.mean_e2e.mean().values, "o-", label=arm)
    ax[1].plot(g.miss_rate.mean().index, g.miss_rate.mean().values, "o-", label=arm)
ax[0].set(title="Mean end-to-end latency (val)", xlabel="hidden contention level", ylabel="ms")
ax[1].set(title="Deadline-miss rate (D = calib p85)", xlabel="hidden contention level", ylabel="miss rate")
ax[0].legend(fontsize=8)
w = 0.38
for i, (arm, (_, rd)) in enumerate(ARMS.items()):
    R = pd.read_csv(f"{rd}/unoq_confirmatory_results.csv"); x = np.arange(len(R))
    ax[2].bar(x + (i - 0.5) * w, R.H3_ece_lr - R.H3_ece_cc, w, label=arm)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set(title="H3: ECE(LR) - ECE(CC) on unseen K3", xlabel="seed", ylabel="> 0 favours CC-plaxity")
ax[2].set_xticks(np.arange(5)); ax[2].set_xticklabels([str(s) for s in range(1, 6)])
plt.tight_layout(); plt.savefig("results/fig_cc3_board_two_arms.png", dpi=150)
print("wrote results/fig_cc3_board_two_arms.png")
