"""Addendum E, E1 as registered (actionable AUROC = max(P0, P1), KP vs Toba-eq, block bootstrap 50-frame blocks, 95% CI).
g11_analysis.py computed the CI for release and any-time AUROC but omitted the actionable one; this script adds it
with the same functions and data (disclosed in the report). Also gives KP vs EWMA-CP (E4, descriptive CI)."""
import numpy as np, pandas as pd
from g_analysis import load
from toba_eq_alarms import attach
from sweep_analysis import all_scores
from cc_plaxity import block_boot_diff, auroc
rows = []
for s in [111, 112, 113]:
    fn = f"data/G11/combined/open_s{s}.csv"; cal, val = load(fn); raw = pd.read_csv(fn)
    cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms"); val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int)
    P, *_ = all_scores(cal, val, D); b = np.arange(len(val)) // 50
    act = {m: S[:, :2].max(1) for m, S in P.items()}
    for ref in ["Toba-eq", "EWMA-CP", "CC"]:
        d = block_boot_diff(act["KP"], act[ref], y, b, auroc)
        rows.append(dict(seed=s, comparison=f"KP - {ref}", dAUC_act=d[0], lo=d[1], hi=d[2], ci_above_0=d[1] > 0))
R = pd.DataFrame(rows); R.to_csv("results/g11_e1_actionable_ci.csv", index=False); print(R.round(4).to_string())
