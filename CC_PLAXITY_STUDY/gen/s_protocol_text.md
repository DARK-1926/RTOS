
---

## Addendum S (frozen 19 Nov 2026, before any confirmatory data): UNO Q replication on 7 fresh seeds + UNO Q fallback stress

**Why.** Reviewer-facing weaknesses of the paper: three seeds per closed-loop configuration, and no constant-contention stress test of
level 2 on the UNO Q (Table 6, "not run"). The Pi 5 is unreachable, so this addendum uses the UNO Q only.

**(a) Replication.** Seeds 261–267 (never used before). Identical platform, code, schedule generator, calibration (3000 + 1500 frames),
exporter, deadlines rule (heavy q = 0.85, light = round(1.6 × D_heavy)), 8 policies, 2000 frames per policy, FB2 = 8, as Addendum P.
Analysis: the frozen `p_confirm.py` rules P1–P11 evaluated on the 7 new seeds with the same formulas (n = 7: "all" = 7/7 or 14/14;
P3 ≥ 13/14; P7b ≥ 6/7; P11 ≥ 12/14; P9 42/42). No P rule is changed. A pooled 10-seed table (P + S) is reported descriptively only.

**(b) Fallback stress.** Seeds 261, 262, 263, 4000 releases each, every frame at level-2 strength (random p = 1 with CC_FB_DIV = 8),
constant contention K3 (env CC_KFIX = 3), same frame filter as the Pi stress (drop K-1 rows, then the first 20). Run before (a).
- **S12:** pooled late frames at 27 ms ≤ 0.5% of retained frames. Prediction: PASS (closed-loop level-2 worst 20.1 ms in Addendum P).
- Descriptive: late at 43 ms, Clopper–Pearson 95% upper bounds, worst 20-window.

**Script changes (transfer and scheduling only, no pipeline change).** `run_cl_uq.sh`: CC_KFIX constant schedule; the dwell sleep
wakes when the pipeline ends (as the Pi script). `uq_cl_phase.sh`: new phase `stress`. Driver `uq_s_driver.sh`. Analysis `s_confirm.py`
(imports the frozen `p_confirm.py`). Smoke-tested on copied Addendum P files only (seed label 299; deleted, never used for claims).

**What each outcome means.** S12 PASS: assumption A is supported on the UNO Q also under constant worst-case contention, and Table 6's
"not run" becomes a measured rate. S12 FAIL: A is not supported on the UNO Q under constant K3 and the paper says so. P-rule results on
7 seeds replace "3 seeds" with "10 seeds" in the paper only for rules that pass; failed rules are reported as failed.
