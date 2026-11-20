# Protocol: Contention-Coupled Plaxity (CC-plaxity)

Frozen 2026-11-06, before any validation-split metric was computed.
Only calibration-split rows were inspected while designing this (Spearman prefix→remaining checks).

## Hypothesis
Toba, Baruah & Azumi (TECS 2026, DOI 10.1145/3808694) compute plaxity by convolving node
execution-time PMFs under statistical independence (their §2.2 explicitly flags this as a limitation,
and §7 lists "lightweight online monitoring ... distribution updates at runtime" as future work).
At a single checkpoint, any plaxity score that depends only on elapsed time e_k is a monotone function
of e_k, so independence-based plaxity cannot rank frames better than e_k itself; it can only be (mis)calibrated.

H1: A one-factor model, where every node's time is scaled by a shared latent contention factor S_t,
filtered across frames by a sticky HMM and updated within the frame by completed nodes, gives
(a) better-calibrated miss probabilities (Brier, ECE) and (b) better discrimination (AUROC) at early
checkpoints than independence plaxity, under a single fixed deadline with unknown, changing contention.
(c) As a result, it raises earlier alarms (more lead time) at matched recall.

H0 / kill: if CC-plaxity does not beat the strongest simple baseline (previous-frame persistence or
logistic regression on the same inputs) on AUROC at checkpoint 0/1, the latent-factor structure adds nothing
beyond a learned model, and we will report that.

## Model
log d_i = beta_i * sigma + eta_i, with S = exp(sigma), eta_i ~ idle-calibration log-duration KDE of node i.
Nodes: N1, N2, F = max(N3, N4) (empirical super-node, same for all methods), N5, OVH = e2e - structural sum.
beta_i: slope of log(regime median / idle median) of node i vs that of e2e across calib regimes.
Remaining time given sigma: R(sigma) = sum_j exp(beta_j sigma) * Y_j over idle-calib joint samples (offline table).
Prior pi(sigma): calib-pooled per-frame MAP sigma histogram (smoothed).
Temporal: S_t = S_{t-1} with prob rho, else redraw from pi. rho is fitted by calib likelihood on a grid.
History is only taken from the immediately preceding frame of the same split chunk; otherwise reset to pi.

## Checkpoints
cp0 = frame release (nothing of the current frame observed), cp1 = after N1, cp2 = after N2, cp3 = after fork.
P_miss(cp_k) = sum_sigma post_k(sigma) * P(R_k(sigma) > D - e_k).

## Baselines (all calibrated on calib only)
B1 Toba-pooled: independent convolution of pooled-calib node PMFs (super-node F), P(R > D - e_k).
B2 Toba-idle: same, using idle-calib PMFs (nominal offline profiling).
B3 Toba-oracle: per-regime PMFs with the true regime label (reference, not deployable).
B4 Persistence: empirical P(miss | previous frame's e2e bin), plus e_k (logistic on log prev_e2e, log e_k).
B5 Logistic regression: features [log e_k, log prev_e2e, log prev_d1, log prev_d2] fitted per checkpoint/deadline.
CC-within: CC-plaxity without temporal filtering (prefix only), to isolate the two information sources.

## Data / splits
Primary: multinode_real_lidar_traces.csv, calib -> val (test split is spent and is not used, even as history).
Deadlines fixed across regimes: D in {35, 50, 80} ms. Evaluation set = all val regimes pooled; per-regime breakdown reported.
Replication 1: pilot_finer_contention_traces.csv (C0..C5), first 60% of frames per level = calib, rest = val; D in {30, 35}.
Replication 2: h_clock_trials.csv, sessions 0-5 = calib, 6-9 = val; D in {28, 32}.
Uncertainty: block bootstrap (blocks of 50 consecutive val frames), 1000 resamples, 95% CI on paired differences.

## Metrics
Per checkpoint: AUROC, Brier, ECE (10 bins).
Pipeline: alarm at the first checkpoint whose P_miss exceeds theta; theta chosen on calib to hit 90% recall;
report val recall, FPR, mean lead time (D - alarm time, TP only), and alarm-checkpoint histogram.
Also Toba-2026 linear variable thresholds (th_min=0.5 -> th_max=0.99 over cp0..cp3), without tuning, for each score.

## Deviations / corrections recorded after the primary run (2026-11-06)
1. KDE evaluation was replaced by a tabulated KDE with linear interpolation (speed only; same estimator).
2. H1(b) as worded ("better discrimination at early checkpoints" from within-frame coupling) is false by
   construction at cp1: a within-frame-only score is monotone in N1's time, as is independence plaxity, so
   AUROC is identical (verified: Spearman 0.99-1.00 between the two scores). All cp0/cp1 AUROC gains come from
   the cross-frame (temporal) term. The within-frame term changes calibration only. The board protocol H2 was
   rewritten as a Brier claim before any board data exists.
3. The transfer tests (cc_transfer.py) were designed after seeing primary results and are EXPLORATORY.

## Board deviations (UNO Q, recorded 2026-11-06 while Arm B was still running, before any Arm B metric was computed)
4. Arm A (frozen script, board default `schedutil` cpufreq governor, 5 seeds) ran exactly as frozen
   (script md5 256e15d85cb8651d01eae4562f9591d7). DVFS confound found from live means during the run: with K = 0 the
   pipeline's ~30% duty cycle lets schedutil drop the clock (cores idle at 1.3-1.4 GHz, down to 300 MHz), so
   K0 frames are the SLOWEST (mean e2e ~60 ms, node 1 7.1 ms) while K1/K2 hogs hold the cores at 2.016 GHz
   (e2e ~26/30 ms, node 1 2.6 ms). The protocol's premise "slowdown increases with K" is false in Arm A, and
   CC-plaxity's idle reference regime (K0) is a low-clock state. Nearly all Arm A misses are K0 frames; the unseen
   K3 frames have 0-3 misses per seed, so H3 is UNINFORMATIVE in Arm A, whatever its pass/fail outcome.
   Arm A is reported as-is as the "shipping configuration" arm.
5. Arm B: the frozen script is run unchanged (same seeds, same md5), with the clock held at max for the whole run.
   Setting the `performance` governor needs root (sudo password), which was not available to the operator, so the
   clock is held instead by four SCHED_IDLE busy-loop processes (`spin.py`, one pinned per core). SCHED_IDLE tasks run
   only when no normal task is runnable, so they keep schedutil at 2.016 GHz without taking CPU from the pipeline
   or hogs. Verified before the run: 1.3-1.4 GHz idle vs 2.016 GHz on all cores with the spinners. Frequencies are logged
   every 5 s to `armB/freq_log.txt`. Residual risk: small wake-up latency or shared-resource effects of the
   spinners. They are ALU-only Python loops that stay in L1, so memory-bandwidth interference is expected to be negligible.
   Early Arm B means are monotone in K (K0 25.2, K1 27.8, K2 30.0 ms), so the protocol premise holds in Arm B.
   H1-H3 and the kill rule are applied to Arm B with the frozen analysis code, unchanged. Arm B is the confirmatory arm for H3.
