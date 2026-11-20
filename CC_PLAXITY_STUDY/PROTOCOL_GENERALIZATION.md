# Protocol: CC-plaxity generalization study (G-study)

Frozen 2026-11-06, **before any G-study data was collected.** Only pilots were run: ≤150 frames each, seed 999,
files in `gen/pilot/`. They were used solely to check that the scripts run and to set the frame period. Pilot frames are never analyzed.

Pilot facts used for design: LiDAR pipeline on the UNO Q ~79 ms/frame (K0 78, K1 79, K2 83 ms); synthetic on the
laptop ~23–32 ms/frame (K0–K2, hog_mult 4).

## Why
The UNO Q confirmation (REPORT.md) used one board, one synthetic workload and one contention type, and compared against one
learned baseline. This study tests whether the two confirmed claims generalize:
- **H1** release-time AUROC: CC-plaxity > Toba-2026 independence plaxity (CI excludes 0).
- **H3** calibration at release on unseen heavier contention (K3): ECE(CC-plaxity) < ECE(learned baseline).
H2 (within-frame Brier) was weak in the confirmation and is reported descriptively only.

## Conditions (script `gen/cc_general.py`; new seeds; every run 'calib' K∈{0,1,2}, 'val' K∈{0,1,2,3})
| ID | Platform | Workload | Contention | Clock | Seeds | Frames (calib/val) | Period |
|---|---|---|---|---|---|---|---|
| G1 | UNO Q | **real nuScenes LiDAR** (404 scans, 8-beam) | memory hogs ×1 | held at 2.016 GHz (SCHED_IDLE spinners) | 11–15 | 2400/1600 | 150 ms |
| G2 | UNO Q | synthetic (same as confirmation) | **CPU-only hogs** ×1 | held | 21–23 | 3000/2000 | 100 ms |
| G3 | **Laptop** Ryzen 7 7735HS, Windows 11, AC, Balanced plan | synthetic | memory hogs ×4 | as shipped | 31–33 | 3000/2000 | 100 ms |
| G4x | UNO Q | real LiDAR | memory hogs ×1 | **schedutil (as shipped)** | 41–42 | 2400/1600 | 150 ms |

## Analysis (frozen)
Primary: `cc_plaxity.run_unoq` **unchanged** (D = calib p85; CC-plaxity with K0 as the nominal regime; Toba independence
plaxity; deadline-conditional LR; block bootstrap, blocks of 50, 1000 resamples). Warm-up: the first 20 frames are dropped (as frozen).

Added baselines (`gen/g_analysis.py`), fit on calib only and evaluated at release (cp0) on unseen-K3 val frames:
- **GBM-deadline**: sklearn HistGradientBoostingClassifier (default hyper-parameters, random_state=0) on exactly the
  LR-deadline features (same deadline set: calib p75/p80/p90/p95).
- **NaiveScale**: P(miss) = fraction of calib e2e values e with e·(prev_e2e / median calib e2e) > D. A no-model
  multiplicative baseline that also extrapolates; it tests whether CC-plaxity's latent-state machinery adds anything beyond "scale by the last frame".

## Hypotheses and pass rules
- **G1 (real workload)**: H1 passes on ≥ 4/5 seeds AND H3 (vs LR-deadline) passes on ≥ 4/5 seeds.
- **G2 (contention type)**: H1 ≥ 2/3 and H3 ≥ 2/3.
- **G3 (second platform)**: H1 ≥ 2/3 and H3 ≥ 2/3.
- **G5 (stronger learned baseline)**: across all G1–G3 runs (11 runs), ECE_K3(CC) < ECE_K3(GBM-deadline) in ≥ 8/11.
- **G6 (vs naive scaling)**: two-sided, no directional prediction. Report the per-run ECE_K3 and release AUROC of CC vs NaiveScale.
  If NaiveScale ≥ CC on both in most runs, the paper must say the latent-state model is not needed beyond multiplicative scaling.
- **G4x (exploratory)**: is mean e2e at K0 > K1 under schedutil for the real workload (DVFS non-monotonicity)? Descriptive only.
- **Secondary (descriptive)**: H1/H3 at D = calib p75 and p95.

Kill / scope rules: if G1 fails, the H3 claim is restricted to synthetic workloads in the paper. If G3 fails, the claim is
restricted to the UNO Q. All results are reported whichever way they come out.

## Known risks, stated in advance
- The pilot shows **weak memory-contention sensitivity** of the pure-Python LiDAR pipeline (K0→K2 only ~+6%). G1 may fail
  H1 or H3 because the hidden state barely moves latency. That would be a true negative about effect size, not a bug.
- On the laptop, Windows scheduling and the 16 hardware threads make contention weaker per hog; hog_mult = 4 was chosen
  from core count before data, not tuned.
- A K3 level whose frames all miss (or none do) makes ECE degenerate. That is reported, not fixed.

## Addendum A (frozen 2026-11-06, after G1 and G3 were analyzed, BEFORE any G2 or G4x metric was computed)
Observation (G1, exploratory): on unseen K3, CC-plaxity under-predicts misses (mean predicted 0.70–0.78 vs actual
0.87–0.91), and NaiveScale is better calibrated on K3 in 5/5 G1 runs, while CC keeps higher AUROC (5/5).
Mechanism: the sticky-HMM transition resets a share (1 − ρ), with ρ = 0.8–0.9 here, of the belief each frame to the population prior π, which is built from
calibration frames only. π has ~0.1% mass beyond the calibration σ support, so the predictive distribution is pulled back toward familiar
contention exactly when contention is novel.

**Variant CC-R (one change, fixed a priori, not tuned):** the reset target becomes π_R = (1 − ε)·π + ε·Uniform(GRID),
ε = 0.05. Everything else (β, KDEs, ρ selected by calib likelihood with π_R, remaining-time tables) is unchanged.
Sensitivity ε ∈ {0.01, 0.10} is reported descriptively.

**G7 (confirmatory, evaluated on G2 + G4x runs only = 5 runs; these were not used to design CC-R):**
 (a) ECE_K3(CC-R) < ECE_K3(CC) in ≥ 4/5 runs;
 (b) cost check: val ECE on K0–K2 frames (in-distribution) of CC-R is not worse than CC by more than 0.02 in ≥ 4/5 runs;
 (c) two-sided report: ECE_K3(CC-R) vs NaiveScale.
G1/G3/Arm-B results for CC-R are reported as exploratory (the design saw G1).

## Addendum B: compiled workload + real performance governor (frozen 2026-11-06, before any G8 or Arm C data)
Pilots (seed 999, 400 frames, never analyzed) were used only to set the period: the C pipeline takes ~16.5 ms/frame on the UNO Q
(K0 14.4 → K2 17.4 ms) and ~9.8 ms on the laptop (K0 7.7 → K2 10.5 ms).
Source: `gen/lidar_c.c` (md5 a84e00cba933d98b30681654b4bd9986); board build: Debian gcc 14.2 (user-space install in ~/tc), `-O2`;
laptop build: MinGW-w64 gcc 16.1, `-O2 -static`. **Full-resolution** nuScenes scans (32 rings, ~34.7k points), N3‖N4 on two threads,
memory hogs are threads (16 MB streaming each).

| ID | Platform | Workload | Hogs | Clock | Seeds | Frames | Period |
|---|---|---|---|---|---|---|---|
| G8a | UNO Q | C real LiDAR (full-res) | mem ×1 | **performance governor** (set by the user as root; spinners only if it is not set) | 51–55 | 3000/2000 | 50 ms |
| G8b | Laptop | C real LiDAR (full-res) | mem ×4 | Windows Balanced, AC | 61–63 | 3000/2000 | 50 ms |
| Arm C | UNO Q | frozen `unoq_cc_confirm.py` (md5 256e15d8…) | as frozen | **performance governor** | 1–5 (same seeds as Arms A/B) | as frozen | as frozen |

Analysis is unchanged: `cc_plaxity.run_unoq` + `gen/g_analysis.py` (GBM, NaiveScale), D = calib p85, secondary p75/p95.
- **G8 (compiled workload):** H1 ≥ 4/5 (G8a) and ≥ 2/3 (G8b); H3 vs LR ≥ 4/5 and ≥ 2/3; CC beats GBM on K3 ECE in ≥ 6/8.
- **G8-naive (directional test of the open question):** G1 suggested NaiveScale out-calibrates CC on unseen K3 for *real-workload* timing.
  If that explanation is right, NaiveScale ECE_K3 < CC ECE_K3 in ≥ 3/5 G8a runs. If it fails, the real-workload explanation is weakened.
- **Arm C (spinner validity):** H1 ≥ 4/5 and H3 ≥ 4/5 (same kill rule as the confirmation). Per-seed paired differences vs Arm B
  (same seeds) are reported descriptively. If Arm C fails where Arm B passed, the spinner method is suspect and the paper says so.

## Addendum C (frozen 2026-11-06, after G8b (61–63) and G8a seeds 51–52 were analyzed, BEFORE any metric on G8a seeds 53–55)
Observation: on the compiled pipeline, H3 fails badly on the board (51–52: ECE_K3 0.59/0.38 vs LR 0.06/0.10), and NaiveScale
out-ranks CC (AUROC 0.95 vs 0.73 on 51). Diagnosis (51–52, exploratory): the contention slowdown is NOT a single shared factor.
At unseen K3 (5 runnable threads on 4 cores), n1 saturates (1.23× vs 1.21× at K2) while n2 (2.06×) and overhead (2.82×) jump,
and CC's cross-frame update ignores n5 and ovh. The narrow-idle-KDE hypothesis was checked and rejected (≤1% of frames have flat likelihood).
**Variant CC-E** (`gen/cc_fullevid.py`): the cross-frame update uses all completed nodes of the previous frame (n1, n2, F, n5, ovh);
π and ρ are re-estimated with that evidence. Nothing else changes. No tuning.
**G9 (confirmatory, G8a seeds 53–55 only):** (a) ECE_K3(CC-E) < ECE_K3(CC) in ≥ 2/3; (b) ECE_K3(CC-E) < ECE_K3(LR) in ≥ 2/3;
(c) release AUROC(CC-E) ≥ AUROC(CC) − 0.01 in ≥ 2/3. The original G8 hypotheses are evaluated as registered, with CC, whatever G9 shows.

## Addendum D: reviewer hardening (frozen 2026-11-07 08:05 IST, BEFORE any G10 data was collected)
Context: a review asked for (i) stronger simple baselines, (ii) a self-checking fallback for the G8 boundary,
(iii) equation-level Toba alarms, (iv) monitor overhead in C on the board, (v) a closed-loop result, and (vi) a check
that calibration and validation do not share scans or scenes.

**Leakage finding (recorded before G10):** in G1, G4x, G8a and G8b, frames cycle over all 404 scans (`idx = frame mod 404`),
so calib and val replay the SAME scans and scenes. Stage-0 and the synthetic workloads have no scene content.
Existing runs are therefore re-analysed post hoc with 3 scene folds (EXPLORATORY, `gen/ext_analysis.py`), and every
new LiDAR run below is scene-disjoint by construction: calib scans come from 7 scenes and val scans from scene-0553,
scene-0796 and scene-1094 only (`gen/idx_calib.txt`, `gen/idx_val.txt`, from `gen/scan_scene_map.csv`).

**New methods (code frozen; md5 in `gen/md5_addendumD.txt`):**
- AR1 and EWMA on log e2e with empirical calibration residuals, plus LR-iso (LR-deadline with isotonic recalibration on
  5-fold out-of-fold calibration predictions). These are the stronger simple baselines (`gen/ext_methods.py`).
- **CC-SC, the self-checking monitor.** For frame t, use NaiveScale instead of CC when ≥ 7 of CC's last 20 release-time PIT values
  u_j = P_CC(e2e_j > observed e2e_j) fall outside [0.05, 0.95]. This is a binomial test with a ~0.2% false-switch rate under calibration.
  The constants were fixed a priori and never tuned. In closed-loop runs, degraded frames are excluded from the PIT history.
- **Equation-level Toba** (`gen/toba_baseline.py`, TECS 2026 Eqs. 8–18), with N6 = measured overhead. It uses th_min 0.8 and th_max 1.0
  (the paper's setting). The primary workload is job count (Eq. 13), which is unit-free. Eq. 14 (mean time) is secondary, because in ms units
  Eq. 18 yields thresholds above th_max when the remaining mean work is < 1 ms, so they are clamped to [th_min, th_max]. All methods
  are alarmed with the same per-node thresholds.
- **`gen/lidar_cl.c`** is the C pipeline of lidar_c.c plus inline monitors (CC, CC-SC, NaiveScale, Toba-eq). All four are
  computed on every frame, so every policy pays the same overhead. It also implements closed-loop degradation: an alarm before N2
  halves the points, and an alarm before N3‖N4 halves the voxels. Monitor cost is logged per frame. Tables are exported by
  `gen/export_tables.py` from the calibration run. Verified on the laptop: C and Python release probabilities agree within
  5e-6, and self-check flags agree 100%.

**Fresh runs:**
| ID | Platform | Workload | Clock | Seeds | Design |
|---|---|---|---|---|---|
| G10a | UNO Q | lidar_cl, full-res, scene-disjoint, mem hogs ×1, 50 ms period | schedutil + SCHED_IDLE spinners (no root; Arm C showed equivalence with the performance governor) | 71–75 | calib 3000 frames (policy none) → tables → 5 val runs of 2000 frames with the same seed, one per policy: none, toba, cc, ccsc, naive |
| G10s | UNO Q | frozen `unoq_cc_confirm.py` (synthetic, md5 256e15d8…) | spinners | 101–103 | as frozen |
| G10b | Laptop | lidar_cl as G10a, hog_mult 4 | as shipped | 91–93 | as G10a (run only while no analysis job loads the laptop) |

The open-loop tests use calib + the `none` validation run (ids offset so history does not cross the gap). Analysis uses the frozen
`run_unoq` metrics plus `gen/ext_analysis.py` (D = calib p85).

**Hypotheses and pass rules:**
- **D-H1** (early warning, scene-disjoint): release AUROC CC > Toba independence, CI > 0, in ≥ 4/5 G10a and ≥ 2/3 G10b.
- **D1** (self-check fixes the G8 boundary, primary): on G10a, ECE_K3(CC-SC) < ECE_K3(CC) in ≥ 4/5, AND AUROC(CC-SC) ≥
  AUROC(CC) − 0.01 in ≥ 4/5.
- **D2** (no cost where CC works): on G10s, ECE_K3(CC-SC) ≤ ECE_K3(CC) + 0.02 in ≥ 2/3, AND ECE_K3(CC-SC) < ECE_K3(LR) in ≥ 2/3.
- **D3** (strong simple baselines, two-sided, wording rule): on all 11 fresh runs, report CC and CC-SC vs AR1, EWMA and LR-iso
  (release AUROC with block-bootstrap CI; K3 ECE). If CC-SC's K3 ECE is not lower than min(AR1, EWMA) in at least half of the runs,
  the paper must state that simple history models capture the advantage over Toba ("the information, not the model").
- **D4** (closed loop, G10a, primary): val miss rate under the CC-SC policy < under the Toba-eq policy in ≥ 4/5 seeds.
  Reported descriptively: degraded-frame fraction, output change (free-sector disagreement from `lidar_cl quality`), and the
  none / cc / naive policies.
- **D5** (overhead): on the G10a board runs, the p99 per-frame total monitor cost (all four monitors) is < 500 µs (1% of the
  period). Median and p99 are reported per monitor.
- **D6** (equation-level Toba alarms, G10a `none` runs): at Toba's job-count thresholds, alarm precision of CC-SC > Toba-eq in ≥ 4/5.
All outcomes are reported whichever way they come out.

## Addendum E: state-conditioned plaxity, guaranteed actionable warnings, ROS 2 (frozen 2026-11-07, BEFORE any G11 data)

**Why.** Round D showed that history (state) gives the gain over Toba-2026 and that the CC model itself is not needed for it.
Round E turns that into a method with a guarantee, and tests it on a real ROS 2 application with real ROS 2 interference.
Exploratory analysis on all 54 existing runs (results/sweep_runs.csv, aci_runs.csv, aci_actionable_runs.csv, info_limit.csv) motivated the
hypotheses below. That analysis is NOT confirmatory and will be labelled exploratory. Of note: the first ACI target
(any alarm before the frame ends) favours methods with late checks, so Toba's N5/N6 checks look cheap. This addendum therefore fixes the
ACTIONABLE target (alarm by N2 start, the last point where degradation helps) as the primary target. That choice was made after seeing
the exploratory data, and it is disclosed here.

**Methods (code frozen, md5 in gen/md5_addendumE.txt).**
- **KP**, Kalman plaxity (`gen/kp_methods.py`, C in `gen/kp_monitor.c`): Toba's independent per-node convolution, applied to
  state-normalised durations, times exp(z_t). z_t is a log-scale state with an EWMA prior over completed frames, updated in-frame by finished nodes.
  It uses 256 quantiles.
- **ACI alarm layer**: alarm at the first checkpoint where P_k(miss) > θ_t. The update is θ ← θ − γ(e − α) with lag 1, γ = 0.01, θ_0 = the static calibration threshold.
  For the actionable target, e = 1{miss and no alarm at checkpoints 0–1}. Deterministic bound:
  |mean(e) − α| ≤ (1 + 4γ)/(γT) + 1/T.
- **Comparators**: Toba-eq (equation-level, `toba_baseline.py`), CC, CC-SC, NaiveCP, EWMA-CP, LR-iso, all exactly as in
  `gen/sweep_analysis.py`.

**G11 (fresh).**
- **Platform:** UNO Q.
- **Pipeline:** the LiDAR DAG as a ROS 2 Jazzy rclcpp application (`ros2/cc_lidar_ros`). It runs 5 nodes in one process on a
  4-thread MultiThreadedExecutor. Messages are PointCloud2 / MultiArray through rmw_fastrtps, with intra-process comms off. Scans are full-resolution and scene-disjoint
  (idx_calib / idx_val). The period is 50 ms.
- **Interference:** K concurrent `autoware_default_singlethreaded` processes (ros-realtime/reference-system @ de8f3bc), each in its own ROS domain.
  Calibration uses K ∈ {0,1,2}; validation uses K ∈ {0,1,2,3}, and K3 is unseen. Dwell times are Exp(mean 8 s) from a seeded RNG. Transition frames (K = −1) are dropped.
- **Clock:** SCHED_IDLE spinners hold the clock, as in G10a.
- **Runs:** seeds 111, 112, 113, each 3000 calibration and 3000 validation frames. D = calibration p85.
- **Format conversion and analysis:** frozen in `gen/g11_analysis.py`.

**Hypotheses (G11, 3 seeds) and pass rules.**
- **E1** (early warning beats Toba): actionable AUROC (max of P_0, P_1) of KP > Toba-eq, block-bootstrap 95% CI > 0, in ≥ 2/3 seeds.
- **E2** (guarantee holds): for every method, seed and α ∈ {0.02, 0.05}, the ACI actionable unwarned-miss rate is within the
  deterministic bound. Any violation counts as a failure (an implementation error).
- **E3** (cheaper guarantee): ACI actionable FPR of KP < Toba-eq at both α in ≥ 2/3 seeds.
- **E4** (simple-baseline wording rule, two-sided): if KP's actionable AUROC is not > EWMA-CP's in ≥ 2/3 seeds, the paper states
  that KP and EWMA-CP are equivalent in detection, and KP's claimed advantages are limited to being a Toba-compatible convolution and per-checkpoint use.
- **E5** (overhead): KP in C on the board, replaying the G11 seed-111 validation stream while K = 3 reference-system instances and the
  spinners run: p99 per-frame cost < 100 µs (0.2% of the period).
- **E6** (information limit, descriptive): report R²_K, AUC*_Gauss and achieved release AUROC for G11, next to the 54 earlier runs.
All outcomes are reported whichever way they come out.

**Addendum E amendment (frozen BEFORE any G11 data; only the seed-999 pilot existed).** Pilot (`ros2/pilot.sh`, 300 frames
per level, clock held): mean e2e was 20.6 / 23.9 / 36.4 ms at K = 0 / 2 / 3, with K3 p95 55 ms and max 82 ms. At a 50 ms period, K3 frames would queue
without bound (and without spinners, K0 alone averaged 49 ms). The G11 period is therefore **100 ms** (`run_g11.sh` changed; md5 re-recorded). Nothing else changes.
The pilot also shows that K ≤ 2 (calibration) contention is weak (×1.16) and K3 is strong (×1.77), so G11 is a hard extrapolation test.

## Addendum F: ROS 2 closed loop with inline KP-64 + ACI, and the overhead fix (frozen 2026-11-07, BEFORE any G12 data)

**Why.** E5 failed (offline KP in C: 175 µs p99 under K3), and no closed loop had been run on ROS 2. Round F puts the monitor *inside*
the ROS 2 pipeline and lets it act.

**Engineering done before freezing, on pilots only (seeds 998/999 with placeholder parameters; never analysed for accuracy):**
- KP uses 64 quantiles instead of 256. On G11 seed 111 the exploratory actionable AUROC was 0.9352 vs 0.9353.
- `erfc` is replaced by a 16,385-point normal-tail lookup table with linear interpolation (error about 1e-7).
- Pilot costs, with the cost measured inline in the live executor:
  - inline with `erfc`: 83 µs median, 119 µs p99 at K0;
  - with the table: 36 µs median, 62–118 µs p99 at K0;
  - with the table, under K3 (400 frames): 34 µs median, 63 µs p99.

The F1 threshold below (100 µs) is the one from E5, unchanged.

**G12 design.**
- **Pipeline:** `ros2/cc_lidar_ros/src/lidar_ros_cl.cpp`: G11's pipeline plus inline monitors at all 4 checkpoints on every frame, under
  every policy.
- **Monitor state:** the KP state is updated only on undegraded frames. ACI uses α = 0.05 and γ = 0.01 on the actionable target.
  Its threshold θ_0 is set from calibration.
- **Degradation:** keep every 2nd point, at N1 when the alarm comes at release, or at N2 start before voxelization when it comes at N2 start.
- **Interference:** identical to G11 (Autoware reference system, K ∈ {0,1,2} in calibration and {0..3} in validation, same seeded schedule
  for every policy of a seed). The period is 100 ms and the clock is held.
- **Seeds 121, 122, 123:**
  - calibration: 3000 frames, policy none;
  - parameters: exported on the laptop (`gen/kp_cl_export.py`: KP-64, θ_0, Toba-eq job-count tables);
  - validation: 2000 frames each for the policies `none`, `kp` (KP + ACI), `toba` (Toba-eq alarm at node 1 or 2 at its thresholds),
    and `random` (degrade at release with p = the kp run's degraded fraction, i.e. the same budget).
- **Scripts:** `ros2/run_g12.sh`, `g12_phase.sh`, `g12_driver.sh`.
- **Analysis:** `gen/g12_analysis.py`. Frames dropped: the first 20, incomplete frames and transition frames.

**Hypotheses and pass rules:**
- **F1** (overhead): inline KP-64 cost per frame (4 checkpoints + end-of-frame update), p99 over K3 frames of the kp run < 100 µs, in 3/3 seeds.
- **F2** (guarantee in closed loop): in the kp run, |unwarned actionable miss rate − 0.05| ≤ the deterministic ACI bound (lag 1), in 3/3.
- **F3** (benefit): miss rate (kp) < miss rate (none), in 3/3.
- **F4** (targeting): miss rate (kp) < miss rate (random, same budget), in ≥ 2/3.
- **F5** (cost vs Toba): degraded fraction (kp) < degraded fraction (toba), in 3/3.
- **Descriptive:**
  - miss rates per K;
  - output change (free-sector disagreement from `data/G10a/quality.csv`, the same scans and kernels);
  - C++ vs Python KP-64 agreement on the none run;
  - the Toba policy's miss rate.

All outcomes are reported whichever way they come out.


## Erratum to Addenda E and F (7 Nov 2026; no frozen text changed)
The ACI guarantee was stated two-sided, as |mean(e) − α| ≤ bound. The proof supports only the upper side:
mean(e) ≤ α + (θ₁ + (L+1)γ)/(γT) + L/T. Reason: θ ≥ −(L+1)γ, because an unwarned miss needs θ ≥ 0. There is no upper limit on θ when
misses are rare, so the rate may fall below α, which is the safe side. The E2/F2 checks are reported under both the registered
two-sided test and the correct one-sided test. Their verdicts are reported as computed.

## Addendum G: contention as OBSERVED state (OS telemetry), real-application interference (frozen 2026-11-07, BEFORE any G13c/G14 data)

**Why.** History-based state (KP) lags at contention switches (bursts of unwarned misses, critique E1) and extrapolates poorly
to unseen contention. The operating system already *observes* contention.

**Design work, on pilot data only.**
- The telemetry pipeline `lidar_ros_tm` was developed on pilot seeds 131–132 (G13pilot) and seed 997 (smoke).
- Telemetry costs:
  - first version, reading `/proc/stat` and per-thread schedstat on the critical path: 1.4–2 ms;
  - final version, a background sampler every 20 ms with the release path copying the snapshot: about 1 µs.
- Exploratory pilot results (2 seeds; `data/G13pilot`, `g13_analysis.py`, `g13_switch.py`):
  - **KP-T (telemetry-only prior)** improved release AUROC in the first 15 frames after a switch (0.776 → 0.831);
  - it improved release ECE on unseen K3 (0.254 → 0.082);
  - actionable AUROC +0.012.
- The history+telemetry prior was not better than telemetry-only, so telemetry-only is primary.
- Late-checkpoint stacking with Toba-eq (fitted on calibration only) was chosen from exploratory data: G11 any-time AUROC 0.999, and no
  cost over 54 earlier runs.

**Method (frozen, `gen/kpt_methods.py`, `gen/g_confirm.py`).** KPT+stack:
- the prior mean of the log-scale state is OLS on telemetry (u = other processes' CPU share since the last release, r = runnable tasks,
  w = own involuntary preemptions);
- 64 quantiles; KP in-frame updates;
- late checkpoints stacked with Toba-eq;
- ACI on the actionable target (α = 0.05 and 0.02, γ = 0.01, lag 1).

Comparators: KP+stack, EWMA-CP+stack, Toba-eq.

**Runs (fresh seeds, board, ROS 2, clock held by SCHED_IDLE + nice-19 spinners, 100 ms period, 3000 + 3000 frames):**
- **G13c**, seeds 141–143: Autoware reference-system interference (as G11).
- **G14**, seeds 151–153: **real-application interference** (`ros2/run_g14.sh`). Calibration K ∈ {0: none, 1: gcc build loop, 2: + gzip of
  the scans}; validation adds **K3: + xz, an interferer type never seen in calibration**.

**Hypotheses (6 runs):**

| ID | Hypothesis | Rule |
|---|---|---|
| G1 | release ECE on unseen K3: KPT < KP | ≥ 5/6 |
| G2 | release AUROC on the first 15 frames after a switch: KPT > KP | ≥ 5/6 |
| G3 | actionable AUROC: KPT ≥ KP − 0.005 (no loss) | ≥ 5/6 |
| G4 | actionable AUROC: KPT > Toba-eq with CI > 0 | 6/6 |
| G5 | any-time AUROC: KPT ≥ Toba-eq − 0.005 | ≥ 5/6 |
| G6 | ACI one-sided guarantee holds at both α | 6/6 |
| G7 | ACI price (FPR, α = 0.05): KPT < Toba-eq | 6/6 |
| G8 | worst 100-frame unwarned rate (α = 0.05): KPT < KP | ≥ 4/6 |
| G9 | critical-path telemetry p99 < 5 µs | 6/6 |

All outcomes are reported whichever way they come out.

## Addendum H: burst guard for unwarned-miss bursts, fresh-seed replication of G, and a new dataset (frozen 2026-11-07, BEFORE any H data)

**Why.** G8 failed (3/6): unwarned misses come in bursts. The exploratory diagnosis on all 8 telemetry runs (`gen/burst_diag.py`) found:
- the bursts sit in the medium-contention regime, not at switches;
- unwarned rate by regime is K0 1.0%, K1 5.8%, K2 10.6%, K3 0.9%;
- 82% of worst-window frames are K2;
- the single ACI threshold (about 0.40) meets the average target by over-covering easy states.

**Exploratory results** (same 8 runs, used to choose the method, NOT evidence):
- `gen/group_aci_explore.py`: one threshold per telemetry group helped little (worst 100-frame window 0.175 → 0.165, 5/8).
- `gen/burst_guard_explore.py`: the burst guard reduced it to 0.120 (8/8 vs ACI) at an FPR cost of +1.6 points.
  - Against plain ACI tuned to the SAME FPR, the guard was better in 7/8 runs (0.120 vs 0.169).
  - So the gain is not just from alarming more.

**Method (frozen, `gen/h_confirm.py`).** Burst-guarded ACI (BG-ACI) on KPT scores at checkpoints 0–1:
- effective threshold θ_eff,t = θ_t − κ·[Σ of the last W known errors − αW]₊, with W = 50 and κ = 0.05;
- θ_t is the ordinary ACI state (γ = 0.01, lag 1), updated with the errors of the effective rule;
- θ₁ comes from `static_theta` on calibration, as before.

The guard only lowers the threshold. When θ_t < 0, every frame alarms and e = 0, so the state's lower barrier is unchanged. The one-sided ACI bound therefore holds unchanged: mean e ≤ α + (θ₁ + (L+1)γ)/(γT) + L/T.

**Part 1: fresh seeds, same setup** (scripts unchanged from G):
- `H13`, seeds 161–163: Autoware interference (`run_g13.sh`).
- `H14`, seeds 171–173: real applications (`run_g14.sh`), with unseen xz in validation.
- Each seed: 3000 calibration + 3000 validation frames.

**Part 2: new dataset AND new modality (drone camera)**, chosen at the user's request: a non-lidar dataset from a UAV.
- **Data.** EuRoC MAV (Burri et al., IJRR 2016), stereo camera on an AscTec Firefly hexacopter.
  - Fetched from a Hugging Face mirror of the official ETH archives (GlowBond/EuRoC_MAV_Dataset; byte sizes identical to the ETH Research Collection).
  - Uses cam0 (752×480 grayscale), every 3rd frame, converted to raw 8-bit images.
  - **Environment-disjoint split:** calibration frames come from MH_04_difficult (machine hall) and validation frames from V1_02_medium (Vicon room 1).
- **Pipeline** `vo_ros_tm`: a 5-node ROS 2 visual-odometry front end with the same DAG and telemetry as `lidar_ros_tm`.
  - N1 camera: load the image, publish sensor_msgs/Image.
  - N2: image pyramid (3 levels) + gradients.
  - Fork:
    - N3: FAST-9 corners + non-max suppression;
    - N4: pyramidal Lucas–Kanade tracking of the previous frame's features.
  - N5 (join): RANSAC homography on the tracks, plus feature replenishment from N3.
  - The CSV format is identical, so the analysis code (`g13_analysis.build`, `g_confirm`, `h_confirm`) is unchanged.
- **Interference.** Real applications as in H14, with the compressors running on the EuRoC images.
  - Calibration K ∈ {0,1,2}; validation adds unseen xz (K3).
- **Seeds.** 181–183, each 3000 calibration + 3000 validation frames.
  - Seed 189 is a development/K0-timing pilot and is never analysed for any hypothesis.
- **Period rule** (no miss labels involved): 100 ms, unless the median K0 e2e of the pilot exceeds 45 ms. In that case it is the smallest multiple of 50 ms that is ≥ 2.2 × that median.
- The pipeline md5 is recorded in `md5_addendumH.txt` after the pilot and BEFORE seed 181.
- Argoverse 2 lidar (log-disjoint) is a possible later Part 3 with the same rules. It is not run unless stated.

**Hypotheses.** Part 1 has n = 6 runs, part 2 has n = 3. Every rule is applied to each part separately.

| ID | Hypothesis | Rule (part 1 / part 2) |
|---|---|---|
| H1 | worst 100-frame unwarned rate: BG-ACI < ACI | ≥ 5/6 / ≥ 2/3 |
| H2 | worst 100-frame unwarned rate: BG-ACI < plain ACI at matched FPR (α grid 0.005–0.08, step 0.0025) | ≥ 4/6 / ≥ 2/3 |
| H3 | the one-sided guarantee holds for BG-ACI at α = 0.05 and 0.02 | 6/6 / 3/3 |
| H4 | BG-ACI FPR < Toba-eq ACI FPR (α = 0.05) | 6/6 / 3/3 |
| H5 | K2 unwarned rate: BG-ACI < ACI | ≥ 4/6 / ≥ 2/3 |
| H6 | replication: G1–G9 rules re-applied on fresh seeds (and on AV2 with n = 3, scaled: ≥ 5/6 → ≥ 2/3, 6/6 → 3/3, ≥ 4/6 → ≥ 2/3) | reported per check |

**Reboots.** A seed that is interrupted is discarded whole: its files are moved to `discarded_*`, never analysed, and the seed is re-run from calibration (`ros2/h_chain.sh`, idempotent).

All outcomes are reported whichever way they come out.

## Addendum I: online-fused prior (KPF), the final method, confirmed on fresh seeds across two pipelines (frozen 2026-11-08, BEFORE any Addendum I data)

**Why.** The first EuRoC drone seed (181) showed that the telemetry-only prior (KPT) is pipeline-specific:
- On drone camera VO, latency follows scene content: lag-1 autocorrelation is 0.70, and history explains 68% of prior variance against 14% for telemetry.
- There KPT lost to plain KP (actionable AUROC 0.792 vs 0.880).
- On lidar, telemetry wins after contention switches and on unseen contention.

**Exploratory on all 15 telemetry runs** (`gen/prior_combo_explore.py`; used for design, NOT evidence):
- Fixed OLS combinations (history + telemetry levels, or + telemetry innovations) beat KP 15/15, but kept only part of KPT's post-switch and unseen-K3 advantages.
- The online fusion KPF:
  - matched KP on drone and KPT on lidar on average;
  - had the lowest mean unseen-K3 calibration error (0.102);
  - cut the worst-case calibration loss against the better single prior from 0.16 to 0.055;
  - beat Toba with CI > 0 in 15/15;
  - was insensitive to the forgetting rate (ρ from 0.8 to 0.99 gives the same metrics).

**Method (frozen, `gen/kpf_methods.py`).** KPF+stack:
- **Prior.** Two experts, H (KP's EWMA history) and T (KPT's telemetry OLS), are fused by inverse recent-MSE weights. Each expert's recent MSE is tracked with exponential forgetting ρ from completed frames.
- **Variance.** The prior variance is the fused prior's own forgetting MSE.
- **ρ.** Chosen on calibration by the prior's Gaussian NLL over {0.5, 0.7, 0.8, 0.9, 0.95}.
- **After the prior.** KP's in-frame updates, then late-checkpoint stacking with Toba-eq, and burst-guarded ACI (Addendum H; W = 50, κ = 0.05, γ = 0.01, lag 1).

**Runs (fresh seeds).** EuRoC drone VO and lidar with real applications, so one pipeline is scene-driven and one contention-driven:
- **EuRoC drone VO**: seeds 182–184, `vo_ros_tm`, 200 ms period (the Part 2 rule).
- **Lidar + real applications**: seeds 191–193, `run_g14.sh`, 100 ms.
- Scripts and binaries are unchanged from Addendum H.
- 3000 + 3000 frames each.

**Hypotheses** (`gen/i_confirm.py`; n = 6, of which 3 are drone):

| ID | Hypothesis | Rule |
|---|---|---|
| I1 | robustness: actionable AUROC KPF ≥ max(KP, KPT) − 0.01 | ≥ 5/6 |
| I2 | actionable AUROC KPF > Toba-eq with block-bootstrap CI > 0 | 6/6 |
| I3 | drone: actionable AUROC KPF > KPT | 3/3 |
| I4 | release ECE on unseen K3: KPF < KP | ≥ 5/6 |
| I5 | BG-ACI one-sided guarantee at α = 0.05 and 0.02 | 6/6 |
| I6 | worst 100-frame unwarned rate: BG-ACI < plain ACI | ≥ 5/6 |
| I7 | BG-ACI false-positive rate: KPF < Toba-eq | 6/6 |

**Reboots.** An interrupted seed is discarded whole and re-run (`ros2/h_chain.sh`, watchdog `ros2/h_watch.sh`).

All outcomes are reported whichever way they come out.

## Addendum J — final method vs standard baselines (written 2026-11-08, BEFORE any baseline result on these runs)

Analysis-only addendum on already-collected data (no new board runs). Disclosure: KPF+stack and Toba-eq results on the
primary runs are already known (Addendum I); no baseline (Budget / LR-all / GBDT-all) has been computed on any run
used below. Rules are fixed here, before the baselines are run; the smoke test on pilot seeds 131-132 checks code only.

Baselines (gen/baselines_j.py; calibration-only, causal, one model per checkpoint, out-of-fold calibration scores for theta_0):
- Budget: classical elapsed-time watchdog, isotonic P(miss | elapsed e_k).
- LR-all: logistic regression on history (prev, EWMA, last-5 mean of log e2e), telemetry (u, log1p r, log1p w),
  completed node durations and elapsed time.
- GBDT-all: gradient-boosted trees on the same features, hyper-parameters by 5-fold time-blocked CV log loss (fixed 8-point grid).
All get the same ACI and burst-guarded ACI (W=50, kappa=0.05).

Primary runs: the 6 Addendum I runs (EuRoC vo 182-184, lidar real apps g14 191-193). Secondary (reported, not tested):
13 earlier runs. Checks (gen/j_confirm.py):
- J1 actionable AUROC KPF+stack > Budget with block-bootstrap CI > 0: 6/6
- J2 actionable AUROC KPF+stack > LR-all: >= 5/6
- J3 actionable AUROC KPF+stack >= GBDT-all - 0.01 (non-inferiority): >= 5/6
- J4 BG-ACI FPR at alpha 0.05: KPF+stack < GBDT-all: >= 5/6
- J5 actionable ECE on unseen K3: KPF+stack < GBDT-all: >= 5/6
- J6 BG-ACI one-sided guarantee (both alpha) holds for every baseline: 18/18
Any failure is reported as such. Code changes after the freeze are allowed only to fix crashes and are logged.

## Addendum K — deadline flexibility without retraining (written 2026-11-08, BEFORE any result)

Question: a deployed monitor is calibrated once, then the deadline changes (mode change, new rate). KP/KPF predict
P(e2e > D) for any D from the same fitted model; learned classifiers are trained for one deadline.
Analysis-only on existing data. Every model is fitted once at D0 = round(q85 of calibration e2e) and is then evaluated,
WITHOUT refitting, at new deadlines D' = round(q75, q80, q90, q95 of calibration e2e).
- KPF (raw, no stack: the Toba stack is fitted to one deadline's labels): evaluated at D'.
- LR-all@D0, GBDT-all@D0 (Addendum J baselines): same scores, labels y' = 1{e2e > D'}.
- Reference only (not a competitor): LR-all retrained at each D'.
Metrics per (run, D'): actionable AUROC (max over checkpoints 0-1) and calibration ECE = mean of ECE at checkpoints 0 and 1.
Primary: the 6 Addendum I runs x 4 deadlines = 24 cases; secondary: the 13 earlier runs (reported only). Checks (gen/k_confirm.py):
- K1 ECE: KPF < LR-all@D0 in >= 20/24
- K2 ECE: KPF < GBDT-all@D0 in >= 20/24
- K3 actionable AUROC: KPF >= LR-all@D0 - 0.01 in >= 20/24
- K4 actionable AUROC: KPF >= retrained LR-all - 0.02 in >= 20/24
Smoke test on pilot 131-132 for code only. Any failure is reported as such.

## Addendum L — second platform: Raspberry Pi 5 (written 2026-11-15, BEFORE any Pi data)

Purpose: answer "one board only". Same experiment as Addenda I/J on a different SoC: Raspberry Pi 5 Model B (4x Cortex-A76,
8 GB, Kali Linux rolling, kernel 6.12.34+rpt-rpi-2712, governor 'performance' as found, active fan), borrowed board.
- Constraints: everything in ~/ccpi (user-space micromamba ROS 2 Jazzy; no root; ROS 2 discovery limited to localhost).
  No clock spinners (governor already 'performance'); CPU clock, SoC temperature and fan speed are logged every 5 s (read-only).
- Pipelines, interferers, schedule RNG, frame counts (3000 + 3000) and periods are unchanged: lidar g14 100 ms, drone vo 200 ms
  (identical periods keep the two boards directly comparable; no miss labels involved). Scripts: ros2/pi/run_pi.sh, pi_chain.sh.
- Pilot seed 199 (300 + 300 frames per pipeline, never analysed) checks the scripts only.
- Fresh seeds: vo 201-203, g14 211-213. Interrupted seeds are discarded whole and re-run.
- Analysis gen/l_confirm.py: per-run code of i_confirm.py and j_confirm.py UNCHANGED (frozen md5s), check rules copied verbatim:
  L-I1..L-I7 (= I1..I7) and L-J1..L-J6 (= J1..J6), same thresholds. I4, J2, J4 failed on the UNO Q; they are kept, not dropped.
- Thermal throttling, if any appears in freq_log, is reported, not used to exclude frames.
All outcomes are reported whichever way they come out.

## Addendum M — deterministic weakly-hard guarantee on unwarned misses (written 2026-11-15, BEFORE any Pi confirmatory data was pulled)

New rule (gen/wh_methods.py, WH-ACI): on top of burst-guarded ACI, if the known unwarned misses among the last K-1 releases
reach m, the next frame is warned unconditionally at release. Theorem (proof in the file): for any score, any latency sequence and
any contention, every window of K consecutive frames contains at most m + lag - 1 unwarned misses (= m here, lag 1), i.e. the
stream of unwarned misses satisfies the weakly-hard constraint (m, K) of Bernat, Burns & Llamosi; the long-run ACI bound is kept.
Exploration (gen/wh_explore.py and an inline variant comparison; the 19 UNO Q runs, all already seen): the bound held in every run;
tight budgets (m ~ alpha K) cost +0.09 to +0.19 FPR; budgets m ~ 4 alpha K cost +0.01 to +0.02 FPR while BG-ACI alone exceeded them
in every run. A ramped-threshold variant was not better and is dropped. Budgets fixed from this: (4,20), (6,50), (10,100), alpha 0.05.
Confirmatory test on the 6 Pi 5 runs of Addendum L (seeds 201-203, 211-213; still running, no data seen), gen/m_confirm.py:
- M1 KPF+stack: worst window <= m in 18/18 (run x budget) (a theorem; any failure means a bug)
- M2 KPF+stack: BG-ACI without the rule exceeds the budget in >= 12/18 (the rule is needed)
- M3 KPF+stack: FPR cost of the rule <= 0.03 in >= 15/18
- M4 KPF+stack: long-run one-sided bound holds in 18/18
- M5 same guarantee, FPR: KPF+stack < Toba-eq in >= 15/18
- M6 the theorem holds for LR-all and Toba-eq scores: 36/36
Caveat known in advance: the Pi runs at low utilisation, so bursts may be milder there (M2 may fail). Reported whichever way.

## Addendum N — trace-driven control case study (written 2026-11-15, BEFORE any Pi confirmatory data was pulled)

Question: what does the warning buy downstream? Simulation driven by the real per-frame outcomes and alarms (gen/control_study.py):
linearised inverted pendulum (w0 = 3 rad/s), LQR at the pipeline period; on-time frame = fresh estimate, warned frame = fast fallback
estimate (noise x10), unwarned miss = actuator holds the previous command. Only outcomes and alarms come from the board; the plant is
simulated (case study, not hardware-in-the-loop). Primary setting (w0 = 3, ratio 10) was fixed before the sensitivity sweep.
Exploration (19 UNO Q runs, all already seen; control_explore.py, control_sweep.py): no monitor crashes 3.4 / 9.4 per 1000 frames
(lidar / drone); KPF+BG has ~10x lower cost, 18-30% below Toba+BG, 38-53% below always-fallback at the primary setting. Known limits from
the sweep: always-fallback wins when the fallback is nearly as good as the fresh estimate and the plant is fast (drone, w0 >= 4, ratio 3);
the (4,20) window rule changes little; the tight (2,20) rule helps only for fast plants with a good fallback and hurts otherwise.
Confirmatory test on the 6 Pi 5 runs (gen/n_confirm.py), primary setting:
- N1 control cost J: KPF+BG < no monitor: 6/6
- N2 J: KPF+BG < Toba+BG: >= 5/6
- N3 J: KPF+BG < always-fallback: >= 5/6
- N4 crashes per 1000: no monitor > KPF+BG: >= 5/6
- N5 the (4,20) window guarantee costs <= 5% of J: >= 5/6
The full w0 x ratio sweep on the Pi runs is reported for every setting (descriptive).

## Addendum O — closed loop v2 on the Pi 5: intervention latch-up and its fix (written 2026-11-15, BEFORE any Pi closed-loop result, pilot included)

Finding that motivates it (exploratory re-reading of the G12 UNO Q runs): the legacy inline monitor updates its contention state only on
undegraded frames. Once it degrades, it stops observing, the state stays high, and it keeps degrading ("intervention latch-up"):
without action it would alarm on 1-3% of K0 frames; in the loop it degraded 59-88% of K0 frames, in runs of up to 665 frames.
G12's random-degradation runs show that degraded frames carry the same contention signal (log e2e relative to the degraded-mode
median rises with K as for undegraded frames; cross-mode lag-1 correlation ~ same-mode). Fix: MODE-AWARE update (kpm).
The guarantees need only the outcomes of unwarned frames, so they remain valid under intervention.

Experiment (borrowed Pi 5, lidar pipeline, real-application interference as g14, 100 ms; ros2/cc_lidar_ros/src/lidar_ros_cl2.cpp,
ros2/pi/run_cl_pi.sh, pi_cl_phase.sh, pi_cl_driver.sh; gen/kp_cl2_export.py). Per seed: calibration none 3000 frames + calibration
random-degradation p=0.5 1500 frames (for ME2E_D) -> export -> validation 3000 frames per policy, same interference schedule:
none (logs what kpm would do), kp (legacy = G12 method), kpm (mode-aware + burst guard), kpmwh (kpm + live weakly-hard rule (4,20),
general-lag form), toba, random (p = degraded fraction of kpm). alpha 0.05, gamma 0.01. Pilot seed 229 (short, code check only).
Fresh seeds 221-223. Frames: first 20, incomplete and transition (K-1) frames dropped, except O7 which uses every released frame in order.
Checks (gen/o_confirm.py):
- O1 latch-up reproduces: legacy kp degraded fraction >= 1.5 x the 'none' would-alarm fraction: >= 2/3
- O2 fix: kpm degrades fewer frames than legacy kp: 3/3
- O3 no latch: kpm degraded fraction <= 1.5 x the 'none' would-alarm fraction: 3/3
- O4 (G12's failed F5 revisited): kpm degrades fewer frames than toba: 3/3
- O5 delivered deadline misses: kpm < none 3/3, and kpm <= random at the same budget >= 2/3
- O6 long-run guarantee in the loop: unwarned-miss rate <= alpha + one-sided bound, for kpm and kpmwh: 6/6
- O7 live weakly-hard guarantee: under kpmwh every window of 20 consecutive released frames has <= 4 unwarned misses: 3/3
- O8 inline monitor cost under kpmwh: p99 < 100 us: 3/3
Known risk: the Pi runs at low utilisation, so latch-up may be weaker than on the UNO Q (O1 may fail). Reported whichever way.

---

## Addendum P (frozen 17 Nov 2026, before any confirmatory data): closed loop v3 on the Arduino UNO Q

**Why.** Addendum O (Pi 5) left five gaps: warned frames still missed the deadline 10–16% of the time (the degraded mode is not a
guaranteed-time fallback, and both guarantees cover unwarned misses only); the load was heavy; the latch-up fix was not isolated from
the burst guard; Toba's apparent instability in the loop was not a frozen test; and the window rule covered only unwarned misses.

**Platform.** Arduino UNO Q (4× Cortex-A53, 2.016 GHz held by SCHED_IDLE spinners as in G14), ROS 2 Jazzy, lidar pipeline
`lidar_ros_cl3` (generated by `ros2/cc_lidar_ros/make_cl3.py` from the frozen `lidar_ros_cl2.cpp`), 100 ms period, G14 real-application
interference (gcc, gzip, xz unseen in calibration, python), K ∈ {0,1,2} in calibration and {0,1,2,3} in validation, exponential dwell
(mean 8 s), seeded schedule. Scripts: `ros2/uq/run_cl_uq.sh`, `uq_cl_phase.sh`, `uq_cl_driver.sh`.

**Seeds.** 231, 232, 233 (never used before). Pilot seed 239 was used only to fix the three choices below and is never analysed for claims.

**Per seed.** Calibration: 3000 frames, policy none; 1500 frames, random level-1 degradation p = 0.5 (gives ME2E_D). Parameters
exported by `gen/kp_cl3_export.py` (identical to the Addendum O exporter at q = 0.85, checked byte for byte). Validation, two loads,
2000 frames per policy, same interference schedule for all policies of a seed:
- **heavy:** deadline D = calibration quantile 0.85 (as in Addendum O);
- **light:** deadline D = round(1.6 × D_heavy) (fixed from the pilot, see below).
Policies: `none`, `kp` (legacy), `kpmn` (mode-aware state, no burst guard), `kpm` (mode-aware + burst guard), `kpmwh` (+ window rule
on unwarned misses), `kpmd` (+ window rule on ALL late frames, enforced by a level-2 fallback), `toba`, `random` (p = degraded fraction
of `kpm` under the same load). Level 1 = keep every 2nd point (as before). Level 2 = keep every **FB2**-th point.

**Choices fixed from the pilot (seed 239) before freezing.**
- FB2 = **8** (rule: smallest tested divisor whose release-degraded frames finished within the pilot's heavy deadline in at
  least 99.5% of frames on the validation schedule with K3; tested 2, 8, 16, 32, 64; 600 frames each, p = 0.5). Pilot: divisor 2 was late in 2.3% of degraded frames (3.1% under K3); 8, 16, 32 and 64 were never late (worst 16.6, 14.2, 14.9, 13.9 ms vs D = 28 ms).
- Light deadline: the planned rule (the calibration quantile at the latency that `none` exceeds 12% of the time in pilot validation)
  gave 44.5 ms = calibration quantile 0.998, too extreme to estimate per seed (the unseen xz interferer is much heavier than
  calibration). Replaced, before any confirmatory data, by **D_light = round(1.6 × D_heavy)**, 1.6 = 44.5 / 28 from the pilot
  (`kp_cl3_export.py ... m1.6`). At this deadline calibration has almost no misses, so θ0 saturates at 1 and the conformal
  threshold must adapt downward online; this is part of what is tested.

**Pilot observations (seed 239, 600 frames per policy, heavy only, never analysed for claims).** The first 60 s of the schedule held
only K3 and K2. `none` missed 54%; every monitor policy and `random` degraded 86–95% of frames; Toba degraded 13% and missed 56%;
`kpmd` used level 2 on 11 frames, none late, and its worst 20-frame window held 4 late frames. Chain, logging and analysis code ran.

**Analysis.** `gen/p_confirm.py`, frozen with this text. Rules (counts over 3 seeds; "both loads" = 6 runs):

| Rule | Statement | Pass |
|---|---|---|
| P1 | Level-2 frames finish on time: late ≤ 0.5% of level-2 frames, `kpmd` | 6/6 |
| P1b | At least 30 level-2 frames in total | yes |
| P2 | Theorem check: worst 20-frame window of late frames not at level 2 ≤ 4, `kpmd` | 6/6 |
| P3 | Worst 20-frame window of all late frames ≤ 4, `kpmd` | ≥ 5/6 |
| P4 | Isolation, heavy: `kpmn` degrades fewer quiet-period (K0) frames than `kp` | 3/3 |
| P5 | `kpmn` does not latch, heavy: K0 degraded ≤ 1.5 × `none` would-alarm in K0 | 3/3 |
| P6 | Light: `kpm` degrades ≤ 25% of frames | 3/3 |
| P7a | Light: `kpm` misses fewer deadlines than `none` | 3/3 |
| P7b | Light: `kpm` misses ≤ `random` at the same budget | ≥ 2/3 |
| P8 | Heavy: range of Toba's degraded fraction across seeds ≥ 0.4, and `kpm`'s range < half of Toba's | yes |
| P9 | Long-run one-sided guarantee, `kpm`, `kpmwh`, `kpmd`, both loads | 18/18 |
| P10 | Inline monitor cost p99 < 200 µs, `kpmd` (100 µs on the Pi 5; the pilot on this slower board gave p99 52–106 µs under K3) | 6/6 |
| P11 | Total late frames: `kpmd` ≤ `kpmwh` | ≥ 4/6 |

"Late" counts frames that never finished. Window checks use every released frame in release order; the other metrics use the
Addendum O frame filter (finished, not in a K transition, first 20 dropped).

**What each outcome means.** P1–P3 pass: the window guarantee holds for **late frames**, not only unwarned ones, and the stability
certificate (`gen/cert_explore.py`) applies to the system as built. P1 fails: the fallback is not reliably on time on this board, and
the certificate stays a simulation result. P4–P5: whether the latch-up fix is the state update itself. P6–P7: whether the monitor is
selective at a realistic load. P8: whether Toba's instability in the loop can be claimed.

## Addendum Q (frozen 18 Nov 2026, before any confirmatory data): closed loop v3 on the Raspberry Pi 5, and fallback timing

**Why.** Addendum P confirmed the all-late window rule (`kpmd`) on the UNO Q, where level 2 was never late. Two questions are left:
1. Does it replicate on a second, faster board?
2. How solid is assumption A (level-2 frames are on time), on which the corollary of Theorem 3 rests?

The Pi has no internet this time. ROS 2 was reinstalled offline from exact package files into `~/ccpi` (no system change).

**Platform and pipeline.**
- Raspberry Pi 5, performance governor (as found), `lidar_ros_cl3` built from the Addendum P source (cpp md5 0e1489458c63…).
- Same interferers and seeded schedule as Addendums L and O. FB_DIV 2, FB2 = 8 (carried over from P, not re-tuned).
- Seeds 241, 242, 243. Per seed: calibration 3000 frames (`none`) + 1500 (`random`).
- Two loads, from calibration: heavy D = q0.85, light D = 1.6 × heavy.
- Validation: 2000 frames per policy and load, for `none`, `kp`, `kpmn`, `kpm`, `kpmwh`, `kpmd`, `toba`, `random` (at `kpm`'s
  degraded fraction).
- Stress runs (fallback timing): every frame at level-2 strength (random p = 1, FB_DIV = 8) under constant contention, via
  `CC_KFIX`. K3: 3000 frames per seed (confirmatory). K4: 1000 frames per seed (exploratory). Late = not finished, or finished
  after the seed's deadline. 95% Clopper–Pearson upper bounds are reported.
- Analysis: `gen/q_confirm.py`, which reuses `p_confirm.run()` unchanged. m = 4, K = 20.

**What the pilot (seed 249, `data/Q_pilot`) showed, and how it shaped the design.** All of this is pilot data and is never used
for claims.
- At the heavy deadline (6 ms), level-2-strength frames were late 1–2% at every divisor from 2 to 32 under the mixed schedule. Under
  constant K3 the figure was 9%. So the residual lateness is OS scheduling jitter, not workload.
- Pinning the interferers to cores 0–2 and the pipeline to core 3 (user-space `taskset`) did not reduce it (30 of 271 late at
  6 ms). Isolation is therefore not part of the design; the option stays in `run_cl3_pi.sh` (`CC_ISO`) and goes unused.
- In the validation run, `kpmd` at heavy load used level 2 on 56 frames, none late; worst 20-window of all late frames = 4.
- The light load (9 ms) degrades nothing: θ0 = 1, so it cannot test the fallback. It is kept as a no-cost check.
- That is why the rules below test assumption A under stress as a measured rate, with a pre-registered *negative* prediction at 6 ms
  (Q4), and keep the exact form of Theorem 3 (Q2) as the claim that does not depend on A.
- Two pilot bugs were fixed before the freeze:
  1. a stale driver process ran phases out of order;
  2. constant-contention runs slept through their 100000 s dwell after the pipeline had finished. The loop now wakes when the
     pipeline exits; seeded runs are unchanged frame by frame.

**Rules** (frozen; m = 4, K = 20; n = 3 seeds).

| Rule | Statement | Needed |
|---|---|---|
| Q1 | Heavy: level-2 frames of `kpmd`, pooled over seeds, late ≤ 2% | yes |
| Q1b | At least 100 level-2 frames in total, heavy `kpmd` | yes |
| Q2 | Theorem 3, exact form: worst 20-window of late non-level-2 frames ≤ 4, `kpmd`, both loads | 6/6 |
| Q3 | Heavy: worst 20-window of ALL late frames ≤ 4, `kpmd` | ≥ 2/3 |
| Q3L | Light: worst 20-window of ALL late frames ≤ 4, `kpmd` | 3/3 |
| Q4 | Prediction, stress K3: level-2 strength is NOT on time at the heavy deadline (late > 0.5%) | ≥ 2/3 |
| Q5 | Heavy: worst window of all late frames, `kpmd` ≤ `kpmwh` | ≥ 2/3 |
| Q6 | Latch-up isolated, heavy: `kpmn` degrades fewer K0 frames than `kp` | 3/3 |
| Q7 | No latch, heavy: `kpmn` K0 degraded ≤ max(1.5 × `none` would-alarm in K0, 0.02) | 3/3 |
| Q8 | Long-run one-sided guarantee, `kpm`, `kpmwh`, `kpmd`, both loads | 18/18 |
| Q9a | Light: `kpm` degrades ≤ 25% of frames | 3/3 |
| Q9b | Light: `kpm` misses ≤ `random` at the same budget | ≥ 2/3 |
| Q10 | Inline monitor cost p99 < 100 µs, `kpmd` | 6/6 |
| Q11 | Stress K3, pooled: level-2 strength late at the light deadline ≤ 1.5% | yes |

**What each outcome means.**
- **Q2** is the theorem itself: a failure means an implementation bug.
- **Q1 / Q3 / Q5** check whether the all-late bound replicates on the Pi in the closed loop.
- **Q4 + Q11** set the boundary of assumption A on this board. The fallback is not on time at 6 ms under sustained heavy
  contention, but is within 1.5% at 9 ms. The paper must then state A as platform- and deadline-dependent and give the measured
  rate, with its bound.
- **Q6–Q10** replicate P's latch-up, cost and selectivity results on a second board.
- Q9 is weak at light load here, because almost nothing is degraded.

## Addendum R (frozen 19 Nov 2026, before any confirmatory data): can an unprivileged scheduling change make the fallback on time?

**Why.** Addendum Q found that level-2 frames are late 2.3–10.2% of the time at 6 ms on the Pi 5 under constant K3. The tail
comes from OS scheduling, not from the amount of work. The draft paper said only that real-time priority or isolation "would likely"
shrink the tail. This round tests the one mechanism available without root: running the background load under SCHED_IDLE
(`chrt -i 0`, set on the interferer processes only; the pipeline is unchanged). It also replicates Addendum Q's stress result on fresh
seeds.

**Design.**
- Pi 5, same binary and data as Addendum Q. Every frame at level-2 strength (random p = 1, CC_FB_DIV = 8), constant K3 (CC_KFIX = 3).
- Seeds 251, 252, 253; 3000 frames per arm and seed.
- Two arms per seed: base (as Addendum Q) and idle (CC_IDLE = 1). The order alternates by seed: 251 base→idle, 252 idle→base,
  253 base→idle.
- Fixed deadlines: 6 ms and 10 ms, Addendum Q's two Pi 5 deadlines (no recalibration).
- Frame filter as `q_confirm.stress()`. Late = not finished, or finished after D. 95% Clopper–Pearson upper bounds.
- Analysis: `gen/r_confirm.py`.

**Pilot (seed 259, 400 frames per arm; never used for claims).**
- 6 ms: base 44/380 late, idle 9/371 late.
- 10 ms: base 2/380, idle 0/371.

**Rules** (frozen).

| Rule | Statement | Needed |
|---|---|---|
| R1 | Replication, base arm: level 2 late at 6 ms > 0.5% | ≥ 2/3 seeds |
| R2 | The idle arm has fewer late level-2 frames at 6 ms than the base arm | 3/3 seeds |
| R3 | Prediction: even with idle priority, A fails at 6 ms (idle arm, pooled late > 0.5%) | yes |
| R4 | Idle arm, 10 ms: pooled late ≤ 0.15% (Addendum Q's base rate) | yes |

**What each outcome means.**
- R2 + R3 pass: an unprivileged change reduces the tail but does not establish A at 6 ms. Real-time priority or a reservation is still
  needed, and the paper says so with numbers instead of "would likely".
- R3 fails: A is supported at 6 ms with SCHED_IDLE background load, and the paper can state that configuration.
- R4 tests whether the same mechanism makes A hold at 10 ms within the measured bound.

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
