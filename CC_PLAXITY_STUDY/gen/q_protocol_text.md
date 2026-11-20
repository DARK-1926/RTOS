
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
