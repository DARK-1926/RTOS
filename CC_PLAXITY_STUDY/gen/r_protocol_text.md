
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
