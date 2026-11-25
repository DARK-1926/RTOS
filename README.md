# CC-plaxity: Deadline-Miss Early Detection with Window Guarantees

Reproduction package for the paper *"Deadline Miss Early Detection for DAG Tasks with Cross-Frame State and Window Guarantees Under Intervention"*.

## Layout

| Path | Contents |
|---|---|
| `PAPER/paper.pdf` | Compiled manuscript |
| `PAPER/source/` | LaTeX source (`main.tex`, `refs.bib`, `figs/`) |
| `CC_PLAXITY_STUDY/PROTOCOL_*.md` | Pre-registered protocols and all addenda, frozen before data collection |
| `CC_PLAXITY_STUDY/THEOREMS.md` | Statements and proofs |
| `CC_PLAXITY_STUDY/REPORT.md` | Full study report, including failed rules |
| `CC_PLAXITY_STUDY/gen/*_confirm.py` | Frozen confirmatory analysis per addendum |
| `CC_PLAXITY_STUDY/gen/md5_*.txt` | MD5 freeze records of rules, scripts and data |
| `CC_PLAXITY_STUDY/gen/paper_figures.py` | Regenerates the paper figures into `PAPER/source/figs/` |
| `CC_PLAXITY_STUDY/gen/proofcheck*.c` | Exhaustive independent checks of the theorems |
| `CC_PLAXITY_STUDY/gen/data/` | Raw per-frame logs from the boards (Arduino UNO Q, Raspberry Pi 5) |
| `CC_PLAXITY_STUDY/gen/results/`, `CC_PLAXITY_STUDY/results/` | Analysis outputs |
| `CC_PLAXITY_STUDY/ros2/` | ROS 2 LiDAR pipeline, board run scripts (`uq/` for UNO Q, `pi/` for Pi 5) |

## Key results

All numbers below are produced by the scripts in this repository from the raw logs in `CC_PLAXITY_STUDY/gen/data/`.

**Held:**
- Window bound on late frames (Theorem 4.6, exact form): held in 26/26 closed-loop runs (UNO Q 20/20 over ten seeds, Pi 5 6/6).
- UNO Q fallback (level 2) in the closed loop: 0 of 2,544 frames late over ten seeds (95% upper bound 0.12%).
- UNO Q fallback stress test (Addendum S, rule S12, limit 0.5%): 4 of 11,929 frames late at 27 ms (0.034%).
- Exhaustive proof checks: 0 violations (`proofcheck.c`, `proofcheck_cost.c`).
- Inline monitor cost: below 200 µs at the 99th percentile in every UNO Q replication run.

**Failed (pre-registered, reported as failed):**
- Pi 5 fallback is not timely: 2.3–10.2% of 8,913 stress frames late at 6 ms, so the plain bound held in 0/3 runs at 6 ms and 1/3 at 10 ms.
- The predictor did not rank frames better than logistic regression (2/6 and 3/6).
- Replication comparative rules P5, P7a, P7b, P8 and P11 failed: at the light deadline the monitor did not reliably reduce misses (mean 5.2% vs 5.8% with no monitor and 5.4% with random degradation).

Descriptive only (not a pre-registered rule): at the heavy UNO Q deadline, policies with the guarantee layer averaged a 4.2–4.8% miss rate over ten seeds, compared with 33.3% with no monitor.

## Reproducing the analysis

Requires Python 3.11 and a C compiler.

```bash
pip install -r requirements.txt
```

```bash
cd CC_PLAXITY_STUDY/gen
python p_confirm.py      # Addendum P (closed loop, seeds 231-233)
python s_confirm.py      # Addendum S (replication, seeds 261-267, and UNO Q stress test)
gcc -O2 -o proofcheck proofcheck.c && ./proofcheck
gcc -O2 -o proofcheck_cost proofcheck_cost.c && ./proofcheck_cost
python paper_figures.py  # writes PAPER/source/figs/
```

Each `*_confirm.py` prints every pre-registered rule with PASS/FAIL. Failed rules are reported as failed in the paper.

## Rebuilding the paper

```bash
cd PAPER/source
tectonic main.tex
```

## Not included

The preprocessed LiDAR point clouds (`gen/lidar_xyz*`) and the offline conda packages used on the Pi are omitted for size; `CC_PLAXITY_STUDY/gen/lidar_prep*.py` regenerates the point clouds from the public dataset.
