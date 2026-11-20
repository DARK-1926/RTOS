# CC-plaxity: Deadline-Miss Early Detection with Window Guarantees

Reproduction package for the paper *"Deadline Miss Early Detection for DAG Tasks that Remains Correct When the System Acts on Its Warnings"* (TECS submission).

## Layout

| Path | Contents |
|---|---|
| `PAPER/CC_PLAXITY_paper_TECS.pdf` | Compiled manuscript |
| `PAPER/tecs/` | LaTeX source (`main.tex`, `refs.bib`, `figs/`) |
| `CC_PLAXITY_STUDY/PROTOCOL_*.md` | Pre-registered protocols and all addenda, frozen before data collection |
| `CC_PLAXITY_STUDY/THEOREMS.md` | Statements and proofs |
| `CC_PLAXITY_STUDY/REPORT.md` | Full study report, including failed rules |
| `CC_PLAXITY_STUDY/gen/*_confirm.py` | Frozen confirmatory analysis per addendum |
| `CC_PLAXITY_STUDY/gen/md5_*.txt` | MD5 freeze records of rules, scripts and data |
| `CC_PLAXITY_STUDY/gen/paper_figures.py` | Regenerates the paper figures into `PAPER/tecs/figs/` |
| `CC_PLAXITY_STUDY/gen/proofcheck*.c` | Exhaustive independent checks of the theorems |
| `CC_PLAXITY_STUDY/gen/data/` | Raw per-frame logs from the boards (Arduino UNO Q, Raspberry Pi 5) |
| `CC_PLAXITY_STUDY/gen/results/`, `CC_PLAXITY_STUDY/results/` | Analysis outputs |
| `CC_PLAXITY_STUDY/ros2/` | ROS 2 LiDAR pipeline, board run scripts (`uq/` for UNO Q, `pi/` for Pi 5) |

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
python paper_figures.py  # writes PAPER/tecs/figs/
```

Each `*_confirm.py` prints every pre-registered rule with PASS/FAIL. Failed rules are reported as failed in the paper.

## Rebuilding the paper

```bash
cd PAPER/tecs
tectonic main.tex
```

## Not included

The preprocessed LiDAR point clouds (`gen/lidar_xyz*`) and the offline conda packages used on the Pi are omitted for size; `CC_PLAXITY_STUDY/gen/lidar_prep*.py` regenerates the point clouds from the public dataset.
