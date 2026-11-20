"""One-off: fold Rounds G-O into CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md (16 Nov 2026). Keeps a backup of the old report."""
import os, shutil
P = "../../CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md"
BAK = "../COMPLETE_REPORT_before_16Nov.md"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("*Status: 7 Nov 2026.", "*Status: 16 Nov 2026 (Rounds G–O added: §1 update, §8.9–8.15, §9–§12 revised).")
rep("   - 8.8 Earlier findings: calibration, DVFS, generalization\n",
    "   - 8.8 Earlier findings: calibration, DVFS, generalization\n"
    "   - 8.9 Telemetry prior and the burst guard (Addenda G, H)\n"
    "   - 8.10 A second pipeline (drone) and the fused predictor KPF (Addenda H2, I)\n"
    "   - 8.11 Strong baselines and a changed deadline (Addenda J, K)\n"
    "   - 8.12 Replication on a Raspberry Pi 5 (Addendum L)\n"
    "   - 8.13 A hard window guarantee: WH-ACI (Addendum M)\n"
    "   - 8.14 What a warning is worth to a controller (Addendum N)\n"
    "   - 8.15 Intervention latch-up, its fix, and the closed loop on the Pi 5 (Addendum O)\n")
rep("12. Next steps" + chr(10) + "13.", "12. Assessment and next steps" + chr(10) + "13.")

UPDATE = """**Update, 8–16 Nov 2026 (Rounds G–O; details in §8.9–8.15).** Nine further pre-registered rounds were run: 25 more open-loop runs, a
second pipeline (drone visual odometry), a second embedded board (Raspberry Pi 5), strong learned baselines, and a second closed loop.

| Question | Answer |
|---|---|
| Is our predictor better than standard machine learning? | **No.** Logistic regression on the same features ranks as well or slightly better (0.893 vs 0.888 AUROC). This failed as a frozen test on both boards (§8.11, §8.12) |
| Does the advantage over Toba hold on a second board and a second pipeline? | Yes. Pi 5: AUROC 0.886 / 0.962 vs Toba's 0.673 / 0.810; guarantee 2–3× cheaper (§8.12) |
| Can unwarned misses be bounded in every short window, not only on average? | Yes. A window rule (an old scheduling idea, here applied to warnings) guarantees at most m unwarned misses in any K frames, for any predictor, with a proof that it forces the fewest possible warnings. Passed 6/6 rules; costs +0.01 to +0.02 false-positive rate (§8.13) |
| Why did the first closed loop degrade over 80% of frames? | **Intervention latch-up**: a monitor that acts on its warnings stops seeing normal frames and stays stuck. Found, named and fixed (§8.15) |
| Does the fix work live? | Yes, on the Pi 5: needless degradation in quiet periods fell from 30% to 0.1%, with the same miss rate. The window bound held exactly (worst window 4 of 20). 8 of 9 rules passed (§8.15) |
| Does it degrade fewer frames than Toba? | **No.** Failed on both boards (F5, O4) |
| Does a warning help a controller? | In a simulation driven by the real traces: 5–10× lower control cost than no monitor, and no crashes (§8.14) |
| Are warned frames then on time? | **Not always.** In the live loop, frames degraded at release still missed 10–16% of the time; the degraded mode is not a guaranteed-time fallback (§8.15, found after the frozen analysis) |

**What the paper can claim now.** Not a better predictor. The claim is a monitor that *keeps working once it is acted upon*: a
score-agnostic guarantee layer (long-run bound plus a hard window bound) that stays valid under intervention, the latch-up problem and
its fix, and the release-blindness result against Toba, with pre-registered evidence on three platforms.

"""
rep("**Headline results (hardware; 57 open-loop runs", UPDATE + "**Headline results of the first study, to 7 Nov (hardware; 57 open-loop runs")
rep("| Closed loop on ROS 2 | See §8.7 (G12) |", "| Closed loop on ROS 2 | Misses 24.7% → 3.0% on the UNO Q, but over 80% of frames degraded (§8.7); explained and fixed in §8.15 |")
rep("| **Laptop** (x86) | Ryzen 7 7735HS, Windows 11 |",
    "| **Laptop** (x86) | Ryzen 7 7735HS, Windows 11 |\n"
    "| **Raspberry Pi 5** (embedded, from Addendum L) | 4× Cortex-A76 at 2.8 GHz, 8 GB RAM, Linux 6.12, `performance` governor, active cooling. ROS 2 Jazzy installed without root |")
rep("| **G12** | **Board, ROS 2, closed loop** | real LiDAR | Autoware reference system | 3 seeds × 4 policies | live closed loop |",
    "| **G12** | **Board, ROS 2, closed loop** | real LiDAR | Autoware reference system | 3 seeds × 4 policies | live closed loop |\n"
    "| G13c / G14 (Addendum G) | Board, ROS 2 + telemetry sampler | real LiDAR | Autoware reference system / real applications | 3 + 3 | telemetry prior |\n"
    "| H13 / H14 (Addendum H) | Board, ROS 2 | real LiDAR | as G13c / G14, fresh seeds | 3 + 3 | burst guard |\n"
    "| H part 2, I (drone) | Board, ROS 2 | **EuRoC drone visual odometry**, 200 ms | real applications | 1 + 3 | second pipeline |\n"
    "| I (lidar) | Board, ROS 2 | real LiDAR | real applications | 3 | fused predictor |\n"
    "| **L** | **Raspberry Pi 5, ROS 2** | drone VO and real LiDAR | real applications | 3 + 3 | second embedded board |\n"
    "| **O** | **Raspberry Pi 5, ROS 2, closed loop** | real LiDAR | real applications | 3 seeds × 6 policies | latch-up fix, live window guarantee |")
rep("In total there are 57 open-loop runs (about 300,000 frames) plus the closed-loop runs.",
    "The first study has 57 open-loop runs (about 300,000 frames) plus the closed-loop runs. Rounds G–O add 25 open-loop runs and 18 closed-loop policy runs.")

rep("**Our method (final form): KP + ACI.**", "**Our method in the first study: KP + ACI** (later extended to KPF and a window guarantee; see the update below).")
i = s.index("---\n\n## 9. Where we beat Toba 2026")
s = s[:i] + open("new_sections.md", encoding="utf-8").read() + "---\n\n" + open("new_tail.md", encoding="utf-8").read()
open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok", len(s.split("\n")), "lines")
