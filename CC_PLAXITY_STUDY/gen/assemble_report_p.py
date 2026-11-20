"""One-off: fold Addendum P (closed loop v3, UNO Q) into CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md (17 Nov 2026).
Keeps a backup of the 16 Nov version and always rebuilds from it, so it can be re-run."""
import os, shutil
P = "../../CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md"
BAK = "../COMPLETE_REPORT_before_17Nov.md"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("*Status: 16 Nov 2026 (Rounds G–O added: §1 update, §8.9–8.15, §9–§12 revised).",
    "*Status: 17 Nov 2026 (Rounds G–P added: §1 update, §8.9–8.16, §9–§12 revised).")
rep("   - 8.15 Intervention latch-up, its fix, and the closed loop on the Pi 5 (Addendum O)\n",
    "   - 8.15 Intervention latch-up, its fix, and the closed loop on the Pi 5 (Addendum O)\n"
    "   - 8.16 A fallback that is on time and a window bound on every late frame (Addendum P)\n")

# §1 update table
rep("**Update, 8–16 Nov 2026 (Rounds G–O; details in §8.9–8.15).** Nine further pre-registered rounds were run: 25 more open-loop runs, a\n"
    "second pipeline (drone visual odometry), a second embedded board (Raspberry Pi 5), strong learned baselines, and a second closed loop.",
    "**Update, 8–17 Nov 2026 (Rounds G–P; details in §8.9–8.16).** Ten further pre-registered rounds were run: 25 more open-loop runs, a\n"
    "second pipeline (drone visual odometry), a second embedded board (Raspberry Pi 5), strong learned baselines, and two more closed loops.")
rep("| Are warned frames then on time? | **Not always.** In the live loop, frames degraded at release still missed 10–16% of the time; the degraded mode is not a guaranteed-time fallback (§8.15, found after the frozen analysis) |",
    "| Are warned frames then on time? | **Not with the first degraded mode:** on the Pi 5, frames degraded at release still missed 10–16% of the time (§8.15). **With a much lighter second mode, yes:** on the UNO Q it ran on 790 frames and was never late (§8.16) |\n"
    "| Can *every* late frame be bounded in every window, live? | **Yes, on the UNO Q.** With the window rule applied to all late frames and the lighter mode as fallback, no 20-frame window held more than 4 late frames in any of 6 runs, under contention unseen in calibration. Other policies reached 7–20. Conditional on the fallback being on time, which was measured, not proven (§8.16) |\n"
    "| Is the latch-up fix the state update itself? | Yes. Without the burst guard, quiet-period degradation is 0–0.7% vs 63–100% for the old monitor. The frozen \"no latch\" ratio rule still failed (P5, 1/3), because the reference rate was exactly 0 (§8.16) |\n"
    "| Is the monitor selective at a light load? | Yes. It degrades 0–7% of frames (the old monitor 18%) and misses no more than random at the same budget. It beat no action in 2 of 3 seeds, failing where almost nothing missed (§8.16) |\n"
    "| Is Toba unstable in the loop and ours not? | **Not shown.** Toba varied 9–69% across seeds, but ours varied too (28–61%); the frozen test failed (P8, §8.16) |")
rep("**What the paper can claim now.** Not a better predictor. The claim is a monitor that *keeps working once it is acted upon*: a\n"
    "score-agnostic guarantee layer (long-run bound plus a hard window bound) that stays valid under intervention, the latch-up problem and\n"
    "its fix, and the release-blindness result against Toba, with pre-registered evidence on three platforms.",
    "**What the paper can claim now.** Not a better predictor. The claim is a monitor that *keeps working once it is acted upon*:\n"
    "- a score-agnostic guarantee layer (long-run bound plus a hard window bound) that stays valid under intervention;\n"
    "- with a fallback mode, a window bound on *every late frame*, held live on hardware, which is exactly the precondition of a control\n"
    "  stability certificate;\n"
    "- the latch-up problem and its isolated fix;\n"
    "- the release-blindness result against Toba;\n"
    "- pre-registered evidence on three platforms.")

# new section before §9
rep("---\n\n## 9. Where we beat Toba 2026", open("new_section_p.md", encoding="utf-8").read() + "## 9. Where we beat Toba 2026")

# §8.14 certificate pointer
rep("switched-system argument of the weakly-hard control literature and holds only if every warned frame delivers on time.",
    "switched-system argument of the weakly-hard control literature and holds only if every warned frame delivers on time.\n"
    "  §8.16 later measured that precondition on the UNO Q: with the all-late window rule and the lighter mode, no window held more\n"
    "  than 4 late frames.")

# §9
rep("5. **With a guarantee at all.** Toba's fixed thresholds have none. We give two: a long-run bound, and a hard bound on every window of K\n"
    "   frames (§8.13), both valid while the system acts on the warnings.",
    "5. **With a guarantee at all.** Toba's fixed thresholds have none. We give two: a long-run bound, and a hard bound on every window of K\n"
    "   frames (§8.13), both valid while the system acts on the warnings. With a fallback mode, the window bound covers every late frame:\n"
    "   at most 4 in any 20 on the UNO Q, against up to 20 for Toba (§8.16).")
rep("6. **In a live loop on two boards** in missed deadlines (3.0% vs 6.9% on the UNO Q, 9.9% vs 15.4% on the Pi 5).",
    "6. **In a live loop on two boards** in missed deadlines (3.0% vs 6.9% on the UNO Q, 9.9% vs 15.4% on the Pi 5; in Addendum P on the\n"
    "   UNO Q, 4.8–5.2% vs 18.3% heavy and 4.9–5.5% vs 7.6% light, descriptive only).")
rep("4. **Frames degraded in the loop.** We do not degrade fewer frames than Toba (F5 and O4 both failed).",
    "4. **Frames degraded in the loop.** We do not degrade fewer frames than Toba (F5 and O4 both failed). At the light load in\n"
    "   Addendum P, Toba degraded almost no frames (and missed more than no action).\n"
    "5. **Stability across seeds.** We could not show that Toba's alarm rate in the loop is less stable than ours (P8 failed).")
rep("(c) the latch-up problem and its fix; (d) pre-registered evidence on three platforms.",
    "(c) the latch-up problem and its fix; (d) a live window bound on every late frame with an on-time fallback; (e) pre-registered\n"
    "  evidence on three platforms.")
rep("of K frames, and keeps working when the system acts on the warnings, as confirmed live on two embedded boards.*",
    "of K frames, and keeps working when the system acts on the warnings, as confirmed live on two embedded boards; with an on-time\n"
    "fallback it caps every late frame, which is what a controller's stability proof needs.*")

# §10
rep("| **Addendum O** (closed loop on the Pi 5, 9 rules) | 8 passed; **O4 failed** (not fewer degraded frames than Toba, 1/3) |",
    "| **Addendum O** (closed loop on the Pi 5, 9 rules) | 8 passed; **O4 failed** (not fewer degraded frames than Toba, 1/3) |\n"
    "| **Addendum P** (closed loop v3 on the UNO Q, 13 rules) | 10 passed (fallback 0/790 late; window bound on all late frames 6/6); **P5 failed** (1/3, zero reference rate), **P7a failed** (2/3), **P8 failed** |")

# §11
rep("- **The guarantees cover unwarned misses only, and the degraded mode is not always on time.** Warned frames still missed the deadline\n"
    "  10–16% of the time in the Pi 5 closed loop. A stability claim needs a fallback with a guaranteed finish time, which is not built.",
    "- **The window bound on all late frames depends on the fallback being on time, which is measured, not proven.** The lighter mode was\n"
    "  never late in 790 frames on the UNO Q, but it has no worst-case execution time analysis, and it was tested on one board and one\n"
    "  pipeline. The first degraded mode (half the points) was late 10–16% of the time on the Pi 5. The control plant is still simulated.")
rep("- **Closed loop.** Two experiments, one pipeline (lidar), three seeds each, heavy load. All monitors degrade 40% or more of frames. We do\n"
    "  not degrade fewer frames than Toba (F5, O4). Latch-up was reproduced in 2 of 3 seeds on the Pi 5.",
    "- **Closed loop.** Three experiments, one pipeline (lidar), three seeds per load. At the heavy load all monitors degrade 40% or more\n"
    "  of frames; at the light load the fixed monitor degrades 0–7% but beat no action in only 2 of 3 seeds (P7a). We do not degrade\n"
    "  fewer frames than Toba (F5, O4), and could not show Toba is less stable in the loop (P8). Latch-up was reproduced on both boards.\n"
    "- **Some frozen rules were badly specified.** P5 compared against a reference rate that turned out to be exactly 0. It stays failed.")

# §12
rep("1. A hard (m, K) window guarantee on unwarned misses for any predictor, with a minimality result, shown live on hardware (§8.13, §8.15).",
    "1. A hard (m, K) window guarantee for any predictor, with a minimality result, shown live on hardware: on unwarned misses (§8.13,\n"
    "   §8.15) and, with an on-time fallback, on every late frame (§8.16). This is the precondition of a control stability certificate.")
rep("2. Intervention latch-up as a named failure of closed-loop monitors, with a fix (§8.15).",
    "2. Intervention latch-up as a named failure of closed-loop monitors, with a fix shown to be the state update itself (§8.15, §8.16).")
rep("3. Run the next closed-loop round (drafted as `ADDENDUM_P_DRAFT.md`, not yet frozen or run): a much lighter fallback and a test that\n"
    "   warned frames finish on time; a lighter load; one more policy that isolates the latch-up fix from the burst guard; a pre-registered\n"
    "   test of whether Toba's alarms are unstable in the loop; and the window rule applied to all late frames, not only unwarned ones.\n"
    "4. If step 3 holds, state the stability certificate: plant gives (m, K), monitor enforces it, loop is stable under any contention.\n"
    "5. Write the manuscript around §9's four contributions, with the failed tests kept in.",
    "3. Done: Addendum P (§8.16), 10 of 13 rules passed, including all window and fallback rules.\n"
    "4. State the stability certificate as a theorem with its two assumptions: the fallback finishes on time, and the plant meets the\n"
    "   block-gain condition for the chosen (m, K). Keep it explicit that the first is measured, not proven.\n"
    "5. Optional, for strength: repeat the all-late window rule on the Pi 5, or bound the fallback's execution time by measurement under\n"
    "   the worst contention.\n"
    "6. Write the manuscript around §9's contributions, with the failed tests kept in.")

# §13
rep("| R9 | `figures/R9_control_study.png` | control case study (Addendum N) |",
    "| R9 | `figures/R9_control_study.png` | control case study (Addendum N) |\n"
    "| R10 | `figures/R10_closed_loop_v3.png` | closed loop v3 on the UNO Q, heavy and light load: window bound on all late frames (Addendum P) |")
rep("- Rounds G–O: `study/gen/results/<g|h|i|j|k|l|m|n|o>_confirm*_summary.txt` and the matching `*_runs.csv`.",
    "- Rounds G–P: `study/gen/results/<g|h|i|j|k|l|m|n|o|p>_confirm*_summary.txt` and the matching `*_runs.csv`.")
rep("- Raw runs: `study/gen/data/` (`I_vo`, `I_g14`, `L_vo`, `L_g14`, `O`, and earlier rounds).",
    "- Raw runs: `study/gen/data/` (`I_vo`, `I_g14`, `L_vo`, `L_g14`, `O`, `P`, and earlier rounds).")
rep("- Scripts: `study/gen/report_curves.py`, `report_figures.py`, `report_figures_r8.py`;",
    "- Scripts: `study/gen/report_curves.py`, `report_figures.py`, `report_figures_r8.py`, `report_figures_r10.py`;")
open(P, "w", encoding="utf-8", newline="\n").write(s)
print("ok", len(s.split("\n")), "lines")
