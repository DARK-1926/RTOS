"""One-off: fold Addendum Q (closed loop v3 on the Pi 5, fallback stress) into CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md (18 Nov 2026).
Keeps a backup of the 17 Nov version and always rebuilds from it, so it can be re-run."""
import os, shutil
P = "../../CC_PLAXITY_PAPER_PACKAGE/COMPLETE_REPORT.md"
BAK = "../COMPLETE_REPORT_before_18Nov.md"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("*Status: 17 Nov 2026 (Rounds G–P added: §1 update, §8.9–8.16, §9–§12 revised).",
    "*Status: 18 Nov 2026 (Rounds G–Q added: §1 update, §8.9–8.17, §9–§12 revised; proofs in `study/THEOREMS.md`, related work in `study/RELATED_WORK_TABLE.md`).")
rep("   - 8.16 A fallback that is on time and a window bound on every late frame (Addendum P)\n",
    "   - 8.16 A fallback that is on time and a window bound on every late frame (Addendum P)\n"
    "   - 8.17 The window rule on a second board, and where the fallback stops being on time (Addendum Q)\n")
rep("| Can *every* late frame be bounded in every window, live? | **Yes, on the UNO Q.** With the window rule applied to all late frames and the lighter mode as fallback, no 20-frame window held more than 4 late frames in any of 6 runs, under contention unseen in calibration. Other policies reached 7–20. Conditional on the fallback being on time, which was measured, not proven (§8.16) |",
    "| Can *every* late frame be bounded in every window, live? | **Yes, on the UNO Q.** With the window rule applied to all late frames and the lighter mode as fallback, no 20-frame window held more than 4 late frames in any of 6 runs, under contention unseen in calibration. Other policies reached 7–20. Conditional on the fallback being on time, which was measured, not proven (§8.16) |\n"
    "| Does it replicate on the Pi 5? | **The theorem, exactly: yes** (≤ 4 late non-fallback frames in every window, 6/6). **The all-late bound: no** (worst 5–7), because the fallback itself was late 5.9% of the time at 6 ms; still far better than every other policy (10–20). At 10 ms the fallback was late 0.15% under sustained stress, yet one late fallback frame was enough to exceed the bound in 2 of 3 runs (§8.17) |")
rep("- with a fallback mode, a window bound on *every late frame*, held live on hardware, which is exactly the precondition of a control\n"
    "  stability certificate;\n",
    "- with a fallback mode, a window bound on *every late frame* in exact form (≤ m + late fallback frames), held live on two boards;\n"
    "  with an on-time fallback (measured on the UNO Q, not on the Pi 5 at 6 ms) this is the precondition of a control stability\n"
    "  certificate;\n")
i = s.index("\n## 9. Where we beat Toba 2026")
s = s[:i] + "\n" + open("new_section_q.md", encoding="utf-8").read().rstrip() + "\n\n---\n" + s[i:]
rep("   at most 4 in any 20 on the UNO Q, against up to 20 for Toba (§8.16).",
    "   at most 4 in any 20 on the UNO Q, against up to 20 for Toba (§8.16); on the Pi 5, where the fallback was not always on time,\n"
    "   5–7 against 17 for Toba (§8.17).")
rep("| **Addendum P** (closed loop v3 on the UNO Q, 13 rules) | 10 passed (fallback 0/790 late; window bound on all late frames 6/6); **P5 failed** (1/3, zero reference rate), **P7a failed** (2/3), **P8 failed** |",
    "| **Addendum P** (closed loop v3 on the UNO Q, 13 rules) | 10 passed (fallback 0/790 late; window bound on all late frames 6/6); **P5 failed** (1/3, zero reference rate), **P7a failed** (2/3), **P8 failed** |\n"
    "| **Addendum Q** (closed loop v3 on the Pi 5 + fallback stress, 14 rules) | 11 passed (theorem exact form 6/6; stress prediction at 6 ms 3/3; 0.15% late at 10 ms); **Q1 failed** (fallback late 5.9%), **Q3 failed** (0/3, worst 5–7), **Q3L failed** (1/3) |")
rep("- **The window bound on all late frames depends on the fallback being on time, which is measured, not proven.** The lighter mode was\n"
    "  never late in 790 frames on the UNO Q, but it has no worst-case execution time analysis, and it was tested on one board and one\n"
    "  pipeline. The first degraded mode (half the points) was late 10–16% of the time on the Pi 5. The control plant is still simulated.",
    "- **The window bound on all late frames depends on the fallback being on time, and on the Pi 5 at 6 ms it was not.** The lighter\n"
    "  mode was never late in 790 frames on the UNO Q, but on the Pi 5 it was late 5.9% of the time in the loop and 2–10% under\n"
    "  sustained stress, from OS scheduling jitter; user-space core pinning did not help. The all-late bound failed there (Q3, Q3L);\n"
    "  only its exact form held. A hard bound needs a fallback under a reservation or real-time priority, which we could not test\n"
    "  without privileges. The control plant is still simulated.")
rep("2. Contact the Toba authors for code or benchmark parity.\n3. Done: Addendum P (§8.16), 10 of 13 rules passed, including all window and fallback rules.",
    "2. Contact the Toba authors for code or benchmark parity (draft email: `study/EMAIL_TO_TOBA_AUTHORS.md`).\n"
    "3. Done: Addendum P (§8.16), 10 of 13 rules passed; Addendum Q (§8.17), 11 of 14, with the theorem's exact form confirmed on a second\n"
    "   board and the fallback's limit found.")
rep("5. Optional, for strength: repeat the all-late window rule on the Pi 5, or bound the fallback's execution time by measurement under\n"
    "   the worst contention.",
    "5. Done: the all-late rule on the Pi 5 and stress tests of the fallback (§8.17). Optional, for strength: the same with the fallback\n"
    "   under SCHED_FIFO or a CPU reservation on a board where privileges are allowed.")
rep("| R10 | `figures/R10_closed_loop_v3.png` | closed loop v3 on the UNO Q, heavy and light load: window bound on all late frames (Addendum P) |",
    "| R10 | `figures/R10_closed_loop_v3.png` | closed loop v3 on the UNO Q, heavy and light load: window bound on all late frames (Addendum P) |\n"
    "| R11 | `figures/R11_closed_loop_v3_pi.png` | closed loop v3 on the Pi 5: all-late windows and the exact form of the theorem (Addendum Q) |")
rep("- Rounds G–P: `study/gen/results/<g|h|i|j|k|l|m|n|o|p>_confirm*_summary.txt`",
    "- Rounds G–Q: `study/gen/results/<g|h|i|j|k|l|m|n|o|p|q>_confirm*_summary.txt` (Q also `q_confirm_stress.csv`)")
rep("(`I_vo`, `I_g14`, `L_vo`, `L_g14`, `O`, `P`, and earlier rounds)", "(`I_vo`, `I_g14`, `L_vo`, `L_g14`, `O`, `P`, `Q`, and earlier rounds)")
rep("`report_figures_r8.py`, `report_figures_r10.py`; theorems checked in `wh_theory.py`.",
    "`report_figures_r8.py`, `report_figures_r10.py`, `report_figures_r11.py`; theorems in `study/THEOREMS.md`, checked in `wh_theory.py`\n"
    "  and exhaustively in `q_theorems.py`.")
rep("   §8.15) and, with an on-time fallback, on every late frame (§8.16). This is the precondition of a control stability certificate.",
    "   §8.15) and, with an on-time fallback, on every late frame (§8.16); in exact form (≤ m + late fallback frames) on two boards\n"
    "   (§8.17). With an on-time fallback this is the precondition of a control stability certificate.")
open(P, "w", encoding="utf-8").write(s)
print("ok", len(s))
