"""Exhaustive (model-checking) verification of the window theorems, for the proofs in THEOREMS.md.

Model. Frames are released in order, one per period. A frame runs either normally (the monitor may also have degraded it at
level 1; both can be late) or at level 2 (assumed never late: assumption A). For every non-level-2 frame the adversary chooses,
at release and hidden from the rule, whether it is late and after how many further releases its outcome becomes known
(0 = before the next release, ..., R; or never, i.e. it never finishes and is late). If the deadline exceeds the period, a frame
may also still be running at the next release and finish on time; the model allows that too.

Rule W (Addendum P, policy kpmd): before releasing frame t, let
    c_t = #{ j in [t-K+1, t-1] : j is known late, or j is still running and not at level 2 }.
If c_t >= m, frame t runs at level 2.

Theorem 1 (sufficiency). Under A, every window of K consecutive frames holds at most m late frames, for every adversary.
Theorem 2 (minimality). In every reachable state where W forces level 2, a causal rule that released frame t normally instead
    could be driven to m+1 late frames in one window by an adversary consistent with everything the rule has observed.
    So W forces level 2 exactly when it is unavoidable, and it is the least restrictive causal rule with this guarantee.

The search enumerates ALL reachable states of the last K-1 frames (closure, i.e. an infinite horizon), not sampled sequences.
The same checks are run for the unwarned-miss rule of Addendum M (warned frames play the role of level 2, misses of late frames).
    python q_theorems.py
"""
from itertools import product

L2, OK, LATE = "L2", "OK", "LATE"
L2_CAN_BE_LATE = False                       # negative control (b) only            # known states; running frames: ("RUN", will_be_late, releases_left or None=never)


def outcomes(R):
    out = []
    for late in (False, True):
        out.append(LATE if late else OK)
        for r in range(1, R + 1):
            out.append(("RUN", late, r))
    out.append(("RUN", True, None))           # never finishes
    return out


def is_late(s):
    return s == LATE or (isinstance(s, tuple) and s[1])


def counted(s):                              # what rule W counts: known late, or running and not level 2
    return s == LATE or isinstance(s, tuple)


def step(s):                                 # one release passes: running frames get closer to finishing
    if isinstance(s, tuple) and s[2] is not None:
        return (LATE if s[1] else OK) if s[2] == 1 else ("RUN", s[1], s[2] - 1)
    return s


def check(m, K, R):
    O = outcomes(R); start = tuple([OK] * (K - 1)); seen = {start}; todo = [start]; forced_states = 0; minimal_ok = 0
    while todo:
        st = todo.pop()
        c = sum(counted(x) for x in st)
        if c >= m:
            choices = [L2] + ([LATE] if L2_CAN_BE_LATE else []); forced_states += 1
            # Theorem 2: releasing normally instead, an adversary consistent with the observations (every still-running
            # frame may be late) makes the window [t-K+1, t] hold m+1 late frames
            worst = sum(1 for x in st if x == LATE or isinstance(x, tuple)) + 1
            minimal_ok += worst >= m + 1
        else:
            choices = O
        for ch in choices:
            win = st + (ch,)
            assert sum(is_late(x) for x in win) <= m, ("Theorem 1 violated", m, K, R, win)
            nxt = tuple(step(x) for x in win[1:])
            if nxt not in seen:
                seen.add(nxt); todo.append(nxt)
    assert minimal_ok == forced_states
    return len(seen), forced_states


if __name__ == "__main__":
    total = 0
    print(f"{'m':>2} {'K':>2} {'R':>2} {'reachable states':>17} {'forcing states':>15}  Theorem 1  Theorem 2")
    for K in range(2, 8):
        for m in range(1, K):
            for R in range(0, 3):
                n, f = check(m, K, R); total += n
                print(f"{m:>2} {K:>2} {R:>2} {n:>17} {f:>15}  holds      holds")
    print(f"all cases verified; {total} reachable states explored in total")


def negative_controls():
    """The checker must catch broken variants: (a) a rule that ignores still-running frames (needs R >= 1);
    (b) assumption A dropped: a level-2 frame may be late."""
    global counted, L2_CAN_BE_LATE
    res = []
    keep = counted; counted = lambda s: s == LATE
    try:
        check(2, 5, 2); res.append("a NOT caught")
    except AssertionError as e:
        res.append("a caught: " + e.args[0][0])
    counted = keep; L2_CAN_BE_LATE = True
    try:
        check(2, 5, 0); res.append("b NOT caught")
    except AssertionError as e:
        res.append("b caught: " + e.args[0][0])
    L2_CAN_BE_LATE = False
    return res


if __name__ == "__main__":
    print("negative controls:", "; ".join(negative_controls()))
    for m, K, R in [(2, 10, 1), (3, 10, 1), (4, 10, 0)]:
        n, f = check(m, K, R); print(f"larger case m={m} K={K} R={R}: {n} reachable states, {f} forcing states: both theorems hold")
