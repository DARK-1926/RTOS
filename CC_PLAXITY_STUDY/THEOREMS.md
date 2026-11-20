# Theorems and proofs (for independent review)

*18 Nov 2026. Written so that a real-time-systems or control researcher can check each statement in one sitting.
Every theorem is also checked by code. `gen/q_theorems.py` explores every reachable state for small parameters, with negative
controls. `gen/wh_theory.py` uses random and adversarial sequences, and `gen/cert_explore.py` enumerates every block pattern.
Code checks are not proofs. The proofs below are the claim, and the code only guards against slips.*

---

## Setting

- Frames t = 0, 1, 2, … are released in order, one per period.
- Each frame has a relative deadline D. Frame t is **late** if it finishes after D or never finishes.
- At release, the monitor gives each frame a mode:
  - **normal**;
  - **level 1** (lighter work; may still be late);
  - **level 2** (much lighter work).
- During the frame, the monitor may raise an **alarm** at checkpoints. A frame is **warned** if it was degraded at release or alarmed
  at any checkpoint.
- An **unwarned miss** is a late frame that was not warned.
- The outcome of frame j (late or not) becomes known to the monitor at most `lag` releases after j. Before that, frame j is
  **running**.
- Nothing is assumed about latencies: the sequence can be adversarial, non-stationary, and shaped by the monitor's own actions.

---

## Theorem 1: (m, K) window bound on unwarned misses (Addendum M)

**Rule W_u.** Before releasing frame t, let

c_t = #{ j ∈ [t−K+1, t−1] : j is a known unwarned miss, or j is unwarned and still running }.

If c_t ≥ m, warn frame t unconditionally (degrade it at release). Otherwise follow the predictor.

**Claim.** For every predictor, every latency sequence and every lag, every window of K consecutive frames contains at most m
unwarned misses.

**Proof.**
1. Suppose some window W = [a, a+K−1] contains m+1 unwarned misses, and let f be the last of them.
2. f was not warned, so W_u did not force it: c_f < m.
3. The other m unwarned misses in W lie in [a, f−1] ⊆ [f−K+1, f−1].
4. At f's release, each of them is either already known (a known unwarned miss) or still running and unwarned. Either way it is
   counted in c_f.
5. Hence c_f ≥ m, which contradicts step 2. ∎

## Theorem 2: minimality of W_u

**Claim.** Let A be any causal policy that guarantees (m, K) on unwarned misses for every latency sequence. Whenever c_t ≥ m, A must
warn frame t.

**Proof.**
1. When c_t ≥ m, the frames counted in c_t lie in [t−K+1, t−1].
2. An adversary consistent with everything A has observed can make every still-running counted frame late, and frame t late too.
3. If A does not warn t, the window [t−K+1, t] then holds at least m+1 unwarned misses. ∎

So the forced warnings of W_u are exactly the unavoidable ones. The predictor decides only how often the budget is spent, which is
the false-alarm price. It never decides whether the guarantee holds.

---

## Theorem 3: (m, K) window bound on all late frames, given an on-time fallback (Addendum P, new)

**Rule W (policy `kpmd`).** Before releasing frame t, let

c_t = #{ j ∈ [t−K+1, t−1] : j is known late, or j is still running and not at level 2 }.

If c_t ≥ m, run frame t at level 2.

**Claim (exact form, no assumption).** For every latency sequence, every window W of K consecutive frames satisfies

#late frames in W ≤ m + #late level-2 frames in W.

**Corollary (assumption A: level-2 frames are never late).** Every window of K consecutive frames holds at most m late frames.

**Proof.**
1. It suffices to show that every window W holds at most m late frames that are *not* at level 2.
2. Suppose W holds m+1 such frames, and let f be the last of them.
3. f is not at level 2, so W did not force it: c_f < m.
4. The other m late, non-level-2 frames of W lie in [f−K+1, f−1].
5. At f's release, each is either known late, or still running and not at level 2. Either way it is counted in c_f.
6. Hence c_f ≥ m, a contradiction. The corollary follows at once. ∎

**Minimality** is as in Theorem 2: when c_t ≥ m, releasing t at anything other than an on-time mode lets an adversary put m+1 late
frames in [t−K+1, t].

**What is assumed, and what is measured.**
- Assumption A is the only assumption. It is not proven for our pipeline: there is no worst-case execution time analysis.
- On the UNO Q (deadline 27 ms), level 2 ran on 790 frames and was never late, with a worst case of 20.1 ms (§8.16).
- On the Pi 5 (Addendum Q, confirmatory, 3 seeds), A fails at a 6 ms deadline. Level 2 was late on 77 of 1312 frames (5.9%) in the
  closed loop, and on 2.3–10.2% of frames in stress runs under constant contention. The tail comes from OS scheduling, not from
  the amount of work: user-space core pinning did not remove it. At 10 ms, 13 of 8913 stress frames were late (0.15%; 95% upper
  bound 0.23%).
- Even so, the exact form held in all 6 runs: the worst window of late frames not at level 2 was exactly m = 4. The all-late count
  reached 5–7, and every excess was a late level-2 frame.
- So A is a property of the platform and the deadline, and must be established separately, by WCET analysis, a reservation or
  isolation mechanism, or measurement with a stated confidence. The exact form of the claim holds regardless and says how the bound
  degrades: one extra late frame per late level-2 frame in the window.

---

## Theorem 4: long-run bound on unwarned misses, valid under intervention

**Rule (adaptive conformal threshold).**
- Frame t raises an alarm if its score exceeds θ_t at some checkpoint. Scores lie in [0, 1].
- Let e_t = 1 if frame t is an unwarned miss, and 0 otherwise.
- The threshold is updated when outcomes become known: θ_{t+1} = θ_t − γ (e_{t−lag} − α).

**Claim.** For every score sequence and every latency sequence, including sequences changed by the monitor's own interventions,

(1/T) Σ_{t<T} e_t ≤ α + (θ_0 + (lag+1)γ) / (γT) + lag/T.

**Proof.**
1. If θ_t < 0, every score exceeds θ_t, so frame t is warned and e_t = 0.
2. The threshold falls only when an unwarned miss is fed back, by γ(1−α) ≤ γ at a time.
3. An unwarned miss at frame s needs θ_s ≥ 0, and its feedback arrives lag releases later. So at most lag+1 decrements can act
   after θ last stood at or above 0, which gives θ_t ≥ −(lag+1)γ for all t.
4. Telescoping: θ_T = θ_0 − γ Σ_{t<T−lag} (e_t − α).
5. Hence Σ_{t<T−lag} (e_t − α) ≤ (θ_0 + (lag+1)γ)/γ.
6. The last lag frames add at most lag. Dividing by T gives the claim. ∎

**Why intervention does not break it.**
- The proof uses no exchangeability or stationarity: it holds for each individual sequence.
- e_t is defined only on unwarned frames, which run unmodified, so their outcomes are observed as they are.
- This is the standard adaptive conformal argument (Gibbs and Candès, NeurIPS 2021), stated in its one-sided form. Its use under
  intervention is ours, but it is not new mathematics.

---

## Theorem 5: stability certificate (control side; plant-specific)

**Setting.**
- Discrete-time plant x⁺ = A_d x + B_d u, with state feedback u = −K x computed from each frame's estimate.
- A frame that is late gives no new command: the controller **holds** the previous one.
- With z = [x; u_prev], an on-time frame applies matrix U and a held frame applies matrix H.
- Level-2 frames are on time, with a noisier estimate. Noise enters as a bounded disturbance, so the deterministic part is U.

**Claim.** Suppose every window of K frames has at most m held frames (Theorem 3 under assumption A). Suppose also that, for some
norm ‖·‖_P,

g = max over all words w ∈ {U, H}^K with at most m H's of ‖Π_{t∈w} M_t‖_P < 1.

Then the closed loop is exponentially stable for every admissible pattern of late frames: ‖z_{nK}‖_P ≤ gⁿ ‖z_0‖_P. With a bounded
disturbance, it is input-to-state stable.

**Proof.**
1. Split time into consecutive, non-overlapping blocks of K frames.
2. Each block is one of the windows bounded by Theorem 3, so it has at most m held frames.
3. Its product is one of the words in the maximum, so its norm is at most g.
4. By sub-multiplicativity of the induced norm, the product of n blocks has norm at most gⁿ.
5. Within a block, the growth is bounded by max(‖U‖, ‖H‖)^K. ∎

**Checked numerically** (`gen/cert_explore.py`, all words enumerated, Lyapunov-shaped norm from the update mode):
- g = 0.001–0.14 for all 8 plant settings and all 6 budgets, including (4, 20).
- With no bound on holds, the hold mode grows 1.22–2.72× per frame, so it is unstable.
- The certificate is sufficient, not necessary.

---

## What a reviewer should check

1. **Theorems 1–3:** the counting argument, and that "running and not at level 2" (or "unwarned and running") is exactly what the
   implementation counts. Implementation: `late_spent()` in `ros2/cc_lidar_ros/src/lidar_ros_cl3.cpp`. Exhaustive check:
   `gen/q_theorems.py`, which explores 38,900 reachable states for K ≤ 7 plus cases up to K = 10. Its negative controls confirm that
   a rule ignoring running frames, and a level 2 that can be late, are both caught.
2. **Theorem 4:** the lower bound on θ with lag, and the one-sided form.
3. **Theorem 5:** that holding the previous command is the right model of a late frame, and that the norm choice is valid
   (sufficient condition).
4. **Prior art:**
   - The counting rule is in the spirit of (m, k)-firm scheduling and the Bi-Modal Scheduler's panic mode (Bernat and Cayssials,
     RTSS 2001).
   - The certificate uses the standard weakly-hard control argument.
   - What we believe is new is the combination: a learned, conformal alarm with a score-agnostic window guarantee; its validity
     under intervention; the late-frame version with a fallback and its exact degradation form; and evidence on hardware.
