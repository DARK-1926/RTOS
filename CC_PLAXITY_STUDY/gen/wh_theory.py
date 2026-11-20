"""Theory companion to wh_methods.py (frozen): the general-lag rule and numerical checks of the two theorems.

General rule (outcome of frame j is known `lag` frames later): before releasing frame t, let
    c_t = (# known unwarned misses) + (# unwarned frames whose outcome is still unknown)   among the last K-1 releases.
Warn frame t unconditionally iff c_t >= m.            (For lag = 1 nothing is unknown, so this is exactly wh_methods.wh_aci.)

Theorem 1 (sufficiency, exact budget). For any score, latency sequence and lag, every window of K consecutive frames holds
    at most m unwarned misses.
Theorem 2 (necessity / minimality). Let A be ANY causal alarm policy that guarantees (m, K) on every latency sequence.
    Whenever c_t >= m, A must warn frame t: otherwise an adversary lets every still-unknown unwarned frame and frame t miss,
    which puts m + 1 unwarned misses in one window. Hence the forced warnings of the rule are exactly the unavoidable ones:
    no policy with the same deterministic guarantee can warn on fewer of these frames. The predictor only decides how often the
    budget gets spent, i.e. the price (false alarms), never the guarantee.
"""
import numpy as np


def wh_general(P, y, theta, lag, m, K):
    """Fixed threshold theta (the adaptive part is irrelevant to the theorems). Returns warned, unwarned-miss, forced arrays."""
    n = len(y); warned = np.zeros(n, bool); err = np.zeros(n); forced = np.zeros(n, bool)
    for t in range(n):
        lo = max(0, t - K + 1); kn = max(lo, t - lag + 1)                 # frames < kn are known, kn..t-1 unknown
        c = err[lo:kn].sum() + (~warned[kn:t]).sum()
        if c >= m:
            warned[t] = forced[t] = True
        else:
            warned[t] = P[t] > theta
        err[t] = float(y[t] == 1 and not warned[t])
    return warned, err, forced


def maxwin(e, K):
    return int(np.convolve(e, np.ones(K), "valid").max())


if __name__ == "__main__":
    rng = np.random.default_rng(0); worst = 0; tight = 0; tot = 0
    for trial in range(4000):
        n = 400; lag = int(rng.integers(1, 6)); K = int(rng.integers(lag + 2, 60)); m = int(rng.integers(1, max(2, K // 3)))
        mode = trial % 4                                                   # random, bursty, adversarial-to-score, all-miss
        y = (rng.random(n) < rng.uniform(0.05, 0.9)).astype(int)
        if mode == 1:
            y = (np.sin(np.arange(n) / rng.uniform(3, 40)) > rng.uniform(-0.5, 0.8)).astype(int)
        P = rng.random(n)
        if mode == 2:
            P = np.where(y == 1, 0.0, 1.0)                                 # predictor is exactly wrong
        if mode == 3:
            y[:] = 1; P[:] = 0.0                                           # every frame misses, predictor never warns
        w, e, f = wh_general(P, y, 0.5, lag, m, K)
        mw = maxwin(e, K); worst = max(worst, mw - m); tot += 1; tight += (mw == m)
        assert mw <= m, (trial, lag, m, K, mw)
    print(f"Theorem 1: {tot} random/adversarial sequences, lags 1-5: max(window count - m) = {worst} (never exceeded); bound attained in {tight}/{tot}")
    # Theorem 2 demo: skip one forced warning -> an adversary breaks the budget
    n, lag, m, K = 200, 3, 2, 10; y = np.ones(n, int); P = np.zeros(n)
    w, e, f = wh_general(P, y, 0.5, lag, m, K); t = int(np.nonzero(f)[0][0])
    e2 = e.copy(); e2[t] = 1.0
    print(f"Theorem 2 demo: first forced frame t={t}; if it were not warned and missed, its window would hold {maxwin(e2[:t+1], min(K, t+1))} > m={m}")
    print(f"all-miss, never-warning predictor: unwarned misses per window = {maxwin(e, K)} (= m), warned fraction = {w.mean():.2f} (minimum possible = {1 - m / K:.2f})")
