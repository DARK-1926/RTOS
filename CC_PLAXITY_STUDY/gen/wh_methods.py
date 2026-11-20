"""Weakly-hard guarded ACI (WH-ACI): a DETERMINISTIC (m, K) window guarantee on unwarned deadline misses.

Rule added to burst-guarded ACI (h_confirm.bg_aci, W=50, kappa=0.05): before frame t is released, count the unwarned misses
among the frames of the last K-1 releases whose outcome is already known (frames <= t-lag, the convention of bg_aci). If that count is >= m, frame t
is warned unconditionally at release (budget spent).

Theorem (any score, any latency sequence, any contention): in every window of K consecutive frames the number of unwarned
misses is at most m + lag - 1 (= m for lag 1).  Proof: let t be the last unwarned miss in a window of K frames and suppose the window holds
m + lag unwarned misses. The other m + lag - 1 lie in the K-1 frames before t; at most lag - 1 of them are in frames
t-lag+1 .. t-1 (outcome not yet known at t's release), so at least m are known and inside the last K-1 releases: the rule
warns frame t, a contradiction.  The ACI long-run bound is kept (forced warnings only remove errors; theta is updated as before).
So the stream of unwarned misses satisfies the weakly-hard constraint (m+lag-1, K) of Bernat, Burns & Llamosi.
"""
import numpy as np


def wh_aci(Pk, y, alpha, gamma, theta0, lag=1, W=50, kappa=0.05, m=2, K=20):
    n = len(y); theta = theta0; first = np.full(n, -1); err = np.zeros(n); forced = np.zeros(n, bool)
    for t in range(n):
        lo, hi = max(0, t - lag - W + 1), t - lag + 1                    # exactly as h_confirm.bg_aci: frames <= t-lag are known
        th = theta - (kappa * max(0.0, err[lo:hi].sum() - alpha * W) if hi > 0 else 0.0)
        lo2 = max(0, t - K + 1)                                          # known frames within the last K-1 releases
        if hi > lo2 and err[lo2:hi].sum() >= m:
            first[t] = 0; forced[t] = True
        else:
            hit = np.nonzero(Pk[t] > th)[0]
            first[t] = hit[0] if len(hit) else -1
        err[t] = float(y[t] == 1 and first[t] < 0)
        if t - lag >= 0:
            theta = theta - gamma * (err[t - lag] - alpha)
    return first, err, forced


def max_window(e, K):
    c = np.convolve(e, np.ones(K), "valid") if len(e) >= K else np.array([e.sum()])
    return int(c.max())
