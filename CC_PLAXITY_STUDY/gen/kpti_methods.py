"""KPTI: Kalman plaxity with a history + telemetry-INNOVATION prior (exploratory, after the EuRoC drone run showed that a
telemetry-only prior loses to history when latency is driven by scene content).

Reasoning: the EWMA history m_t already contains the current contention level, but lags after a contention switch.
The new information telemetry adds is how contention has CHANGED since that history was formed, so the prior uses
the telemetry innovation  d_x = x_t - EWMA_lambda(x)_{t-1}  (same lambda as m, so the two are aligned):
    mu_t = a + b m_t + sum_x c_x x_t + sum_x g_x d_x,t       x in {u, log1p r, log1p w}
In steady state d_x ~ 0 and the prior is history (+ levels); just after a switch d_x is large and corrects the stale m_t.
OLS on calibration frames; the model is nested (it contains KP's history prior and the telemetry-only prior), so its
calibration fit is never worse than either. Everything after the prior is KP unchanged.
"""
import numpy as np
from kp_methods import _ewma
from kpt_methods import KalmanPlaxityT


def _ewma_prev(x, hp, lam):
    """EWMA of x over previous frames (value available at release of frame t uses x_0..x_{t-1}); resets where hp == 0."""
    out = np.full(len(x), np.nan); m = np.nan
    for t in range(len(x)):
        if not hp[t]:
            m = np.nan
        out[t] = m
        if np.isfinite(x[t]):
            m = x[t] if not np.isfinite(m) else lam * m + (1 - lam) * x[t]
    return out


class KalmanPlaxityTI(KalmanPlaxityT):
    def __init__(self, cal, seed=0, nq=64, features=("m", "u", "r", "w", "du", "dr", "dw")):
        super().__init__(cal, seed=seed, nq=nq, features=features)

    def _design(self, df):
        z = np.log(df.e2e.values / self.me2e); hp = df.has_prev.values
        m = _ewma(z, hp, self.lam)
        base = {"u": df.tm_u.values, "r": np.log1p(df.tm_r.values), "w": np.log1p(df.tm_w.values)}
        cols = {"m": m, **base}
        for k, x in base.items():
            cols["d" + k] = x - _ewma_prev(x, hp, self.lam)
        X = np.column_stack([cols[f] for f in self.features])
        ok = np.isfinite(X).all(1)
        return X, z, ok
