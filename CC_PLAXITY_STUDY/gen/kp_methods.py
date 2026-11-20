"""Round E methods (PROTOCOL_GENERALIZATION.md, Addendum E).

KP  -- Kalman plaxity ("state-conditioned plaxity").  Toba-style remaining-time convolution of per-node
       distributions, conditioned on one latent log-scale state z_t shared by all nodes of frame t:
           log(n_tj / med_j) = z_t + r_tj,         r_tj ~ empirical per-node residual (independent, as Toba)
           z_t | past frames ~ N(m_t, q),           m_t = EWMA of z over completed frames (lambda by calib MSE)
       In-frame evidence (finished nodes) updates z_t by a scalar Kalman/Gaussian step.  Because
       sum_j med_j exp(z + r_j) = exp(z) * B, the remaining time at checkpoint k is exp(z) * B_k with B_k the
       Toba convolution of the residual-scaled node PMFs (represented by NQ quantiles).  Per-frame cost:
       NQ normal-tail evaluations per checkpoint.

ACI -- adaptive conformal alarm threshold (Gibbs & Candes 2021, adapted to alarms).  Any method gives
       P_k(miss) at checkpoints k.  Alarm at the first k with P_k > theta_t.  Error e_t = 1{frame misses and
       no alarm was raised} ("unwarned miss").  Update when frame t's outcome is known (lag L frames):
           theta <- theta - gamma * (e_t - alpha).
       Guarantee (any score, any latency sequence, including unseen contention):
           (1/T) sum_t e_t  <=  alpha + (1 + (L+1) gamma) / (gamma T) + L / T      (ONE-SIDED; erratum 7 Nov)
       (theta >= -(L+1)gamma because an unwarned miss needs theta >= 0; telescoping gives the upper bound. theta has no upper bound.)
"""
import os, sys
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cc_plaxity import CP_DONE, NODES

NQ = 256
GRID = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]


def _ewma(z, hp, lam):
    m = np.full(len(z), np.nan); cur = None
    for i in range(len(z)):
        if not hp[i]:
            cur = None
        if cur is not None:
            m[i] = cur
        cur = z[i] if cur is None else lam * cur + (1 - lam) * z[i]
    return m


class KalmanPlaxity:
    def __init__(self, cal, seed=0, nq=NQ):
        self.nq = nq
        X = cal[NODES].values
        self.med = np.median(X, 0)
        self.me2e = np.median(cal.e2e.values)
        z = np.log(cal.e2e.values / self.me2e)
        hp = cal.has_prev.values
        best = None
        for lam in GRID:
            m = _ewma(z, hp, lam); ok = ~np.isnan(m)
            mse = np.mean((z[ok] - m[ok]) ** 2)
            if best is None or mse < best[0]:
                best = (mse, lam)
        self.q, self.lam = best
        self.z0 = 0.0
        self.q0 = np.var(z)                         # prior variance with no history
        R = np.log(X / self.med) - z[:, None]       # per-node residuals
        self.b = R.mean(0); self.v = np.maximum(R.var(0), 1e-6)
        rng = np.random.default_rng(seed)
        M = 20000
        draws = np.stack([rng.choice(R[:, j], M) for j in range(len(NODES))], 1)   # independent residuals
        S = self.med * np.exp(draws)
        qs = (np.arange(nq) + 0.5) / nq
        self.B = {k: np.quantile(S[:, [j for j, c in enumerate(NODES) if c not in done]].sum(1), qs)
                  for k, done in CP_DONE.items()}

    def prior(self, df):
        z = np.log(df.e2e.values / self.me2e)
        m = _ewma(z, df.has_prev.values, self.lam)
        q = np.where(np.isnan(m), self.q0, self.q)
        return np.where(np.isnan(m), self.z0, m), q

    def predict(self, df, D):
        m, q = self.prior(df)
        out = {}
        for k, done in CP_DONE.items():
            prec = 1.0 / q; num = m / q
            for c in done:
                j = NODES.index(c)
                x = np.log(np.maximum(df[c].values, 1e-3) / self.med[j]) - self.b[j]
                prec = prec + 1.0 / self.v[j]; num = num + x / self.v[j]
            mu, sd = num / prec, np.sqrt(1.0 / prec)
            slack = D - df["e%d" % k].values
            P = np.ones(len(df))
            ok = slack > 0
            # P(exp(z) * B > slack) = mean_B P(z > log(slack / B))
            thr = np.log(slack[ok, None] / self.B[k][None, :])
            P[ok] = (1 - ndtr((thr - mu[ok, None]) / sd[ok, None])).mean(1)
            out[k] = P
        return out


def aci_alarms(Pk, y, alpha, gamma, theta0, lag=1):
    """Pk: (n, K) miss probabilities at the checkpoints (time order). Returns alarm checkpoint (-1 = none),
    unwarned-miss indicator and the theta path.  Frame t's error is fed back after `lag` frames."""
    n, K = Pk.shape
    theta = theta0; first = np.full(n, -1); err = np.zeros(n); th = np.empty(n)
    for t in range(n):
        th[t] = theta
        hit = np.nonzero(Pk[t] > theta)[0]
        first[t] = hit[0] if len(hit) else -1
        err[t] = float(y[t] == 1 and first[t] < 0)
        if t - lag >= 0:                     # outcome of frame t-lag becomes known after frame t is released
            theta = theta - gamma * (err[t - lag] - alpha)
    return first, err, th


def aci_bound(T, gamma, lag=1):
    """Value used by the pre-registered E2/F2 checks (frozen). It is LARGER than the correct one-sided margin
    aci_bound_upper(); a check 'rate - alpha <= aci_bound_upper' is the strict, correct test (erratum 7 Nov)."""
    return (1 + 2 * (lag + 1) * gamma) / (gamma * T) + lag / T


def aci_bound_upper(T, gamma, lag=1, theta0=1.0):
    """Correct one-sided guarantee: mean unwarned-miss rate <= alpha + this, for any score and latency sequence."""
    return (theta0 + (lag + 1) * gamma) / (gamma * T) + lag / T


def static_theta(Pk_cal, y_cal, alpha):
    """Largest global threshold whose unwarned-miss rate on calibration frames is <= alpha."""
    s = Pk_cal.max(1)
    cand = np.unique(np.r_[s, 0.0, 1.0])[::-1]
    for c in cand:
        if np.mean((y_cal == 1) & (s <= c)) <= alpha:
            return c
    return 0.0
