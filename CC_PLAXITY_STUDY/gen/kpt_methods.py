"""Round G: KP-T, Kalman plaxity with an OS-telemetry prior ("contention is observable state").

Telemetry is read at each frame release by lidar_ros_tm (no root):
  /proc/stat cpu line (USER_HZ ticks)  -> share of CPU time used by OTHER processes since the previous release
                                          u_t = (d(user+system+irq+softirq+steal) - d(own utime+stime)) / d(all ticks)
                                          (spinners run SCHED_IDLE + nice 19, so they land in 'nice' and are excluded)
  /proc/stat procs_running             -> runnable tasks right now (r_t)
  /proc/self/task/*/schedstat          -> run-queue delay suffered by our own threads since the previous release (w_t, ms)
KP-T prior for frame t (log-scale state z_t = log(e2e_t / median)):
  mu_t = a + b * m_t + c * u_t + d * log1p(r_t) + e * log1p(w_t)     (OLS on calibration frames; m_t = KP's EWMA prior)
  q_t  = residual variance of that regression
Everything after the prior (in-frame Gaussian update, remaining-time quantiles B_k) is KP unchanged.
Telemetry deltas need the previous release, so the first frame of each chunk falls back to KP's prior.
"""
import numpy as np, pandas as pd
from kp_methods import KalmanPlaxity, _ewma, NQ

TM_COLS = ["tm_u", "tm_r", "tm_w"]


def telemetry_features(raw):
    """raw: lidar_ros_tm v3 CSV rows in frame order (one run). Counters are snapshots taken by a background sampler every
    20 ms; deltas are between consecutive releases. Returns tm_u, tm_r, tm_w (NaN on the first frame).
      tm_u = share of all CPU time used by OTHER processes = (d stat_busy - d own_cpu) / d(total ticks)
             (stat_* are USER_HZ=100 ticks; own_cpu_us / 1e4 converts to ticks; spinners are in stat_nice and excluded)
      tm_r = runnable tasks at the snapshot (/proc/loadavg)
      tm_w = involuntary context switches (preemptions) of our process since the previous release (getrusage)"""
    total = raw[["stat_busy", "stat_nice", "stat_idle", "stat_steal"]].sum(1).diff()
    u = ((raw.stat_busy.diff() - raw.own_cpu_us.diff() / 1e4) / total.replace(0, np.nan)).clip(0, 1)
    return pd.DataFrame({"tm_u": u.values, "tm_r": raw.procs_running.values.astype(float),
                         "tm_w": raw.nivcsw.diff().clip(lower=0).values})


class KalmanPlaxityT(KalmanPlaxity):
    def __init__(self, cal, seed=0, nq=NQ, features=("m", "u", "r", "w")):
        super().__init__(cal, seed=seed, nq=nq)
        self.features = features
        X, z, ok = self._design(cal)
        A = np.column_stack([np.ones(ok.sum()), X[ok]])
        self.coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
        r = z[ok] - A @ self.coef
        self.qt = float(np.var(r))
        self.r2_prior = 1 - self.qt / float(np.var(z[ok]))

    def _design(self, df):
        z = np.log(df.e2e.values / self.me2e)
        m = _ewma(z, df.has_prev.values, self.lam)
        cols = {"m": m, "u": df.tm_u.values, "r": np.log1p(df.tm_r.values), "w": np.log1p(df.tm_w.values)}
        X = np.column_stack([cols[f] for f in self.features])
        ok = np.isfinite(X).all(1)
        return X, z, ok

    def prior(self, df):
        m0, q0 = super().prior(df)
        X, _, ok = self._design(df)
        mu = np.where(ok, np.column_stack([np.ones(len(df)), np.nan_to_num(X)]) @ self.coef, m0)
        q = np.where(ok, self.qt, q0)
        return mu, q
