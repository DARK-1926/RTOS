"""KPF: Kalman plaxity with an ONLINE-FUSED prior (exploratory).

Two experts predict the log-scale state z_t before frame t is released:
  H (history):   KP's EWMA of past frames                   -> mu_H,t
  T (telemetry): OLS on OS telemetry (u, log1p r, log1p w)  -> mu_T,t   (KPT's prior)
Each expert's recent accuracy is tracked online with exponential forgetting rho, from frames that have already completed:
  v_E,t = rho v_E,t-1 + (1 - rho) (z_{t-1} - mu_E,t-1)^2
and the prior is their precision-weighted fusion   mu_F,t = (mu_H / v_H + mu_T / v_T) / (1 / v_H + 1 / v_T).
Its variance is ALSO tracked online from the fused prior's own recent errors:  q_F,t = rho q_F,t-1 + (1 - rho)(z_{t-1} - mu_F,t-1)^2,
so the prior widens by itself when contention changes beyond anything seen in calibration.
Why: which source is informative depends on the pipeline (scene-driven latency -> history; contention-driven -> telemetry)
and on the moment (history is stale for a few frames after a contention switch). A fixed OLS weighting fitted mostly on
steady-state frames cannot do both. rho is chosen on calibration frames only (Gaussian NLL of the prior); v and q start from
the calibration MSEs and reset to them at chunk starts. Everything after the prior is KP unchanged.
Causality: frame t's prior uses z only up to frame t-1, the same convention as KP's EWMA.
"""
import numpy as np
from kp_methods import KalmanPlaxity
from kpt_methods import KalmanPlaxityT

RHO_GRID = [0.5, 0.7, 0.8, 0.9, 0.95]
Q_FLOOR = 1e-4


class KalmanPlaxityF(KalmanPlaxity):
    def __init__(self, cal, seed=0, nq=64, rho=None):
        super().__init__(cal, seed=seed, nq=nq)
        self.tel = KalmanPlaxityT(cal, seed=seed, nq=8, features=("u", "r", "w"))   # only its prior is used
        z = np.log(cal.e2e.values / self.me2e)
        mH, _ = KalmanPlaxity.prior(self, cal); mT, _ = self.tel.prior(cal)
        self.vH0 = float(np.mean((z - mH) ** 2)); self.vT0 = float(np.mean((z - mT) ** 2))
        wH = (1 / self.vH0) / (1 / self.vH0 + 1 / self.vT0)
        self.qF0 = float(np.mean((z - (wH * mH + (1 - wH) * mT)) ** 2))
        if rho is None:
            best = None
            for r in RHO_GRID:
                mu, q = self._fuse(z, mH, mT, cal.has_prev.values, r)
                nll = np.mean(0.5 * np.log(q) + 0.5 * (z - mu) ** 2 / q)
                if best is None or nll < best[0]:
                    best = (nll, r)
            rho = best[1]
        self.rho = rho

    def _fuse(self, z, mH, mT, hp, rho):
        n = len(z); mu = np.empty(n); q = np.empty(n)
        vH, vT, qF = self.vH0, self.vT0, self.qF0
        for t in range(n):
            if not hp[t]:
                vH, vT, qF = self.vH0, self.vT0, self.qF0
            a, b = 1.0 / max(vH, Q_FLOOR), 1.0 / max(vT, Q_FLOOR)
            mu[t] = (a * mH[t] + b * mT[t]) / (a + b); q[t] = max(qF, Q_FLOOR)
            vH = rho * vH + (1 - rho) * (z[t] - mH[t]) ** 2          # z_t becomes known before frame t+1's prior
            vT = rho * vT + (1 - rho) * (z[t] - mT[t]) ** 2
            qF = rho * qF + (1 - rho) * (z[t] - mu[t]) ** 2
        return mu, q

    def prior(self, df):
        z = np.log(df.e2e.values / self.me2e)
        mH, _ = KalmanPlaxity.prior(self, df); mT, _ = self.tel.prior(df)
        return self._fuse(z, mH, mT, df.has_prev.values, self.rho)
