"""Extended methods for the reviewer-hardening round (PROTOCOL_GENERALIZATION.md, Addendum D).

Stronger simple baselines, all fit on calibration frames only, all causal (use only frames before t):
  AR1      AR(1) on log e2e: y_t = a + phi*y_{t-1} + eps, eps = empirical calib residuals.
  EWMA     exponentially weighted mean of past log e2e (lambda picked on calib MSE from a fixed grid), empirical residuals.
  LR-iso   deadline-conditional LR (cc_transfer_lr.LRdeadline) + isotonic recalibration fit on 5-fold
           time-blocked out-of-fold calib predictions.
  Naive    NaiveScale (g_analysis.naive_scale), extended to every checkpoint by scaling calib remaining-time samples.
Self-checking monitor:
  CC-SC    CC-plaxity, but for frame t use Naive whenever CC's own release-time PIT values on the last W frames
           are out of range: u_j = P_CC(e2e_j > observed e2e_j) at frame j's release; count of u_j < 0.05 or
           u_j > 0.95 among the last W = 20 frames (same contiguous chunk) >= 7.  Under calibration the count is
           Binomial(20, 0.10), so P(count >= 7) ~= 0.002 (false-switch rate). Constants fixed a priori, never tuned.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cc_plaxity import CP_DONE, NODES
from cc_transfer_lr import LRdeadline

SC_W, SC_TAIL, SC_K = 20, 0.05, 7


def _chunks(df):
    return df.chunk.values, df.has_prev.values


class AR1:
    def __init__(self, cal):
        y = np.log(cal.e2e.values); hp = cal.has_prev.values
        x, z = y[:-1][hp[1:]], y[1:][hp[1:]]
        self.phi, self.a = np.polyfit(x, z, 1)
        self.res = np.sort(z - (self.a + self.phi * x))
        self.marg = np.sort(y)

    def predict(self, df, D):
        y = np.log(df.e2e.values); hp = df.has_prev.values
        mu = np.where(hp, self.a + self.phi * np.r_[np.nan, y[:-1]], np.nan)
        p = np.empty(len(df))
        for i, m in enumerate(mu):
            p[i] = (1 - np.searchsorted(self.marg, np.log(D), "right") / len(self.marg)) if np.isnan(m) else \
                   (1 - np.searchsorted(self.res, np.log(D) - m, "right") / len(self.res))
        return p


class EWMA:
    GRID = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

    def __init__(self, cal):
        best = None
        for lam in self.GRID:
            m = self._means(cal, lam, np.median(np.log(cal.e2e)))
            r = np.log(cal.e2e.values) - m
            ok = cal.has_prev.values
            mse = np.mean(r[ok] ** 2)
            if best is None or mse < best[0]:
                best = (mse, lam, np.sort(r[ok]))
        _, self.lam, self.res = best
        self.y0 = np.median(np.log(cal.e2e))
        self.marg = np.sort(np.log(cal.e2e.values))

    @staticmethod
    def _means(df, lam, y0):
        y = np.log(df.e2e.values); hp = df.has_prev.values
        m = np.empty(len(df)); cur = None
        for i in range(len(df)):
            if not hp[i]:
                cur = None
            m[i] = np.nan if cur is None else cur
            cur = y[i] if cur is None else lam * cur + (1 - lam) * y[i]
        return m

    def predict(self, df, D):
        m = self._means(df, self.lam, self.y0)
        return np.array([(1 - np.searchsorted(self.marg, np.log(D), "right") / len(self.marg)) if np.isnan(mi) else
                         (1 - np.searchsorted(self.res, np.log(D) - mi, "right") / len(self.res)) for mi in m])


class LRIso:
    def __init__(self, cal, Ds, D, folds=5):
        n = len(cal); f = np.arange(n) * folds // n
        oof = {k: np.empty(n) for k in CP_DONE}
        for j in range(folds):
            tr, te = f != j, f == j
            m = LRdeadline(cal[tr].reset_index(drop=True), Ds)
            p = m.predict(cal[te].reset_index(drop=True), D)
            for k in CP_DONE:
                oof[k][te] = p[k]
        y = (cal.e2e > D).values.astype(int)
        self.iso = {k: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[k], y) for k in CP_DONE}
        self.lr, self.D = LRdeadline(cal, Ds), D

    def predict(self, df):
        p = self.lr.predict(df, self.D)
        return {k: self.iso[k].predict(p[k]) for k in CP_DONE}


class NaiveCP:
    """NaiveScale at every checkpoint: scale calib remaining-time samples by prev_e2e / median calib e2e."""
    def __init__(self, cal):
        self.med = np.median(cal.e2e.values)
        self.R = {k: np.sort(cal[[c for c in NODES if c not in done]].sum(1).values) for k, done in CP_DONE.items()}

    def scale(self, df):
        return (df.prev_e2e / self.med).fillna(1.0).values

    def predict(self, df, D):
        s = self.scale(df)
        return {k: 1.0 - np.searchsorted(self.R[k], (D - df["e%d" % k].values) / s, side="right") / len(self.R[k])
                for k in CP_DONE}


def sc_flags(u, df):
    """True at frame t if the PIT check on frames before t (same chunk) is out of range."""
    out = (u < SC_TAIL) | (u > 1 - SC_TAIL)
    flag = np.zeros(len(u), bool); hist = []
    hp = df.has_prev.values
    for t in range(len(u)):
        if not hp[t]:
            hist = []
        flag[t] = sum(hist[-SC_W:]) >= SC_K
        hist.append(bool(out[t]))
    return flag


def cc_pit(cc, df):
    """u_t = P_CC(e2e_t > observed e2e_t) at release (the frame's own e2e is not used by the release prior)."""
    return cc.predict(df, df.e2e.values, True)[0]


def cc_sc(cc, naive, df, D):
    p_cc = cc.predict(df, D, True); p_nv = naive.predict(df, D)
    flag = sc_flags(cc_pit(cc, df), df)
    return {k: np.where(flag, p_nv[k], p_cc[k]) for k in CP_DONE}, flag
