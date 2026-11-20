"""Addendum J baselines: standard learned / classical monitors that a reviewer would ask for.
All are fit on calibration frames only and are causal (frame t uses only what is known at its checkpoint k:
completed node durations, elapsed time, past frames' e2e, and release-time telemetry). One model per checkpoint.

  Budget   classical elapsed-time watchdog: P(miss | elapsed e_k) by isotonic regression on calibration
           (k = 0 has no elapsed information -> base rate).
  LR-all   logistic regression (standardised, L2, C=1) on all features below; NaNs imputed with calib median + indicator.
  GBDT-all HistGradientBoostingClassifier on the same features (NaN handled natively). Hyper-parameters chosen per
           checkpoint by 5-fold time-blocked CV log loss on calibration from a small fixed grid (GBDT_GRID).

Features at checkpoint k: log elapsed e_k (k>0), log of each completed node duration, log prev e2e,
EWMA of past log e2e (lambda 0.8, fixed), mean log e2e of the last 5 frames, telemetry u, log1p r, log1p w.
For the ACI threshold the learned baselines use OUT-OF-FOLD calibration predictions (in-sample GBDT
probabilities would be over-confident and give an unfair theta_0).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cc_plaxity import CP_DONE
from kp_methods import _ewma

FOLDS = 5
LAM = 0.8
GBDT_GRID = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=lf, min_samples_leaf=20, l2_regularization=1.0)
             for lr in (0.05, 0.1) for it in (100, 300) for lf in (15, 31)]


def features(df, k):
    hp = df.has_prev.values
    le = np.log(df.e2e.values)
    prev = np.where(hp, np.r_[np.nan, le[:-1]], np.nan)
    ew = _ewma(le, hp, LAM)
    last5 = pd.Series(le).groupby(df.chunk.values).transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values
    cols = {"prev": prev, "ewma": ew, "last5": last5,
            "u": df.tm_u.values, "r": np.log1p(df.tm_r.values), "w": np.log1p(df.tm_w.values)}
    for c in CP_DONE[k]:
        cols["d_" + c] = np.log(np.maximum(df[c].values, 1e-3))
    if k > 0:
        cols["elapsed"] = np.log(np.maximum(df["e%d" % k].values, 1e-3))
    return pd.DataFrame(cols)


def _blocks(n):
    return np.arange(n) * FOLDS // n


class _PerCheckpoint:
    """Base: fit(X, y) per checkpoint; out-of-fold calibration predictions for theta_0."""
    def __init__(self, cal, D):
        y = (cal.e2e > D).values.astype(int); f = _blocks(len(cal))
        self.models, self.oof = {}, {}
        for k in CP_DONE:
            X = features(cal, k)
            self.models[k] = self._fit(X, y, k, f)
            o = np.empty(len(cal))
            for j in range(FOLDS):
                tr, te = f != j, f == j
                o[te] = self._proba(self._fit(X[tr].reset_index(drop=True), y[tr], k, None, self.models[k]), X[te])
            self.oof[k] = o

    def predict(self, df):
        return {k: self._proba(self.models[k], features(df, k)) for k in CP_DONE}

    def cal_oof(self):
        return np.column_stack([self.oof[k] for k in sorted(CP_DONE)])


class Budget(_PerCheckpoint):
    def _fit(self, X, y, k, f, ref=None):
        if k == 0:
            return ("const", float(y.mean()))
        return ("iso", IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(X.elapsed.values, y))

    def _proba(self, m, X):
        return np.full(len(X), m[1]) if m[0] == "const" else m[1].predict(X.elapsed.values)


class LRAll(_PerCheckpoint):
    def _fit(self, X, y, k, f, ref=None):
        med = X.median()
        Z = self._prep(X, med)
        sc = StandardScaler().fit(Z)
        return (med, sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Z), y))

    @staticmethod
    def _prep(X, med):
        ind = X.isna().astype(float).add_suffix("_na")
        return pd.concat([X.fillna(med), ind], axis=1).fillna(0.0).values

    def _proba(self, m, X):
        med, sc, lr = m
        return lr.predict_proba(sc.transform(self._prep(X, med)))[:, 1]


class GBDTAll(_PerCheckpoint):
    def _fit(self, X, y, k, f, ref=None):
        if ref is not None:                       # OOF refit with the hyper-parameters chosen on full calibration
            p = ref.get_params()
            return HistGradientBoostingClassifier(random_state=0, **{q: p[q] for q in GBDT_GRID[0]}).fit(X, y)
        best = None
        for g in GBDT_GRID:
            ll = []
            for j in range(FOLDS):
                tr, te = f != j, f == j
                if y[tr].min() == y[tr].max():
                    continue
                m = HistGradientBoostingClassifier(random_state=0, **g).fit(X[tr], y[tr])
                ll.append(log_loss(y[te], np.clip(m.predict_proba(X[te])[:, 1], 1e-4, 1 - 1e-4), labels=[0, 1]))
            s = np.mean(ll)
            if best is None or s < best[0]:
                best = (s, g)
        self.chosen = getattr(self, "chosen", {}); self.chosen[k] = best[1]
        return HistGradientBoostingClassifier(random_state=0, **best[1]).fit(X, y)

    def _proba(self, m, X):
        return m.predict_proba(X)[:, 1]
