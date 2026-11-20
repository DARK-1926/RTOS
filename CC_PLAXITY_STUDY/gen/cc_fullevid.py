"""CC-E: CC-plaxity whose cross-frame (temporal) update uses ALL completed nodes of the previous frame
(n1, n2, F, n5, ovh) as evidence for sigma, instead of only (n1, n2, F). Within-frame checkpoint updates are unchanged.
Registered in PROTOCOL_GENERALIZATION.md Addendum C before seeds 53-55 were analyzed."""
import os, sys
import numpy as np
from scipy.stats import gaussian_kde
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cc_plaxity import CCPlaxity, CP_DONE, GRID

ALL = ["n1", "n2", "F", "n5", "ovh"]


class CCFullEvidence(CCPlaxity):
    def __init__(self, calib, nominal_regime):
        super().__init__(calib, nominal_regime)
        idle = self.idle
        for c in ("n5", "ovh"):
            k = gaussian_kde(np.log(idle[c].values))
            zg = np.linspace(-8, 8, 4001); fg = k(zg)
            self.kde[c] = (lambda zz, zg=zg, fg=fg: np.interp(zz, zg, fg, left=0.0, right=0.0))
        L = self.loglik(calib, ALL)
        post = np.exp(L - L.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        pi = post.mean(0) + 1e-6; self.pi = pi / pi.sum()
        best = None
        for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
            ll = self._forward(calib, rho, loglik_full=L)[1]
            if best is None or ll > best[1]:
                best = (rho, ll)
        self.rho = best[0]

    def predict(self, df, D, temporal=True):
        Lfull = self.loglik(df, ALL)
        priors = self._forward(df, self.rho, Lfull)[0] if temporal else np.tile(self.pi, (len(df), 1))
        out = {}
        for k, done in CP_DONE.items():
            post = priors * (np.exp(self.loglik(df, list(done))) if done else 1.0)
            post /= post.sum(1, keepdims=True)
            lax = D - df["e%d" % k].values; M = self.R[k].shape[1]
            surv = np.empty((len(df), len(GRID)))
            for g in range(len(GRID)):
                surv[:, g] = 1.0 - np.searchsorted(self.R[k][g], lax, side="right") / M
            out[k] = np.clip((post * surv).sum(1), 0, 1)
        return out
