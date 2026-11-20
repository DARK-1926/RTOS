"""CC-R: CC-plaxity with a tail-robust reset target (PROTOCOL_GENERALIZATION.md, Addendum A).
pi_R = (1 - eps) * pi + eps * Uniform(GRID); rho re-selected by calib likelihood under pi_R. Nothing else changes."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cc_plaxity import CCPlaxity, GRID, EVID


class CCRobust(CCPlaxity):
    def __init__(self, calib, nominal_regime, eps=0.05):
        super().__init__(calib, nominal_regime)
        self.pi = (1 - eps) * self.pi + eps / len(GRID)
        self.pi /= self.pi.sum()
        L = self.loglik(calib, EVID)
        best = None
        for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
            ll = self._forward(calib, rho, loglik_full=L)[1]
            if best is None or ll > best[1]:
                best = (rho, ll)
        self.rho = best[0]
