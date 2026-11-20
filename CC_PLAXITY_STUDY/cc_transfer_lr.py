import numpy as np
from sklearn.linear_model import LogisticRegression
from cc_plaxity import CP_DONE, feats


class LRdeadline:
    """deadline-conditional LR: features + log(D - e_k) (clipped), trained over several deadlines."""
    def __init__(self, calib, Ds):
        self.med = calib[["e2e", "n1", "n2"]].median()
        self.m = {}
        for k in CP_DONE:
            X, y = [], []
            for D in Ds:
                X.append(self._x(calib, k, D)); y.append((calib.e2e > D).values.astype(int))
            self.m[k] = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack(X), np.concatenate(y))

    def _x(self, df, k, D):
        base = feats(df, k, self.med)
        lax = np.log(np.clip(D - df["e%d" % k].values, 0.1, None))
        return np.column_stack([base, lax, np.full(len(df), np.log(D))])

    def predict(self, df, D):
        return {k: self.m[k].predict_proba(self._x(df, k, D))[:, 1] for k in CP_DONE}


