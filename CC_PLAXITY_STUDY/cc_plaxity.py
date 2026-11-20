"""Contention-Coupled Plaxity (CC-plaxity) vs Toba-2026-style independence plaxity.

See PROTOCOL_CC_PLAXITY.md. Run: python cc_plaxity.py <data_dir> <out_dir>
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20261106)
NODES = ["n1", "n2", "F", "n5", "ovh"]          # execution order; F = max(N3, N4) super-node
EVID = ["n1", "n2", "F"]                        # nodes used as evidence for sigma
CP_DONE = {0: [], 1: ["n1"], 2: ["n1", "n2"], 3: ["n1", "n2", "F"]}
GRID = np.linspace(np.log(0.5), np.log(8.0), 121)   # sigma = log S


# ---------------------------------------------------------------- data adapters


def load_lidar(dd):
    d = pd.read_csv(f"{dd}/multinode_real_lidar_traces.csv")
    d = d[d.split.isin(["calib", "val"])].sort_values("global_frame_id")
    o = pd.DataFrame({
        "n1": d.dur_node1_ms, "n2": d.dur_node2_ms,
        "F": np.maximum(d.dur_node3_ms, d.dur_node4_ms), "n5": d.dur_node5_ms,
        "e2e": d.end_to_end_ms, "regime": d.regime, "split": d.split,
        "order": d.global_frame_id, "cluster": d.bag_frame_idx}).reset_index(drop=True)
    return o, "idle"


def load_pilot(dd):
    d = pd.read_csv(f"{dd}/pilot_finer_contention_traces.csv").sort_values("pilot_frame_id")
    rank = d.groupby("level").cumcount()
    n = d.groupby("level").pilot_frame_id.transform("size")
    o = pd.DataFrame({
        "n1": d.dur_node1_ms, "n2": d.dur_node2_ms,
        "F": np.maximum(d.dur_node3_ms, d.dur_node4_ms), "n5": d.dur_node5_ms,
        "e2e": d.total_dag_ms, "regime": d.level,
        "split": np.where(rank < 0.6 * n, "calib", "val"),
        "order": d.pilot_frame_id, "cluster": d.cloud_idx}).reset_index(drop=True)
    return o, "C0"


def load_hclock(dd):
    d = pd.read_csv(f"{dd}/h_clock_trials.csv").sort_values(["session_id", "trial_idx"])
    o = pd.DataFrame({
        "n1": d.d1, "n2": d.d2, "F": np.maximum(d.d3, d.d4), "n5": d.d5, "e2e": d.total_ms,
        "regime": d.regime, "split": np.where(d.session_id <= 5, "calib", "val"),
        "order": np.arange(len(d)), "cluster": d.bag_frame_idx,
        "session": d.session_id}).reset_index(drop=True)
    return o, "idle"


def prep(o):
    o = o.copy()
    o["ovh"] = np.clip(o.e2e - (o.n1 + o.n2 + o.F + o.n5), 1e-3, None)
    for c in ["n1", "n2", "F", "n5"]:
        o[c] = np.clip(o[c], 1e-3, None)
    # temporal chunks: consecutive order ids within the same split (and session if present)
    key = o.split + (o["session"].astype(str) if "session" in o else "")
    brk = (o.order.diff() != 1) | (key != key.shift())
    o["chunk"] = brk.cumsum()
    o["has_prev"] = ~brk
    for c in ["n1", "n2", "e2e"]:
        o["prev_" + c] = o[c].shift(1).where(o.has_prev)
    o["e1"] = o.n1
    o["e2"] = o.n1 + o.n2
    o["e3"] = o.n1 + o.n2 + o.F
    o["e0"] = 0.0
    return o


# ---------------------------------------------------------------- CC-plaxity
class CCPlaxity:
    def __init__(self, calib, nominal_regime):
        idle = calib[calib.regime == nominal_regime]
        self.idle = idle
        # loadings beta_i: slope of log median ratio vs e2e log median ratio across regimes
        med = calib.groupby("regime")[NODES + ["e2e"]].median()
        lr = np.log(med / med.loc[nominal_regime])
        x = lr["e2e"].values
        self.beta = {}
        for c in NODES:
            y = lr[c].values
            self.beta[c] = float(np.dot(x, y) / max(np.dot(x, x), 1e-9)) if np.dot(x, x) > 1e-6 else 1.0
        # evidence densities (log domain KDE of idle durations)
        # KDE tabulated on a fine log grid, then linearly interpolated (fast lookup)
        self.kde = {}
        for c in EVID:
            k = gaussian_kde(np.log(idle[c].values))
            zg = np.linspace(-8, 8, 4001)
            fg = k(zg)
            self.kde[c] = (lambda zz, zg=zg, fg=fg: np.interp(zz, zg, fg, left=0.0, right=0.0))
        # remaining-time sample tables per checkpoint and sigma
        Y = idle[NODES].values  # joint idle samples
        self.R = {}
        for k, done in CP_DONE.items():
            rem = [j for j, c in enumerate(NODES) if c not in done]
            scale = np.exp(np.outer(GRID, [self.beta[NODES[j]] for j in rem]))  # G x r
            self.R[k] = np.sort(np.einsum("gr,mr->gm", scale, Y[:, rem]), axis=1)   # G x M
        # population prior from per-frame posteriors on calib (flat start)
        L = self.loglik(calib, EVID)
        post = np.exp(L - L.max(1, keepdims=True)); post /= post.sum(1, keepdims=True)
        pi = post.mean(0) + 1e-6
        self.pi = pi / pi.sum()
        # sticky-HMM rho by calib likelihood
        best = None
        for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
            ll = self._forward(calib, rho, loglik_full=L)[1]
            if best is None or ll > best[1]:
                best = (rho, ll)
        self.rho = best[0]

    def loglik(self, df, nodes):
        L = np.zeros((len(df), len(GRID)))
        for c in nodes:
            z = np.log(df[c].values)[:, None] - self.beta[c] * GRID[None, :]
            L += np.log(self.kde[c](z.ravel()).reshape(z.shape) + 1e-300)
        return L

    def _forward(self, df, rho, loglik_full):
        """Returns per-frame prior (before observing the frame) and total log-likelihood."""
        n = len(df)
        priors = np.zeros((n, len(GRID)))
        prev_post = None
        tot = 0.0
        hp = df.has_prev.values
        for t in range(n):
            if not hp[t] or prev_post is None or rho == 0.0:
                pr = self.pi.copy()
            else:
                pr = rho * prev_post + (1 - rho) * self.pi
            priors[t] = pr
            l = loglik_full[t]
            m = l.max()
            w = pr * np.exp(l - m)
            s = w.sum()
            tot += np.log(s) + m
            prev_post = w / s
        return priors, tot

    def predict(self, df, D, temporal=True):
        Lfull = self.loglik(df, EVID)
        if temporal:
            priors, _ = self._forward(df, self.rho, Lfull)
        else:
            priors = np.tile(self.pi, (len(df), 1))
        out = {}
        for k, done in CP_DONE.items():
            post = priors * np.exp(self.loglik(df, [c for c in done]) if done else 0.0)
            post /= post.sum(1, keepdims=True)
            lax = D - df["e%d" % k].values          # n
            # survival P(R > lax | sigma) for each sigma: searchsorted per grid row
            M = self.R[k].shape[1]
            surv = np.empty((len(df), len(GRID)))
            for g in range(len(GRID)):
                surv[:, g] = 1.0 - np.searchsorted(self.R[k][g], lax, side="right") / M
            out[k] = np.clip((post * surv).sum(1), 0, 1)
        return out


# ---------------------------------------------------------------- independence plaxity (Toba-style)
class IndepPlaxity:
    """P(sum of independently drawn remaining nodes > D - e_k): Monte Carlo convolution."""
    def __init__(self, ref, nmc=40000):
        self.R = {}
        for k, done in CP_DONE.items():
            rem = [c for c in NODES if c not in done]
            s = np.zeros(nmc)
            for c in rem:
                s += RNG.choice(ref[c].values, nmc)
            self.R[k] = np.sort(s)

    def predict(self, df, D):
        return {k: 1.0 - np.searchsorted(self.R[k], D - df["e%d" % k].values, side="right") / len(self.R[k])
                for k in CP_DONE}


class OraclePlaxity:
    def __init__(self, calib):
        self.m = {r: IndepPlaxity(g) for r, g in calib.groupby("regime")}

    def predict(self, df, D):
        out = {k: np.zeros(len(df)) for k in CP_DONE}
        for r, idx in df.groupby("regime").groups.items():
            p = self.m[r].predict(df.loc[idx], D)
            for k in CP_DONE:
                out[k][df.index.get_indexer(idx)] = p[k]
        return out


# ---------------------------------------------------------------- learned baselines
def feats(df, k, med, persistence_only=False):
    pe = df.prev_e2e.fillna(med["e2e"]).values
    X = [np.log(pe), (~df.has_prev).astype(float).values]
    if not persistence_only:
        X += [np.log(df.prev_n1.fillna(med["n1"]).values), np.log(df.prev_n2.fillna(med["n2"]).values)]
    if k >= 1:
        X.append(np.log(df["e%d" % k].values))
    if k >= 2 and not persistence_only:
        X += [np.log(df.n1.values), np.log(df.n2.values)]
    if k >= 3 and not persistence_only:
        X.append(np.log(df.F.values))
    return np.column_stack(X)


class LR:
    def __init__(self, calib, D, persistence_only=False):
        self.med = calib[["e2e", "n1", "n2"]].median()
        y = (calib.e2e > D).astype(int).values
        self.po = persistence_only
        self.m = {}
        for k in CP_DONE:
            if y.min() == y.max():
                self.m[k] = float(y.mean())
                continue
            self.m[k] = LogisticRegression(C=1.0, max_iter=2000).fit(feats(calib, k, self.med, self.po), y)

    def predict(self, df):
        out = {}
        for k in CP_DONE:
            m = self.m[k]
            out[k] = np.full(len(df), m) if isinstance(m, float) else m.predict_proba(feats(df, k, self.med, self.po))[:, 1]
        return out


# ---------------------------------------------------------------- metrics
def ece(p, y, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def auroc(p, y):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def pipeline(scores, df, D, theta):
    """theta: dict cp->threshold. alarm at first cp with p>theta. returns alarm cp (or -1)."""
    n = len(df)
    first = np.full(n, -1)
    for k in sorted(CP_DONE):
        hit = (scores[k] > theta[k]) & (first < 0)
        first[hit] = k
    t_alarm = np.where(first >= 0, df[["e0", "e1", "e2", "e3"]].values[np.arange(n), np.maximum(first, 0)], np.nan)
    return first, D - t_alarm


def tune_uniform_theta(scores, y, target=0.90):
    """single theta for all cps (as in Toba-2024 single threshold) hitting target recall on calib."""
    cand = np.unique(np.concatenate([scores[k] for k in CP_DONE]))
    cand = np.quantile(cand, np.linspace(0, 1, 400))
    best = None
    for th in cand[::-1]:
        alarm = np.zeros(len(y), bool)
        for k in CP_DONE:
            alarm |= scores[k] > th
        rec = alarm[y == 1].mean() if y.sum() else np.nan
        if rec >= target:
            return th
    return cand[0]


def summarize(name, sc, df, y, D, theta_u):
    rows = []
    for k in CP_DONE:
        rows.append(dict(method=name, D=D, cp=k, auroc=auroc(sc[k], y), brier=np.mean((sc[k] - y) ** 2),
                         ece=ece(sc[k], y), mean_p=sc[k].mean(), base_rate=y.mean()))
    th = {k: theta_u for k in CP_DONE}
    first, lead = pipeline(sc, df, D, th)
    alarm = first >= 0
    tp = alarm & (y == 1)
    # toba-2026 style linear variable thresholds, untuned
    tv = {0: 0.5, 1: 0.5 + (0.99 - 0.5) / 3, 2: 0.5 + 2 * (0.99 - 0.5) / 3, 3: 0.99}
    f2, l2 = pipeline(sc, df, D, tv)
    a2 = f2 >= 0
    pipe = dict(method=name, D=D, theta=theta_u,
                recall=alarm[y == 1].mean(), fpr=alarm[y == 0].mean(),
                lead_ms_tp=np.nanmean(lead[tp]) if tp.any() else np.nan,
                alarm_at_cp0=(first[tp] == 0).mean() if tp.any() else np.nan,
                alarm_at_cp1=(first[tp] == 1).mean() if tp.any() else np.nan,
                vt_recall=a2[y == 1].mean(), vt_fpr=a2[y == 0].mean(),
                vt_lead_ms_tp=np.nanmean(l2[a2 & (y == 1)]) if (a2 & (y == 1)).any() else np.nan)
    return rows, pipe


def block_boot_diff(pa, pb, y, blocks, metric, B=1000):
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    d = []
    for _ in range(B):
        s = np.concatenate([idx[b] for b in RNG.choice(ub, len(ub))])
        yy = y[s]
        if 0 < yy.sum() < len(yy):
            d.append(metric(pa[s], yy) - metric(pb[s], yy))
    d = np.array(d)
    return np.mean(d), np.quantile(d, 0.025), np.quantile(d, 0.975)


def run(dataset, dd, Ds, outdir):
    loader = {"lidar": load_lidar, "pilot": load_pilot, "hclock": load_hclock}[dataset]
    o, nominal = loader(dd)
    o = prep(o)
    cal = o[o.split == "calib"].reset_index(drop=True)
    val = o[o.split == "val"].reset_index(drop=True)
    cc = CCPlaxity(cal, nominal)
    info = dict(dataset=dataset, beta=cc.beta, rho=cc.rho, n_calib=len(cal), n_val=len(val))
    tp_ = IndepPlaxity(cal)
    ti_ = IndepPlaxity(cal[cal.regime == nominal])
    orc = OraclePlaxity(cal)
    blocks = (np.arange(len(val)) // 50)
    allrows, pipes, boots = [], [], []
    for D in Ds:
        yv = (val.e2e > D).astype(int).values
        yc = (cal.e2e > D).astype(int).values
        lr = LR(cal, D)
        pers = LR(cal, D, persistence_only=True)
        meths = {
            "CC-plaxity": (lambda df: cc.predict(df, D, True)),
            "CC-within (no temporal)": (lambda df: cc.predict(df, D, False)),
            "Toba-pooled (indep)": (lambda df: tp_.predict(df, D)),
            "Toba-idle (indep)": (lambda df: ti_.predict(df, D)),
            "Toba-oracle-regime": (lambda df: orc.predict(df, D)),
            "Persistence+elapsed LR": pers.predict,
            "Logistic regression": lr.predict,
        }
        vs = {}
        for name, f in meths.items():
            sc_c = f(cal)
            th = tune_uniform_theta(sc_c, yc, 0.90)
            sc_v = f(val)
            vs[name] = sc_v
            rows, pipe = summarize(name, sc_v, val, yv, D, th)
            for r in rows:
                r["dataset"] = dataset
            pipe["dataset"] = dataset
            allrows += rows
            pipes.append(pipe)
            # per-regime breakdown at cp1
            for reg, g in val.groupby("regime"):
                ii = g.index.values
                allrows.append(dict(dataset=dataset, method=name, D=D, cp="1@" + str(reg),
                                    auroc=auroc(sc_v[1][ii], yv[ii]), brier=np.mean((sc_v[1][ii] - yv[ii]) ** 2),
                                    ece=ece(sc_v[1][ii], yv[ii]), mean_p=sc_v[1][ii].mean(), base_rate=yv[ii].mean()))
        for comp in ["Toba-pooled (indep)", "Logistic regression", "Persistence+elapsed LR", "CC-within (no temporal)"]:
            for k in CP_DONE:
                for mname, mf in [("auroc", auroc), ("brier", lambda p, y: np.mean((p - y) ** 2))]:
                    m, lo, hi = block_boot_diff(vs["CC-plaxity"][k], vs[comp][k], yv, blocks, mf, B=400)
                    boots.append(dict(dataset=dataset, D=D, cp=k, metric=mname, comparator=comp,
                                      diff_cc_minus_comp=m, ci_lo=lo, ci_hi=hi))
        np.savez_compressed(f"{outdir}/{dataset}_scores_D{D}.npz", y=yv,
                            **{f"{n}|{k}": v[k] for n, v in vs.items() for k in v},
                            e2e=val.e2e.values, regime=val.regime.astype(str).values)
    return pd.DataFrame(allrows), pd.DataFrame(pipes), pd.DataFrame(boots), info


def run_unoq(board_dir, outdir):
    """Frozen confirmatory analysis for unoq_cc_confirm.py traces (one CSV per seed)."""
    import glob
    res = []
    for fn in sorted(glob.glob(f"{board_dir}/*.csv")):
        d = pd.read_csv(fn).iloc[20:]                       # drop warm-up frames
        o = pd.DataFrame({"n1": d.dur_node1_ms, "n2": d.dur_node2_ms,
                          "F": np.maximum(d.dur_node3_ms, d.dur_node4_ms), "n5": d.dur_node5_ms,
                          "e2e": d.end_to_end_ms, "regime": d.regime, "split": d.split,
                          "order": d.global_frame_id, "cluster": d.bag_frame_idx}).reset_index(drop=True)
        o = prep(o)
        cal = o[o.split == "calib"].reset_index(drop=True); val = o[o.split == "val"].reset_index(drop=True)
        D = float(np.round(np.quantile(cal.e2e, 0.85)))
        y = (val.e2e > D).astype(int).values
        cc = CCPlaxity(cal, "K0"); ind = IndepPlaxity(cal)
        p_cc, p_ccw, p_ind = cc.predict(val, D, True), cc.predict(val, D, False), ind.predict(val, D)
        from cc_transfer_lr import LRdeadline
        lrd = LRdeadline(cal, list(np.round(np.quantile(cal.e2e, [0.75, 0.8, 0.9, 0.95]))))
        p_lr = lrd.predict(val, D)
        blocks = np.arange(len(val)) // 50
        unseen = (val.regime == "K3").values
        h1 = block_boot_diff(p_cc[0], p_ind[0], y, blocks, auroc)
        h2 = block_boot_diff(p_ccw[1], p_ind[1], y, blocks, lambda p, yy: np.mean((p - yy) ** 2))
        e_cc, e_lr = ece(p_cc[0][unseen], y[unseen]), ece(p_lr[0][unseen], y[unseen])
        res.append(dict(file=os.path.basename(fn), D=D, n_val=len(val), miss_rate=y.mean(),
                        H1_auroc_diff=h1[0], H1_lo=h1[1], H1_pass=h1[1] > 0,
                        H2_brier_diff=h2[0], H2_hi=h2[2], H2_pass=h2[2] < 0,
                        H3_ece_cc=e_cc, H3_ece_lr=e_lr, H3_pass=e_cc < e_lr))
    R = pd.DataFrame(res); R.to_csv(f"{outdir}/unoq_confirmatory_results.csv", index=False); print(R.round(3).to_string())
    print("KILL (H3 fails on >=3 seeds):", (~R.H3_pass).sum() >= 3)


if __name__ == "__main__":
    if sys.argv[1] == "--unoq":
        os.makedirs(sys.argv[3] if len(sys.argv) > 3 else "out", exist_ok=True)
        run_unoq(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "out"); sys.exit()
    dd, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    cfg = {"lidar": [35, 50, 80], "pilot": [30, 35], "hclock": [28, 32]}
    R, P, B, I = [], [], [], []
    for ds, Ds in cfg.items():
        r, p, b, i = run(ds, dd, Ds, outdir)
        R.append(r); P.append(p); B.append(b); I.append(i)
        print(json.dumps(i), flush=True)
    pd.concat(R).to_csv(f"{outdir}/cc_metrics_by_checkpoint.csv", index=False)
    pd.concat(P).to_csv(f"{outdir}/cc_pipeline.csv", index=False)
    pd.concat(B).to_csv(f"{outdir}/cc_bootstrap_diffs.csv", index=False)
    json.dump(I, open(f"{outdir}/cc_model_info.json", "w"), indent=1)
