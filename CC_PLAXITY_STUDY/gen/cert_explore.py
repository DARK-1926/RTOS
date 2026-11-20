"""EXPLORATORY (not pre-registered), 16 Nov 2026. Two questions behind a possible 'stability certificate' contribution.
  A. Assumption check on the Addendum O closed-loop data: do WARNED frames meet the deadline? (A certificate needs a timely fallback.)
  B. Certificate for the Addendum N plant: with at most m 'hold' frames in any K, is the loop provably stable for ANY miss pattern?
     Deterministic part of control_study.py: state z = [x, u_prev].  Update frame: x+ = (Ad - Bd K) x, u = -K x.  Hold frame: x+ = Ad x + Bd u_prev.
     Every admissible infinite sequence splits into blocks of length L; each block has <= m holds per K-window. If the largest norm of a
     block product is < 1 (in any fixed norm), the constrained switched system is exponentially stable (sub-multiplicativity).
     We use L = K and the norm induced by the Lyapunov matrix of the update mode.  Exhaustive over all words with <= m holds in K.
    python cert_explore.py
"""
import itertools, sys, os
import numpy as np, pandas as pd
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from control_study import plant
from o_confirm import params


def part_a():
    print("A. warned frames vs deadline (Addendum O, Pi 5)")
    rows = []
    for s in (221, 222, 223):
        D = params(f"data/O/params_s{s}.txt")["D"]
        for p in ("kpm", "kpmwh", "random"):
            a = pd.read_csv(f"data/O/cl_s{s}_val_{p}.csv"); r = a[(a.done == 1) & (a.regime != "K-1")].iloc[20:]
            miss = (r.e5 > D).values; d1 = (r.degraded == 1).values; d2 = (r.degraded == 2).values; d0 = (r.degraded == 0).values
            rows.append(dict(seed=s, policy=p, miss_normal=miss[d0].mean(), miss_degr_release=miss[d1].mean(), miss_degr_late=miss[d2].mean() if d2.any() else np.nan,
                             frac_release=d1.mean(), frac_late=d2.mean(), p99_over_D_release=np.quantile(r.e5.values[d1], 0.99) / D,
                             max_over_D_release=r.e5.values[d1].max() / D))
    R = pd.DataFrame(rows); pd.set_option("display.width", 220); print(R.round(3).to_string(index=False)); return R


def mats(h, w0):
    Ad, Bd, K = plant(h, w0)
    U = np.zeros((3, 3)); U[:2, :2] = Ad - Bd @ K; U[2, :2] = -K           # update: new command computed
    H = np.zeros((3, 3)); H[:2, :2] = Ad; H[:2, 2:] = Bd; H[2, 2] = 1.0   # hold: previous command kept
    return U, H


def worst_block(U, H, m, K):
    """max over words of length K with <= m holds of ||product|| in a Lyapunov-shaped norm; also unconstrained-hold growth."""
    # norm: z' P z with P from the update mode on x, plus a weight on u_prev (U is nilpotent in u, so use a few-step lifted Lyapunov)
    n = 3; best = np.inf
    for q in (1e-3, 1e-2, 1e-1, 1.0):
        try:
            P = solve_discrete_lyapunov((U * 1.0).T, np.diag([1.0, 1.0, q]))
        except Exception:
            continue
        S = np.real(sqrtm(P)); Si = np.linalg.inv(S); worst = 0.0
        # dynamic programming over number of holds is not norm-exact, so enumerate (K <= 20, m <= 5 -> <= 21,700 words)
        for k in range(m + 1):
            for pos in itertools.combinations(range(K), k):
                M = np.eye(n); ps = set(pos)
                for t in range(K):
                    M = (H if t in ps else U) @ M
                worst = max(worst, np.linalg.norm(S @ M @ Si, 2))
        best = min(best, worst)
    return best


def part_b():
    print("\nB. certificate: worst K-block gain under <= m holds per K (gain < 1 => stable for every admissible miss pattern)")
    rows = []
    for h in (0.1, 0.2):
        for w0 in (2.0, 3.0, 4.0, 5.0):
            U, H = mats(h, w0); rho_h = max(abs(np.linalg.eigvals(H)))
            for (m, K) in ((1, 10), (2, 10), (2, 20), (3, 20), (4, 20), (5, 20)):
                g = worst_block(U, H, m, K)
                rows.append(dict(h=h, w0=w0, m=m, K=K, hold_growth_per_frame=rho_h, block_gain=g, certified=bool(g < 1)))
    R = pd.DataFrame(rows); R.to_csv("results/cert_explore.csv", index=False)
    print(R.pivot_table(index=["h", "w0"], columns=["m", "K"], values="block_gain").round(3).to_string())
    print("\nper-frame growth of the hold mode (no monitor, sustained misses):")
    print(R.groupby(["h", "w0"]).hold_growth_per_frame.first().round(3).to_string())
    return R


if __name__ == "__main__":
    part_a(); part_b()
