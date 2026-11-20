"""Addendum H confirmatory analysis (FROZEN before any H data).
    python h_confirm.py part1   -> H13 (Autoware, 161-163) + H14 (real apps, 171-173)
    python h_confirm.py part2   -> EuRoC drone camera VO pipeline (181-183)
Outputs results/h_confirm_<part>_runs.csv and results/h_confirm_<part>_summary.txt.
H1-H5: burst-guarded ACI (BG-ACI, W=50, kappa=0.05) vs plain ACI on KPT scores at checkpoints 0-1.
H6: the Addendum G checks G1-G9 re-applied unchanged (rules scaled to n).
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import numpy as np, pandas as pd
from cc_plaxity import auroc, ece, block_boot_diff
from g13_analysis import build, mat
from g_confirm import stack, since_switch
from kp_methods import KalmanPlaxity, aci_alarms, aci_bound_upper, static_theta
from kpt_methods import KalmanPlaxityT
from sweep_analysis import toba_scores, EWMACP, sweep_metrics, GAMMA, LAG

W_BG, KAPPA = 50, 0.05
PARTS = {"part1": [("H13", "data/H13", "g13", s) for s in (161, 162, 163)] + [("H14", "data/H14", "g14", s) for s in (171, 172, 173)],
         "part2": [("EUROC", "data/H15", "vo", s) for s in (181, 182, 183)]}
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def bg_aci(Pk, y, alpha, gamma, theta0, lag=1, W=W_BG, kappa=KAPPA):
    """Burst-guarded ACI. W=0 or kappa=0 gives plain ACI (identical to kp_methods.aci_alarms)."""
    n = len(y); th = theta0; first = np.full(n, -1); err = np.zeros(n)
    for t in range(n):
        lo = max(0, t - lag - W + 1); hi = t - lag + 1
        guard = kappa * max(0.0, err[lo:hi].sum() - alpha * W) if (W and hi > 0) else 0.0
        hit = np.nonzero(Pk[t] > th - guard)[0]
        first[t] = hit[0] if len(hit) else -1
        err[t] = float(y[t] == 1 and first[t] < 0)
        if t - lag >= 0:
            th = th - gamma * (err[t - lag] - alpha)
    return first, err


def w100(e):
    return pd.Series(e).rolling(100).mean().max()


def run_one(cond, d, pre, s):
    cal = build(f"{d}/{pre}_s{s}_calib.csv", "calib", 0); val = build(f"{d}/{pre}_s{s}_val.csv", "val", 100000)
    raw = pd.read_csv(f"{d}/{pre}_s{s}_val.csv")
    D = float(np.round(np.quantile(cal.e2e, 0.85))); y = (val.e2e > D).values.astype(int); yc = (cal.e2e > D).values.astype(int)
    T, Tc = toba_scores(cal, val, D), toba_scores(cal, cal, D)
    kpt = KalmanPlaxityT(cal, nq=64, features=("u", "r", "w")); kp = KalmanPlaxity(cal, nq=64); ew = EWMACP(cal)
    raw_s = {"KPT": (mat(kpt.predict(val, D)), mat(kpt.predict(cal, D))), "KP": (mat(kp.predict(val, D)), mat(kp.predict(cal, D))),
             "EWMA-CP": (mat(ew.predict(val, D)), mat(ew.predict(cal, D)))}
    S = {m + "+stack": (stack(V, C, T, Tc, yc), stack(C, C, Tc, Tc, yc)) for m, (V, C) in raw_s.items()}
    S["Toba-eq"] = (T, Tc)
    reg = val.regime.values; k3 = reg == "K3"; sw = since_switch(reg) < 15; b = np.arange(len(val)) // 50
    say(f"--- {cond} seed {s}: n_val={len(val)} D={D} miss={y.mean():.3f} K3 miss={y[k3].mean():.3f} KPT prior R2={kpt.r2_prior:.3f}; "
        f"telemetry us p99 {raw.us_tm.quantile(.99):.2f}")
    rows = []
    for m, (V, C) in S.items():
        r = dict(cond=cond, seed=s, method=m, D=D, miss=y.mean(), **sweep_metrics(V, y, val, D),
                 ece_rel_K3=ece(V[k3, 0], y[k3]), auc_rel_switch=auroc(V[sw, 0], y[sw]), us_tm_p99=raw.us_tm.quantile(0.99))
        for al, tag in [(0.05, ""), (0.02, "_02")]:
            th0 = static_theta(C[:, :2], yc, al); ub = al + aci_bound_upper(len(y), GAMMA, LAG, max(th0, 0))
            f, e, _ = aci_alarms(V[:, :2], y, al, GAMMA, th0, LAG)
            fb, eb = bg_aci(V[:, :2], y, al, GAMMA, th0, LAG)
            r.update({f"aci_unw{tag}": e.mean(), f"aci_fpr{tag}": (f >= 0)[y == 0].mean(), f"aci_ub{tag}": ub,
                      f"bg_unw{tag}": eb.mean(), f"bg_fpr{tag}": (fb >= 0)[y == 0].mean()})
            if al == 0.05:
                r.update(worst100=w100(e), bg_worst100=w100(eb), aci_unw_K2=e[reg == "K2"].mean(), bg_unw_K2=eb[reg == "K2"].mean())
                if m == "KPT+stack":   # matched-FPR plain-ACI reference
                    tgt = r["bg_fpr"]; best = None
                    for a in np.arange(0.005, 0.0801, 0.0025):
                        f2, e2, _ = aci_alarms(V[:, :2], y, a, GAMMA, static_theta(C[:, :2], yc, a), LAG)
                        dd = abs((f2 >= 0)[y == 0].mean() - tgt)
                        if best is None or dd < best[0]:
                            best = (dd, w100(e2), a, (f2 >= 0)[y == 0].mean())
                    r.update(ref_worst100=best[1], ref_alpha=best[2], ref_fpr=best[3])
        if m != "Toba-eq":
            dd = block_boot_diff(V[:, :2].max(1), T[:, :2].max(1), y, b, auroc); r.update(dact_vs_toba=dd[0], lo_vs_toba=dd[1])
        rows.append(r)
    return rows


def main(part):
    rows = []
    for run in PARTS[part]:
        rows += run_one(*run)
    R = pd.DataFrame(rows); R.to_csv(f"results/h_confirm_{part}_runs.csv", index=False)
    n = R.seed.nunique(); full = n; most = 5 if n == 6 else 2; four = 4 if n == 6 else 2
    k = R[R.method == "KPT+stack"].set_index("seed"); t = R[R.method == "Toba-eq"].set_index("seed")
    P = lambda c: R.pivot_table(index="seed", columns="method", values=c)
    rule = lambda x, need: f"{x}/{n} (rule {'>=' if need < n else ''}{need}/{n}) -> {'PASS' if x >= need else 'FAIL'}"
    say(f"\n===== Addendum H {part} =====")
    say("H1 worst100 BG-ACI < ACI:                 " + rule(int((k.bg_worst100 < k.worst100).sum()), most))
    say("H2 worst100 BG-ACI < ACI at matched FPR:  " + rule(int((k.bg_worst100 < k.ref_worst100).sum()), four))
    h3 = int(((k.bg_unw <= k.aci_ub) & (k.bg_unw_02 <= k.aci_ub_02)).sum())
    say("H3 BG-ACI one-sided guarantee, both alpha: " + rule(h3, full))
    say("H4 BG-ACI FPR < Toba-eq ACI FPR:          " + rule(int((k.bg_fpr < t.aci_fpr).sum()), full))
    say("H5 K2 unwarned BG-ACI < ACI:              " + rule(int((k.bg_unw_K2 < k.aci_unw_K2).sum()), four))
    a, sw_, ek, ay, wo, fp, tm = P("auc_act"), P("auc_rel_switch"), P("ece_rel_K3"), P("auc_any"), P("worst100"), P("aci_fpr"), P("us_tm_p99")
    say("H6 replication of Addendum G:")
    say("  G1 ECE unseen K3 KPT < KP:           " + rule(int((ek["KPT+stack"] < ek["KP+stack"]).sum()), most))
    say("  G2 post-switch AUROC KPT > KP:       " + rule(int((sw_["KPT+stack"] > sw_["KP+stack"]).sum()), most))
    say("  G3 actionable AUROC KPT >= KP-0.005: " + rule(int((a["KPT+stack"] >= a["KP+stack"] - 0.005).sum()), most))
    say("  G4 actionable KPT > Toba, CI > 0:    " + rule(int((k.lo_vs_toba > 0).sum()), full))
    say("  G5 any-time KPT >= Toba - 0.005:     " + rule(int((ay["KPT+stack"] >= ay["Toba-eq"] - 0.005).sum()), most))
    say("  G6 ACI guarantee both alpha:         " + rule(int(((k.aci_unw <= k.aci_ub) & (k.aci_unw_02 <= k.aci_ub_02)).sum()), full))
    say("  G7 ACI FPR KPT < Toba:               " + rule(int((fp["KPT+stack"] < fp["Toba-eq"]).sum()), full))
    say("  G8 worst100 KPT < KP:                " + rule(int((wo["KPT+stack"] < wo["KP+stack"]).sum()), four))
    say("  G9 telemetry p99 < 5 us:             " + rule(int((tm["KPT+stack"] < 5).sum()), full))
    pd.set_option("display.width", 250)
    say("\n" + k[["cond", "miss", "auc_act", "aci_unw", "bg_unw", "aci_ub", "aci_fpr", "bg_fpr", "ref_fpr", "worst100", "bg_worst100",
                  "ref_worst100", "aci_unw_K2", "bg_unw_K2"]].round(3).to_string())
    say("Toba-eq ACI FPR: " + t.aci_fpr.round(3).to_string().replace("\n", " | "))
    say("\nmeans by method:\n" + R.groupby(["cond", "method"])[["auc_rel", "auc_act", "auc_any", "recall_fpr10", "lead_ms_fpr10", "aci_fpr",
                                                                  "bg_fpr", "worst100", "bg_worst100"]].mean().round(3).to_string())
    open(f"results/h_confirm_{part}_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main(sys.argv[1])
