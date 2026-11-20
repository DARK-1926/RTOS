"""Addendum D confirmatory analysis (G10a board, G10b laptop, G10s board synthetic). Run from gen/:
    python g10_analysis.py
Writes results/g10_openloop.csv, g10_closedloop.csv, g10_overhead.csv, g10_alarms.csv, g10_summary.txt
"""
import glob, os, sys
import numpy as np, pandas as pd
from ext_analysis import score
from toba_eq_alarms import evaluate, attach
from g_analysis import load
from export_tables import check

OUT = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.append(s)


def combined(cond, s):
    """calib run + 'none' val run as one trace (val ids offset so the HMM never links across the gap)."""
    d = f"data/{cond}"
    c = pd.read_csv(f"{d}/cal_s{s}.csv"); v = pd.read_csv(f"{d}/val_s{s}_none.csv")
    v = v.assign(global_frame_id=v.global_frame_id + 100000)
    fn = f"{d}/combined/open_s{s}.csv"; os.makedirs(os.path.dirname(fn), exist_ok=True)
    pd.concat([c, v])[c.columns[:12]].to_csv(fn, index=False)
    return fn


def seeds(cond):
    return sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(f"data/{cond}/cal_s*.csv"))


def main():
    rows, cl, ov, al = [], [], [], []
    q = pd.read_csv("data/G10a/quality.csv").set_index("scan_idx")
    for cond in ["G10a", "G10b"]:
        for s in seeds(cond):
            if not os.path.exists(f"data/{cond}/val_s{s}_naive.csv"):
                continue
            fn = combined(cond, s)
            say(f"--- {cond} seed {s}: C vs Python check (none run)")
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                check(f"data/{cond}/cal_s{s}.csv", f"data/{cond}/val_s{s}_none.csv")
            say(buf.getvalue().strip())
            r = score(fn, cond, "standard"); r["seed"] = s; rows.append(r)
            cal, val = load(fn); raw = pd.read_csv(fn)
            cal = attach(cal, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
            val = attach(val, raw, "global_frame_id", "dur_node3_ms", "dur_node4_ms")
            for a in evaluate(cal, val, r["D"], "K0", f"{cond}_s{s}"):
                a.update(cond=cond, seed=s); al.append(a)
            for p in ["none", "toba", "cc", "ccsc", "naive"]:
                v = pd.read_csv(f"data/{cond}/val_s{s}_{p}.csv").iloc[20:]
                sd = np.where(v.degraded == 1, q.loc[v.bag_frame_idx, "sector_diff_m1"].values,
                              np.where(v.degraded == 2, q.loc[v.bag_frame_idx, "sector_diff_m2"].values, 0))
                k3 = v.regime == "K3"
                cl.append(dict(cond=cond, seed=s, policy=p, D=r["D"], n=len(v), miss=(v.end_to_end_ms > r["D"]).mean(),
                               miss_K3=(v.end_to_end_ms[k3] > r["D"]).mean(), miss_K0_2=(v.end_to_end_ms[~k3] > r["D"]).mean(),
                               degraded=(v.degraded > 0).mean(), degraded_pre_N2=(v.degraded == 1).mean(),
                               sector_change=sd.mean() / 72, alarms=(v.alarm_cp >= 0).mean(),
                               mean_e2e=v.end_to_end_ms.mean(), p99_e2e=v.end_to_end_ms.quantile(.99)))
                for m in ["us_cc", "us_ccsc_extra", "us_naive", "us_toba", "us_total"]:
                    ov.append(dict(cond=cond, seed=s, policy=p, monitor=m, median=v[m].median(), p99=v[m].quantile(.99),
                                   max=v[m].max(), mean=v[m].mean()))
    R = pd.DataFrame(rows); C = pd.DataFrame(cl); O = pd.DataFrame(ov); A = pd.DataFrame(al)
    # G10s: frozen synthetic workload
    for fn in sorted(glob.glob("data/G10s/*.csv")):
        r = score(fn, "G10s", "standard"); r["seed"] = int(fn[-7:-4]); rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv("results/g10_openloop.csv", index=False); C.to_csv("results/g10_closedloop.csv", index=False)
    O.to_csv("results/g10_overhead.csv", index=False); A.to_csv("results/g10_alarms.csv", index=False)
    pd.set_option("display.width", 250)
    for cond, need in [("G10a", 4), ("G10b", 2)]:
        g = R[R.cond == cond]
        if not len(g):
            continue
        say(f"\n===== {cond} ({len(g)} seeds)")
        say(g[["seed", "D", "K3_miss", "auroc_cc", "auroc_ccsc", "auroc_ar1", "auroc_ewma", "auroc_lriso", "eceK3_cc", "eceK3_ccsc",
               "eceK3_lr", "eceK3_lriso", "eceK3_naive", "eceK3_ar1", "eceK3_ewma", "sc_switch_in", "sc_switch_K3"]].round(3).to_string(index=False))
        h1 = (g.dAUC_cc_indep_lo > 0).sum()
        d1a = (g.eceK3_ccsc < g.eceK3_cc).sum(); d1b = (g.auroc_ccsc >= g.auroc_cc - 0.01).sum()
        say(f"D-H1 CC > Toba-indep (CI>0): {h1}/{len(g)}  -> {'PASS' if h1 >= need else 'FAIL'}")
        if cond == "G10a":
            say(f"D1a ECE_K3(CC-SC) < ECE_K3(CC): {d1a}/5 ; D1b AUROC(CC-SC) >= AUROC(CC)-0.01: {d1b}/5 -> "
                f"{'PASS' if d1a >= 4 and d1b >= 4 else 'FAIL'}")
        say(f"H3-style CC < LR on K3 ECE: {(g.eceK3_cc < g.eceK3_lr).sum()}/{len(g)}; CC-SC < LR: {(g.eceK3_ccsc < g.eceK3_lr).sum()}/{len(g)}")
    g = R[R.cond == "G10s"]
    if len(g):
        say(f"\n===== G10s ({len(g)} seeds)")
        say(g[["seed", "D", "K3_miss", "auroc_cc", "auroc_ccsc", "auroc_ewma", "eceK3_cc", "eceK3_ccsc", "eceK3_lr", "eceK3_ewma",
               "eceK3_ar1", "sc_switch_in", "sc_switch_K3"]].round(3).to_string(index=False))
        d2a = (g.eceK3_ccsc <= g.eceK3_cc + 0.02).sum(); d2b = (g.eceK3_ccsc < g.eceK3_lr).sum()
        say(f"D2a ECE_K3(CC-SC) <= CC+0.02: {d2a}/3 ; D2b CC-SC < LR: {d2b}/3 -> {'PASS' if d2a >= 2 and d2b >= 2 else 'FAIL'}")
        say(f"H1 (CC > indep, CI>0): {(g.dAUC_cc_indep_lo > 0).sum()}/3")
    F = R[R.cond.isin(["G10a", "G10b", "G10s"])]
    say(f"\n===== D3 (all {len(F)} fresh runs, two-sided)")
    for m in ["ar1", "ewma", "lriso"]:
        say(f"  release AUROC CC vs {m}: CC sig. better {(F[f'dAUC_cc_{m}_lo'] > 0).sum()}, sig. worse {(F[f'dAUC_cc_{m}_hi'] < 0).sum()}")
    for m in ["ar1", "ewma"]:
        say(f"  release AUROC CC-SC vs {m}: sig. better {(F[f'dAUC_ccsc_{m}_lo'] > 0).sum()}, sig. worse {(F[f'dAUC_ccsc_{m}_hi'] < 0).sum()}")
    k = (F.eceK3_ccsc < np.minimum(F.eceK3_ar1, F.eceK3_ewma)).sum()
    say(f"  K3 ECE: CC-SC < min(AR1, EWMA) in {k}/{len(F)}; CC < min: {(F.eceK3_cc < np.minimum(F.eceK3_ar1, F.eceK3_ewma)).sum()}/{len(F)}"
        f"; CC-SC < LR-iso: {(F.eceK3_ccsc < F.eceK3_lriso).sum()}/{len(F)}")
    say("  WORDING RULE TRIGGERED (simple history models capture the advantage)" if k < len(F) / 2 else "  wording rule not triggered")
    if len(C):
        say("\n===== D4 closed loop (val miss rate / degraded fraction / output sector change)")
        P = C.pivot_table(index=["cond", "seed"], columns="policy", values=["miss", "degraded", "sector_change"])
        say(P.round(3).to_string())
        for cond, need in [("G10a", 4), ("G10b", 2)]:
            x = C[C.cond == cond].pivot(index="seed", columns="policy", values="miss")
            if len(x):
                w = (x.ccsc < x.toba).sum()
                say(f"D4 {cond}: miss(CC-SC) < miss(Toba-eq) in {w}/{len(x)}" + (f" -> {'PASS' if w >= need else 'FAIL'}" if cond == "G10a" else " (descriptive)"))
        say("\n===== EXPLORATORY (not pre-registered): misses vs a random-degradation policy with the SAME budget")
        say("  random(f) = f * miss(all frames degraded, from the toba or naive run that degraded >= 98%) + (1 - f) * miss(none)")
        ex = []
        for (cond, s), x in C.groupby(["cond", "seed"]):
            x = x.set_index("policy")
            full = [p for p in ["toba", "naive"] if x.loc[p, "degraded"] >= 0.98]
            if not full:
                continue
            md = x.loc[full, "miss"].mean(); m0 = x.loc["none", "miss"]
            for p in ["cc", "ccsc"]:
                f = x.loc[p, "degraded"]; rnd = f * md + (1 - f) * m0
                ex.append(dict(cond=cond, seed=s, policy=p, degraded=f, miss=x.loc[p, "miss"], miss_random_same_budget=rnd,
                               misses_avoided_vs_none=m0 - x.loc[p, "miss"], misses_avoided_random=m0 - rnd))
        E = pd.DataFrame(ex); E.to_csv("results/g10_budget_exploratory.csv", index=False)
        if len(E):
            say(E.round(3).to_string(index=False))
            say(f"  CC-SC beats random at same budget in {(E[E.policy == 'ccsc'].miss < E[E.policy == 'ccsc'].miss_random_same_budget).sum()}"
                f"/{(E.policy == 'ccsc').sum()} runs; CC in {(E[E.policy == 'cc'].miss < E[E.policy == 'cc'].miss_random_same_budget).sum()}"
                f"/{(E.policy == 'cc').sum()}")
        say("\n===== D5 overhead (us per frame; all policies pooled)")
        say(O.groupby(["cond", "monitor"])[["median", "p99", "max"]].agg({"median": "median", "p99": "max", "max": "max"}).round(1).to_string())
        t = O[(O.cond == "G10a") & (O.monitor == "us_total")]
        if len(t):
            say(f"D5 G10a: worst per-run p99 total monitor cost = {t.p99.max():.1f} us -> {'PASS' if t.p99.max() < 500 else 'FAIL'} (< 500 us)")
    if len(A):
        say("\n===== D6 alarm precision at Toba job-count thresholds (none runs)")
        a = A[A.policy == "var_job_0.8_1.0"].pivot_table(index=["cond", "seed"], columns="method", values=["precision", "recall", "fpr", "lead_ms_tp"])
        say(a.round(3).to_string())
        x = A[(A.policy == "var_job_0.8_1.0") & (A.cond == "G10a")].pivot(index="seed", columns="method", values="precision")
        if len(x):
            w = (x["CC-SC"] > x["Toba-eq"]).sum(); say(f"D6 G10a: precision CC-SC > Toba-eq in {w}/{len(x)} -> {'PASS' if w >= 4 else 'FAIL'}")
    open("results/g10_summary.txt", "w").write("\n".join(OUT))


if __name__ == "__main__":
    main()
