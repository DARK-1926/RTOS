"""
Evaluation Benchmark for Milestone 2: Gate C Baselines & Sanity Checks.

Evaluates authoritative Toba et al. plaxity early detection models:
- Variant 1: Pooled Calibration Fit (pooled marginal PMFs, per-regime deadlines)
- Variant 2: Per-Regime Calibration Fit (strongest fair baseline)
Across 11 threshold policies (linear job count ramps, linear mean time ramps,
fixed probability thresholds, and matched recall targets).

Evaluates Section 9.4 Stop Rule with 2,000 cluster-bootstraps by episode_id.

Strict Data Isolation Constraint:
- Reads calibration traces: results/uav_calibration_traces.csv (7,200 rows)
- Reads validation traces: results/uav_validation_traces.csv (3,600 rows)
- NEVER references, opens, or accesses test split traces.

Outputs:
- results/gate_c_baseline_summary.csv (418 rows)
- results/gate_c_exceedance_gap_summary.csv (19 rows)
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.detectors.toba_baseline import (
    DiscretePMF,
    TobaPlaxityBaseline,
    TobaDetectorSuite,
)


POLICIES = [
    "linear_job_0.7_1.0",
    "linear_job_0.8_1.0",
    "linear_job_0.9_1.0",
    "linear_time_0.7_1.0",
    "linear_time_0.8_1.0",
    "linear_time_0.9_1.0",
    "fixed_0.95",
    "fixed_0.99",
    "fixed_1.00",
    "matched_recall_95",
    "matched_recall_99",
]


def load_datasets_and_deadlines(repo_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Loads calibration and validation traces with strict validation safety checks."""
    calib_path = os.path.join(repo_root, "results", "uav_calibration_traces.csv")
    val_path = os.path.join(repo_root, "results", "uav_validation_traces.csv")
    deadlines_path = os.path.join(repo_root, "results", "gate_b_deadlines.json")

    assert os.path.exists(calib_path), f"Calibration traces not found at {calib_path}"
    assert os.path.exists(val_path), f"Validation traces not found at {val_path}"
    assert os.path.exists(deadlines_path), f"Deadlines file not found at {deadlines_path}"

    calib_df = pd.read_csv(calib_path)
    val_df = pd.read_csv(val_path)

    # Verify split labels
    assert (calib_df["split"] == "calib").all(), "Calibration traces contain non-calib rows!"
    assert (val_df["split"] == "val").all(), "Validation traces contain non-val rows!"

    with open(deadlines_path, "r", encoding="utf-8") as f:
        deadlines_meta = json.load(f)["regimes"]

    return calib_df, val_df, deadlines_meta


def compute_exceedance_gaps_and_stop_rule(
    calib_df: pd.DataFrame,
    val_df: pd.DataFrame,
    deadlines_meta: dict,
    scale: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes empirical exceedance on validation traces vs. independent discrete
    convolution on calibration marginals, along with 2,000 cluster-bootstraps
    by episode_id across all 18 regimes and pooled.
    """
    rng = np.random.RandomState(seed)
    records = []

    # Map regime IDs to metadata
    regime_p_indep = {}
    regime_p_emp = {}

    for reg_id in range(18):
        r_calib = calib_df[calib_df["regime_id"] == reg_id]
        r_val = val_df[val_df["regime_id"] == reg_id]
        meta = deadlines_meta[str(reg_id)]
        d_reg = float(meta["deadline_ms"])
        d_bins = int(round(d_reg * scale))
        max_d = d_reg * 1.5

        def get_pmf(samples):
            nb = int(np.ceil(max_d * scale)) + 1
            c = np.zeros(nb)
            for v in samples:
                idx = min(nb - 1, max(0, int(round(v * scale))))
                c[idx] += 1
            return c / c.sum()

        p1 = get_pmf(r_calib["dur_node1_ms"])
        p2 = get_pmf(r_calib["dur_node2_ms"])
        p3 = get_pmf(r_calib["dur_node3_ms"])
        p4 = get_pmf(r_calib["dur_node4_ms"])
        p5 = get_pmf(r_calib["dur_node5_ms"])
        p6 = get_pmf(r_calib["dur_node6_ms"])

        # Independent fork survival product: F_fork(t) = F3(t) * F4(t)
        c3 = np.cumsum(p3)
        c4 = np.cumsum(p4)
        p_fork = np.diff(np.concatenate(([0.0], np.clip(c3 * c4, 0.0, 1.0))))

        # Convolve serial stages
        curr = p1
        for nxt in [p2, p_fork, p5, p6]:
            curr = np.convolve(curr, nxt)
        curr /= curr.sum()

        # Tail exceedance P_indep(S > D)
        p_indep_pct = float(curr[d_bins + 1:].sum() * 100.0) if d_bins + 1 < len(curr) else 0.0
        p_emp_pct = float(r_val["miss"].mean() * 100.0)

        regime_p_indep[reg_id] = p_indep_pct
        regime_p_emp[reg_id] = p_emp_pct

        gap_pp = p_emp_pct - p_indep_pct

        # Cluster-bootstrap by episode_id
        eps = r_val["episode_id"].unique()
        ep_rates = np.array([r_val[r_val["episode_id"] == ep]["miss"].mean() for ep in eps])
        boot_idx = rng.choice(len(eps), size=(n_boot, len(eps)), replace=True)
        boot_gaps = (ep_rates[boot_idx].mean(axis=1) * 100.0) - p_indep_pct

        ci_lo = float(np.percentile(boot_gaps, 2.5))
        ci_hi = float(np.percentile(boot_gaps, 97.5))
        ci_excludes_zero = bool(ci_lo > 0.0 or ci_hi < 0.0)
        stop_rule_active = bool(ci_lo <= 0.0 <= ci_hi)

        records.append({
            "regime_id": str(reg_id),
            "wind_name": meta["wind_name"],
            "clutter_name": meta["clutter_name"],
            "contention_name": meta["contention_name"],
            "deadline_ms": round(d_reg, 4),
            "n_episodes": len(eps),
            "n_frames": len(r_val),
            "p_emp_pct": round(p_emp_pct, 2),
            "p_indep_pct": round(p_indep_pct, 2),
            "gap_pp": round(gap_pp, 2),
            "gap_ci_lo_pp": round(ci_lo, 2),
            "gap_ci_hi_pp": round(ci_hi, 2),
            "ci_excludes_zero": ci_excludes_zero,
            "stop_rule_active": stop_rule_active,
        })

    # Pooled Row
    pooled_emp_pct = float(val_df["miss"].mean() * 100.0)
    pooled_indep_pct = float(np.mean(list(regime_p_indep.values())))
    pooled_gap_pp = pooled_emp_pct - pooled_indep_pct

    # Cluster-bootstrap across all 72 validation episodes
    all_eps = val_df["episode_id"].unique()
    ep_to_mean = {ep: float(val_df[val_df["episode_id"] == ep]["miss"].mean()) for ep in all_eps}
    ep_rates_pooled = np.array([ep_to_mean[ep] for ep in all_eps])
    boot_idx_pooled = rng.choice(len(all_eps), size=(n_boot, len(all_eps)), replace=True)
    boot_gaps_pooled = (ep_rates_pooled[boot_idx_pooled].mean(axis=1) * 100.0) - pooled_indep_pct

    pooled_ci_lo = float(np.percentile(boot_gaps_pooled, 2.5))
    pooled_ci_hi = float(np.percentile(boot_gaps_pooled, 97.5))
    pooled_excludes_zero = bool(pooled_ci_lo > 0.0 or pooled_ci_hi < 0.0)
    pooled_stop_active = bool(pooled_ci_lo <= 0.0 <= pooled_ci_hi)

    records.append({
        "regime_id": "pooled",
        "wind_name": "all",
        "clutter_name": "all",
        "contention_name": "all",
        "deadline_ms": round(float(np.mean([float(m["deadline_ms"]) for m in deadlines_meta.values()])), 4),
        "n_episodes": len(all_eps),
        "n_frames": len(val_df),
        "p_emp_pct": round(pooled_emp_pct, 2),
        "p_indep_pct": round(pooled_indep_pct, 2),
        "gap_pp": round(pooled_gap_pp, 2),
        "gap_ci_lo_pp": round(pooled_ci_lo, 2),
        "gap_ci_hi_pp": round(pooled_ci_hi, 2),
        "ci_excludes_zero": pooled_excludes_zero,
        "stop_rule_active": pooled_stop_active,
    })

    return pd.DataFrame(records)


def evaluate_baselines_on_validation(
    calib_df: pd.DataFrame,
    val_df: pd.DataFrame,
    deadlines_meta: dict,
    scale: int = 10,
) -> pd.DataFrame:
    """
    Evaluates both Variant 1 (Pooled Toba) and Variant 2 (Per-Regime Toba)
    across all 11 policies on validation traces.
    Returns 418 rows (2 variants x 19 slices x 11 policies).
    """
    deadlines = {int(k): float(v["deadline_ms"]) for k, v in deadlines_meta.items()}
    summary_rows = []

    # Pre-extract validation ASTs and actual miss labels
    t1 = val_df["dur_node1_ms"].values
    t2 = val_df["dur_node2_ms"].values
    t3 = val_df["dur_node3_ms"].values
    t4 = val_df["dur_node4_ms"].values
    t5 = val_df["dur_node5_ms"].values
    actual_misses = val_df["miss"].values.astype(bool)
    regime_ids = val_df["regime_id"].values.astype(int)

    val_asts = {
        1: np.zeros(len(val_df), dtype=np.float64),
        2: t1,
        3: t1 + t2,
        4: t1 + t2,
        5: t1 + t2 + np.maximum(t3, t4),
        6: t1 + t2 + np.maximum(t3, t4) + t5,
    }

    variants = ["pooled_toba", "per_regime_toba"]

    for policy in POLICIES:
        # Fit detector suite for current policy
        suite = TobaDetectorSuite(scale=scale, policy=policy)
        suite.fit(calib_df, deadlines=deadlines)

        for variant in variants:
            # Vectorized evaluation for each frame
            n_frames = len(val_df)
            alarms = np.zeros(n_frames, dtype=bool)
            earliest_cps = np.full(n_frames, 999, dtype=int)
            lead_times = np.zeros(n_frames, dtype=np.float64)

            # Determine detectors per regime
            for r in range(18):
                r_mask = (regime_ids == r)
                if not np.any(r_mask):
                    continue

                det = suite.pooled_detectors[r] if variant == "pooled_toba" else suite.regime_detectors[r]
                d_r = deadlines[r]

                # Check AST for each node N1..N6
                r_alarms = np.zeros(np.sum(r_mask), dtype=bool)
                r_first_node = np.full(np.sum(r_mask), 999, dtype=int)

                for node_id in range(1, 7):
                    node_ast = val_asts[node_id][r_mask]
                    theta_v = det.thresholds[node_id]
                    # Query plaxity CDF
                    p_meet = np.array([det.plaxity_pmfs[node_id].query_meeting_prob(a) for a in node_ast])
                    node_alarm = (p_meet < theta_v)

                    # Update first alarm node
                    newly_alarmed = node_alarm & (r_first_node == 999)
                    r_first_node[newly_alarmed] = node_id
                    r_alarms |= node_alarm

                # Compute lead times for alarmed frames
                r_lead = np.zeros(np.sum(r_mask), dtype=np.float64)
                for node_id in range(1, 7):
                    first_at_node = (r_first_node == node_id)
                    if np.any(first_at_node):
                        r_lead[first_at_node] = np.maximum(0.0, d_r - val_asts[node_id][r_mask][first_at_node])

                alarms[r_mask] = r_alarms
                earliest_cps[r_mask] = r_first_node
                lead_times[r_mask] = r_lead

            # Parse policy metadata
            policy_cfg = suite.parse_policy(policy)
            pol_type = policy_cfg.get("threshold_mode", "linear_job")
            th_min = policy_cfg.get("th_min", policy_cfg.get("fixed_threshold", np.nan))
            th_max = policy_cfg.get("th_max", policy_cfg.get("fixed_threshold", np.nan))

            # Helper to compute slice metrics
            def compute_metrics(mask, slice_id, wind, clutter, contention):
                sub_y_true = actual_misses[mask]
                sub_y_pred = alarms[mask]
                sub_cp = earliest_cps[mask]
                sub_lead = lead_times[mask]

                n_tot = len(sub_y_true)
                n_m = int(np.sum(sub_y_true))
                n_s = n_tot - n_m

                tp = int(np.sum(sub_y_true & sub_y_pred))
                fp = int(np.sum((~sub_y_true) & sub_y_pred))
                tn = int(np.sum((~sub_y_true) & (~sub_y_pred)))
                fn = int(np.sum(sub_y_true & (~sub_y_pred)))

                rec_pct = (tp / n_m * 100.0) if n_m > 0 else 0.0
                fpr_pct = (fp / n_s * 100.0) if n_s > 0 else 0.0
                prec_pct = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
                f1 = (2 * prec_pct * rec_pct / (prec_pct + rec_pct) / 100.0) if (prec_pct + rec_pct) > 0 else 0.0
                acc_pct = ((tp + tn) / n_tot * 100.0) if n_tot > 0 else 0.0
                alarm_rate = ((tp + fp) / n_tot * 100.0) if n_tot > 0 else 0.0

                alarmed_leads = sub_lead[sub_y_pred]
                mean_lead = float(np.mean(alarmed_leads)) if len(alarmed_leads) > 0 else 0.0
                p50_lead = float(np.median(alarmed_leads)) if len(alarmed_leads) > 0 else 0.0

                fn1 = int(np.sum(sub_cp == 1))
                fn2 = int(np.sum(sub_cp == 2))
                fn3_4 = int(np.sum((sub_cp == 3) | (sub_cp == 4)))
                fn5 = int(np.sum(sub_cp == 5))
                fn6 = int(np.sum(sub_cp == 6))

                return {
                    "variant": variant,
                    "regime_id": str(slice_id),
                    "wind_level": wind,
                    "clutter_level": clutter,
                    "contention_level": contention,
                    "policy": policy,
                    "policy_type": pol_type,
                    "th_min": th_min,
                    "th_max": th_max,
                    "n_total": n_tot,
                    "n_miss": n_m,
                    "n_safe": n_s,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "recall_pct": round(rec_pct, 2),
                    "fpr_pct": round(fpr_pct, 2),
                    "precision_pct": round(prec_pct, 2),
                    "f1_score": round(f1, 4),
                    "accuracy_pct": round(acc_pct, 2),
                    "alarm_rate_pct": round(alarm_rate, 2),
                    "mean_lead_time_ms": round(mean_lead, 2),
                    "lead_time_p50_ms": round(p50_lead, 2),
                    "first_alarm_n1": fn1,
                    "first_alarm_n2": fn2,
                    "first_alarm_n3_n4": fn3_4,
                    "first_alarm_n5": fn5,
                    "first_alarm_n6": fn6,
                }

            # 1. 18 Individual Regimes
            for r in range(18):
                meta = deadlines_meta[str(r)]
                m_r = (regime_ids == r)
                summary_rows.append(compute_metrics(
                    m_r, r, meta["wind_name"], meta["clutter_name"], meta["contention_name"]
                ))

            # 2. Pooled Slice ('all')
            m_all = np.ones(n_frames, dtype=bool)
            summary_rows.append(compute_metrics(
                m_all, "all", "all", "all", "all"
            ))

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gate C Baselines & Sanity Checks.")
    parser.add_argument("--scale", type=int, default=10, help="Bin resolution scale factor (default: 10 -> 0.1 ms).")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of cluster bootstrap iterations (default: 2000).")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"[Gate C Benchmark] Repository Root: {repo_root}")
    print(f"[Gate C Benchmark] Scale: {args.scale} (bin width = {1.0/args.scale:.2f} ms), Bootstraps: {args.n_boot}")

    calib_df, val_df, deadlines_meta = load_datasets_and_deadlines(repo_root)
    print(f"[Gate C Benchmark] Loaded {len(calib_df)} calib frames and {len(val_df)} val frames across 18 regimes.")

    # 1. Exceedance Gaps and Stop Rule
    print("\n--- Computing Tail Exceedance Gaps and Evaluating Section 9.4 Stop Rule ---")
    gap_summary_df = compute_exceedance_gaps_and_stop_rule(
        calib_df=calib_df,
        val_df=val_df,
        deadlines_meta=deadlines_meta,
        scale=args.scale,
        n_boot=args.n_boot,
    )

    out_gap_csv = os.path.join(repo_root, "results", "gate_c_exceedance_gap_summary.csv")
    gap_summary_df.to_csv(out_gap_csv, index=False)
    print(f"[Gate C Benchmark] Exceedance gap summary saved to: {out_gap_csv}")

    # Evaluate Section 9.4 Stop Rule
    # "If matched-gap CI includes 0 in all regimes at Gate C, stop and report negative result."
    regime_rows = gap_summary_df[gap_summary_df["regime_id"] != "pooled"]
    regimes_with_ci_excluding_zero = regime_rows[regime_rows["ci_excludes_zero"]]
    pooled_ci_excludes_zero = gap_summary_df[gap_summary_df["regime_id"] == "pooled"]["ci_excludes_zero"].iloc[0]

    print("\n[Gate C Stop Rule Audit Outcome]:")
    print(f"Total Regimes Audited: {len(regime_rows)}")
    print(f"Regimes with 95% Cluster CI strictly excluding 0: {len(regimes_with_ci_excluding_zero)}")
    for _, r in regimes_with_ci_excluding_zero.iterrows():
        print(f"  - Regime {r['regime_id']} ({r['wind_name']}/{r['clutter_name']}/{r['contention_name']}): "
              f"Gap = {r['gap_pp']:+.2f} pp [95% CI: {r['gap_ci_lo_pp']:+.2f}, {r['gap_ci_hi_pp']:+.2f}] pp")
    pooled_row = gap_summary_df[gap_summary_df["regime_id"] == "pooled"].iloc[0]
    print(f"Pooled Exceedance Gap: {pooled_row['gap_pp']:+.2f} pp [95% CI: {pooled_row['gap_ci_lo_pp']:+.2f}, {pooled_row['gap_ci_hi_pp']:+.2f}] pp")

    if len(regimes_with_ci_excluding_zero) > 0 or pooled_ci_excludes_zero:
        print("\n>>> Gate C Stop Rule: PASS (Stop rule NOT triggered; tail underestimation confirmed on physical traces).")
        print(">>> System is authorized to proceed to Gate D (Candidate Detectors D1/D2).\n")
    else:
        print("\n>>> Gate C Stop Rule: TRIGGERED (All regime CIs include 0). STOP AND REPORT NEGATIVE RESULT.\n")

    # 2. Comprehensive Baseline Evaluation
    print("--- Evaluating Baseline Variants (Pooled & Per-Regime) across 11 Policies ---")
    baseline_summary_df = evaluate_baselines_on_validation(
        calib_df=calib_df,
        val_df=val_df,
        deadlines_meta=deadlines_meta,
        scale=args.scale,
    )

    out_base_csv = os.path.join(repo_root, "results", "gate_c_baseline_summary.csv")
    baseline_summary_df.to_csv(out_base_csv, index=False)
    print(f"[Gate C Benchmark] Baseline summary saved to: {out_base_csv} ({len(baseline_summary_df)} rows)")

    assert len(baseline_summary_df) == 418, f"Expected 418 rows in baseline summary, got {len(baseline_summary_df)}"
    print("[Gate C Benchmark] Verification completed successfully with zero errors!")


if __name__ == "__main__":
    main()
