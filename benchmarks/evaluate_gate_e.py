r"""
Evaluation Benchmark for Milestone 4: Gate E Confirmatory Test Split Evaluation.

Performs the single-pass confirmatory evaluation on the fresh, untouched test split:
1. Baseline 1 (Pooled Toba Plaxity): Static marginal PMFs, per-regime deadlines (Eq. 9-12).
2. Baseline 2 (Oracle Per-Regime Toba Plaxity): Per-regime marginal PMFs with oracle ground-truth regime ID.
3. Candidate D1 (Observable-State Regime-Conditioned): Rolling median observable state (W=5)
   mapping to nearest calibration centroid; selects per-regime plaxity model.
4. Candidate D2 (Super-Node + Observable-State): D1 observable-state conditioning combined
   with empirical fork super-node N_fork = max(X3, X4), bypassing Eq. 11.

Generates:
- results/gate_e_test_summary.csv (76 rows: 19 slices x 4 detectors)
- results/gate_e_bootstrap_ci.csv (19 slices x paired cluster-bootstrap CIs)
- results/gate_e_control_regimes.csv (18 regimes control audit)

Protocol Rules (PREREG.md §9.5, §9.6):
- Fits detectors strictly on results/uav_calibration_traces.csv (7,200 rows) with
  pre-registered deadlines from results/gate_b_deadlines.json.
- Evaluates strictly on the fresh test split (7,200 rows across 144 episodes).
- Maintains state reset per episode for online observable state estimators.
- 2,000 paired cluster-bootstraps resampled strictly at the episode_id cluster level.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detectors.toba_baseline import (
    DiscretePMF,
    TobaPlaxityBaseline,
    TobaDetectorSuite,
)
from src.detectors.regime_detector import (
    ObservableStateEstimator,
    SuperNodePlaxityBaseline,
    RegimeConditionedDetectorD1,
    SuperNodePlaxityDetectorD2,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_TRACES_PATH = os.path.join(REPO_ROOT, "results", "uav_test_traces.csv")


def load_datasets_and_deadlines(
    repo_root: str,
    test_traces_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float], Dict[str, Any]]:
    """
    Loads calibration and fresh test traces with strict safety and split assertions.
    """
    calib_path = os.path.join(repo_root, "results", "uav_calibration_traces.csv")
    if test_traces_path is None:
        test_traces_path = os.path.join(repo_root, "results", "uav_test_traces.csv")
    deadlines_path = os.path.join(repo_root, "results", "gate_b_deadlines.json")

    assert os.path.exists(calib_path), f"Calibration traces not found at {calib_path}"
    assert os.path.exists(test_traces_path), f"Evaluation traces not found at {test_traces_path}"
    assert os.path.exists(deadlines_path), f"Deadlines file not found at {deadlines_path}"

    calib_df = pd.read_csv(calib_path)
    test_df = pd.read_csv(test_traces_path)

    assert (calib_df["split"] == "calib").all(), "Calibration traces contain non-calib rows!"
    assert (test_df["split"] == "test").all(), "Evaluation traces contain non-test rows!"

    assert len(calib_df) == 7200, f"Expected 7200 calib rows, got {len(calib_df)}"
    assert len(test_df) == 7200, f"Expected 7200 test rows, got {len(test_df)}"
    assert test_df["episode_id"].nunique() == 144, f"Expected 144 episodes, got {test_df['episode_id'].nunique()}"

    # Verify disjoint episodes
    calib_eps = set(calib_df["episode_id"].unique())
    test_eps = set(test_df["episode_id"].unique())
    overlap = calib_eps.intersection(test_eps)
    assert len(overlap) == 0, f"Critical split contamination: {len(overlap)} episodes overlap between calib and test!"

    with open(deadlines_path, "r", encoding="utf-8") as f:
        meta = json.load(f)["regimes"]

    deadlines = {int(k): float(v["deadline_ms"]) for k, v in meta.items()}
    return calib_df, test_df, deadlines, meta


def fit_all_detectors(
    calib_df: pd.DataFrame,
    deadlines: Dict[int, float],
    scale: int = 10,
    policy: str = "matched_recall_95",
    window: int = 5,
    fork_mode: str = "wall",
) -> Dict[str, Any]:
    """
    Fits all 4 detector configurations strictly on calibration traces.
    """
    # 1. Baseline 1 & Baseline 2: TobaDetectorSuite
    suite = TobaDetectorSuite(scale=scale, policy=policy)
    suite.fit(calib_df, deadlines=deadlines)

    # 2. Candidate D1: Observable-State Regime Conditioned
    detector_d1 = RegimeConditionedDetectorD1(
        scale=scale, policy=policy, window=window
    )
    detector_d1.fit(calib_df, deadlines=deadlines)

    # 3. Candidate D2: Super-Node + Observable State
    detector_d2 = SuperNodePlaxityDetectorD2(
        scale=scale, policy=policy, window=window, fork_mode=fork_mode
    )
    detector_d2.fit(calib_df, deadlines=deadlines)

    return {
        "suite": suite,
        "detector_d1": detector_d1,
        "detector_d2": detector_d2,
    }


def evaluate_stream(
    eval_df: pd.DataFrame,
    models: Dict[str, Any],
) -> pd.DataFrame:
    """
    Evaluates 4 detector configurations sequentially across test episodes,
    maintaining strict episode state reset.
    """
    suite = models["suite"]
    det_d1 = models["detector_d1"]
    det_d2 = models["detector_d2"]

    records = []
    episodes = eval_df["episode_id"].unique()

    for ep in episodes:
        ep_df = eval_df[eval_df["episode_id"] == ep].sort_values("frame_id")
        det_d1.reset_episode()
        det_d2.reset_episode()

        for _, row in ep_df.iterrows():
            f_id = int(row["frame_id"])
            true_r = int(row["regime_id"])
            actual_miss = bool(row["miss"]) if "miss" in row else (
                float(row["s_dag_ms"]) > float(row["deadline_ms"])
            )

            # 1. Baseline 1 (Pooled Toba)
            b1_res = suite.pooled_detectors[true_r].evaluate_frame(row)

            # 2. Baseline 2 (Oracle Per-Regime Toba)
            b2_res = suite.regime_detectors[true_r].evaluate_frame(row)

            # 3. Candidate D1 (Observable State W=5)
            d1_res = det_d1.evaluate_frame(row, update_state=True)

            # 4. Candidate D2 (Super-Node + Observable State)
            d2_res = det_d2.evaluate_frame(row, update_state=True)

            records.append({
                "episode_id": ep,
                "frame_id": f_id,
                "regime_id": true_r,
                "wind_name": str(row["wind_name"]),
                "clutter_name": str(row["clutter_name"]),
                "contention_name": str(row["contention_name"]),
                "actual_miss": actual_miss,
                "d1_pred_regime": d1_res["pred_regime"],
                "d2_pred_regime": d2_res["pred_regime"],
                # Alarms
                "b1_alarm": bool(b1_res["alarm"]),
                "b1_lead": float(b1_res["lead_time_ms"]),
                "b1_cp": b1_res["earliest_checkpoint"],
                "b2_alarm": bool(b2_res["alarm"]),
                "b2_lead": float(b2_res["lead_time_ms"]),
                "b2_cp": b2_res["earliest_checkpoint"],
                "d1_alarm": bool(d1_res["alarm"]),
                "d1_lead": float(d1_res["lead_time_ms"]),
                "d1_cp": d1_res["earliest_checkpoint"],
                "d2_alarm": bool(d2_res["alarm"]),
                "d2_lead": float(d2_res["lead_time_ms"]),
                "d2_cp": d2_res["earliest_checkpoint"],
            })

    return pd.DataFrame(records)


def compute_metrics_for_slice(
    df_slice: pd.DataFrame,
    detector_id: str,
    alarm_col: str,
    lead_col: str,
    cp_col: str,
    slice_id: Any,
    wind_name: str,
    clutter_name: str,
    contention_name: str,
) -> dict:
    """Computes full classification, lead time, and checkpoint metrics for a slice."""
    y_true = df_slice["actual_miss"].values.astype(bool)
    y_pred = df_slice[alarm_col].values.astype(bool)

    n_total = len(y_true)
    n_miss = int(np.sum(y_true))
    n_safe = n_total - n_miss

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    tn = int(np.sum((~y_true) & (~y_pred)))

    rec = float(tp / n_miss * 100.0) if n_miss > 0 else 0.0
    prec = float(tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fpr = float(fp / n_safe * 100.0) if n_safe > 0 else 0.0
    alarm_rate = float((tp + fp) / n_total * 100.0) if n_total > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec) / 100.0) if (prec + rec) > 0 else 0.0
    acc = float((tp + tn) / n_total * 100.0) if n_total > 0 else 0.0

    all_leads = df_slice[y_pred][lead_col].values
    tp_leads = df_slice[y_true & y_pred][lead_col].values

    mean_lead = float(np.mean(all_leads)) if len(all_leads) > 0 else 0.0
    p50_lead = float(np.median(all_leads)) if len(all_leads) > 0 else 0.0
    tp_mean_lead = float(np.mean(tp_leads)) if len(tp_leads) > 0 else 0.0
    tp_p50_lead = float(np.median(tp_leads)) if len(tp_leads) > 0 else 0.0

    cps = df_slice[y_pred][cp_col].values
    n1_cnt = int(np.sum(cps == 1))
    n2_cnt = int(np.sum(cps == 2))
    n3_n4_cnt = int(np.sum((cps == 3) | (cps == 4)))
    n5_cnt = int(np.sum(cps == 5))
    n6_cnt = int(np.sum(cps == 6))

    return {
        "detector_id": detector_id,
        "regime_id": str(slice_id),
        "wind_name": wind_name,
        "clutter_name": clutter_name,
        "contention_name": contention_name,
        "n_total": n_total,
        "n_miss": n_miss,
        "n_safe": n_safe,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall_pct": round(rec, 4),
        "precision_pct": round(prec, 4),
        "fpr_pct": round(fpr, 4),
        "alarm_rate_pct": round(alarm_rate, 4),
        "f1_score": round(f1, 4),
        "accuracy_pct": round(acc, 4),
        "mean_lead_time_ms": round(mean_lead, 4),
        "lead_time_p50_ms": round(p50_lead, 4),
        "tp_mean_lead_time_ms": round(tp_mean_lead, 4),
        "tp_lead_time_p50_ms": round(tp_p50_lead, 4),
        "first_alarm_n1": n1_cnt,
        "first_alarm_n2": n2_cnt,
        "first_alarm_n3_n4": n3_n4_cnt,
        "first_alarm_n5": n5_cnt,
        "first_alarm_n6": n6_cnt,
    }


def run_cluster_bootstrap(
    eval_df: pd.DataFrame,
    deadlines_meta: dict,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Performs 2,000 cluster-bootstraps resampled strictly by episode_id
    across all 18 individual regimes and pooled test traces.
    """
    rng = np.random.RandomState(seed)
    ci_records = []

    # Evaluate for 18 regimes and pooled
    slices: List[Tuple[Any, pd.DataFrame]] = []
    for r in range(18):
        slices.append((str(r), eval_df[eval_df["regime_id"] == r]))
    slices.append(("pooled", eval_df))

    for slice_id, df_slice in slices:
        unique_eps = df_slice["episode_id"].unique()
        n_eps = len(unique_eps)

        y_true = df_slice["actual_miss"].values.astype(bool)

        def get_counts(alarm_col: str):
            y_pred = df_slice[alarm_col].values.astype(bool)
            tp_arr = np.array([np.sum((df_slice["episode_id"] == ep) & y_true & y_pred) for ep in unique_eps])
            fp_arr = np.array([np.sum((df_slice["episode_id"] == ep) & (~y_true) & y_pred) for ep in unique_eps])
            return tp_arr, fp_arr

        b1_tp, b1_fp = get_counts("b1_alarm")
        b2_tp, b2_fp = get_counts("b2_alarm")
        d1_tp, d1_fp = get_counts("d1_alarm")
        d2_tp, d2_fp = get_counts("d2_alarm")

        boot_indices = rng.choice(n_eps, size=(n_boot, n_eps), replace=True)

        def boot_prec_arr(tp_arr, fp_arr):
            tp_b = tp_arr[boot_indices].sum(axis=1)
            fp_b = fp_arr[boot_indices].sum(axis=1)
            tot = tp_b + fp_b
            res = np.zeros(n_boot, dtype=np.float64)
            mask = tot > 0
            res[mask] = tp_b[mask] / tot[mask] * 100.0
            return res

        b1_prec = boot_prec_arr(b1_tp, b1_fp)
        b2_prec = boot_prec_arr(b2_tp, b2_fp)
        d1_prec = boot_prec_arr(d1_tp, d1_fp)
        d2_prec = boot_prec_arr(d2_tp, d2_fp)

        delta_d1_vs_b1 = d1_prec - b1_prec
        delta_d2_vs_b1 = d2_prec - b1_prec
        delta_d2_vs_d1 = d2_prec - d1_prec

        d1_mean_delta = float(np.mean(delta_d1_vs_b1))
        d1_lo = float(np.percentile(delta_d1_vs_b1, 2.5))
        d1_hi = float(np.percentile(delta_d1_vs_b1, 97.5))
        d1_passes = bool(d1_mean_delta >= 5.0 and d1_lo > 0.0)

        d2_mean_delta = float(np.mean(delta_d2_vs_b1))
        d2_lo = float(np.percentile(delta_d2_vs_b1, 2.5))
        d2_hi = float(np.percentile(delta_d2_vs_b1, 97.5))
        d2_passes = bool(d2_mean_delta >= 5.0 and d2_lo > 0.0)

        # TP Lead time calculation for budget check
        b2_tp_leads = df_slice[y_true & df_slice["b2_alarm"]]["b2_lead"].values
        d1_tp_leads = df_slice[y_true & df_slice["d1_alarm"]]["d1_lead"].values
        d2_tp_leads = df_slice[y_true & df_slice["d2_alarm"]]["d2_lead"].values

        b2_lead_mean = float(np.mean(b2_tp_leads)) if len(b2_tp_leads) > 0 else 0.0
        d1_lead_mean = float(np.mean(d1_tp_leads)) if len(d1_tp_leads) > 0 else 0.0
        d2_lead_mean = float(np.mean(d2_tp_leads)) if len(d2_tp_leads) > 0 else 0.0

        if b2_lead_mean > 0.0:
            d1_lead_deg = float((d1_lead_mean - b2_lead_mean) / b2_lead_mean * 100.0)
            d2_lead_deg = float((d2_lead_mean - b2_lead_mean) / b2_lead_mean * 100.0)
        else:
            d1_lead_deg = 0.0
            d2_lead_deg = 0.0

        # PREREG §9.6: "without losing lead time > 20%"
        lead_ok = bool(d1_lead_deg >= -20.0 or d2_lead_deg >= -20.0)

        ci_records.append({
            "regime_id": slice_id,
            "b1_prec_mean": round(float(np.mean(b1_prec)), 4),
            "b1_prec_ci_lo": round(float(np.percentile(b1_prec, 2.5)), 4),
            "b1_prec_ci_hi": round(float(np.percentile(b1_prec, 97.5)), 4),
            "b2_prec_mean": round(float(np.mean(b2_prec)), 4),
            "b2_prec_ci_lo": round(float(np.percentile(b2_prec, 2.5)), 4),
            "b2_prec_ci_hi": round(float(np.percentile(b2_prec, 97.5)), 4),
            "d1_prec_mean": round(float(np.mean(d1_prec)), 4),
            "d1_prec_ci_lo": round(float(np.percentile(d1_prec, 2.5)), 4),
            "d1_prec_ci_hi": round(float(np.percentile(d1_prec, 97.5)), 4),
            "d2_prec_mean": round(float(np.mean(d2_prec)), 4),
            "d2_prec_ci_lo": round(float(np.percentile(d2_prec, 2.5)), 4),
            "d2_prec_ci_hi": round(float(np.percentile(d2_prec, 97.5)), 4),
            "delta_d1_vs_b1_mean": round(d1_mean_delta, 4),
            "delta_d1_vs_b1_ci_lo": round(d1_lo, 4),
            "delta_d1_vs_b1_ci_hi": round(d1_hi, 4),
            "d1_vs_b1_passes_gate": d1_passes,
            "delta_d2_vs_b1_mean": round(d2_mean_delta, 4),
            "delta_d2_vs_b1_ci_lo": round(d2_lo, 4),
            "delta_d2_vs_b1_ci_hi": round(d2_hi, 4),
            "d2_vs_b1_passes_gate": d2_passes,
            "delta_d2_vs_d1_mean": round(float(np.mean(delta_d2_vs_d1)), 4),
            "delta_d2_vs_d1_ci_lo": round(float(np.percentile(delta_d2_vs_d1, 2.5)), 4),
            "delta_d2_vs_d1_ci_hi": round(float(np.percentile(delta_d2_vs_d1, 97.5)), 4),
            "lead_degradation_d1_vs_b2_pct": round(d1_lead_deg, 2),
            "lead_degradation_d2_vs_b2_pct": round(d2_lead_deg, 2),
            "lead_within_20pct_budget": lead_ok,
        })

    return pd.DataFrame(ci_records)


def audit_control_regimes(
    summary_df: pd.DataFrame,
    deadlines_meta: dict,
) -> pd.DataFrame:
    """
    Constructs the stationary control regime audit across the 18 individual regimes.
    """
    records = []
    for r in range(18):
        meta = deadlines_meta[str(r)]
        sub = summary_df[summary_df["regime_id"] == str(r)]

        b2_row = sub[sub["detector_id"] == "per_regime_toba_oracle"].iloc[0]
        d1_row = sub[sub["detector_id"] == "detector_d1"].iloc[0]
        d2_row = sub[sub["detector_id"] == "detector_d2"].iloc[0]

        b2_prec = float(b2_row["precision_pct"])
        d1_prec = float(d1_row["precision_pct"])
        d2_prec = float(d2_row["precision_pct"])

        delta_d1 = round(d1_prec - b2_prec, 4)
        delta_d2 = round(d2_prec - b2_prec, 4)

        if r == 0:
            attribution = "null_control"
        elif abs(delta_d1) <= 1.0:
            attribution = "environment_awareness"
        else:
            attribution = "transition_adaptation"

        records.append({
            "regime_id": r,
            "wind_name": meta["wind_name"],
            "clutter_name": meta["clutter_name"],
            "contention_name": meta["contention_name"],
            "b2_precision": b2_prec,
            "d1_precision": d1_prec,
            "d2_precision": d2_prec,
            "delta_d1_vs_b2_prec": delta_d1,
            "delta_d2_vs_b2_prec": delta_d2,
            "attribution": attribution,
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gate E Confirmatory Test Split")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root path")
    parser.add_argument("--test-traces", type=str, default=None, help="Path to test traces CSV")
    parser.add_argument("--scale", type=int, default=10, help="Temporal scale (bins/ms)")
    parser.add_argument("--policy", type=str, default="matched_recall_95", help="Threshold policy")
    parser.add_argument("--window", type=int, default=5, help="Observable state moving median window")
    parser.add_argument("--fork-mode", type=str, default="wall", choices=["wall", "max"], help="Super-node fork mode")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of cluster bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSVs")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("GATE E CONFIRMATORY EVALUATION ON FRESH TEST SPLIT")
    print("=" * 80)

    # Load datasets
    print(f"\nLoading calibration and fresh test traces...")
    calib_df, test_df, deadlines, deadlines_meta = load_datasets_and_deadlines(
        args.repo_root, test_traces_path=args.test_traces
    )
    print(f"Loaded {len(calib_df)} calibration rows, {len(test_df)} test rows across {len(deadlines)} regimes.")
    print(f"Episodes: {test_df['episode_id'].nunique()} independent test episodes.")

    # Fit detectors strictly on calibration data
    print(f"\nFitting 4 detector configurations on calibration traces (scale={args.scale}, policy={args.policy}, window={args.window}, fork_mode={args.fork_mode})...")
    t0 = time.perf_counter()
    models = fit_all_detectors(
        calib_df, deadlines,
        scale=args.scale, policy=args.policy, window=args.window, fork_mode=args.fork_mode,
    )
    t_fit = time.perf_counter()
    print(f"Detectors successfully fitted in {t_fit - t0:.2f} s.")

    # Evaluate test stream
    print(f"\nEvaluating stream across {len(test_df['episode_id'].unique())} test episodes (7,200 frames)...")
    eval_df = evaluate_stream(test_df, models)
    t_eval = time.perf_counter()
    print(f"Stream evaluation finished in {t_eval - t_fit:.2f} s.")

    # Compute test summary across 19 slices x 4 detectors
    print("\nComputing slice metrics across 18 regimes and pooled...")
    summary_records = []
    detectors = [
        ("pooled_toba", "b1_alarm", "b1_lead", "b1_cp"),
        ("per_regime_toba_oracle", "b2_alarm", "b2_lead", "b2_cp"),
        ("detector_d1", "d1_alarm", "d1_lead", "d1_cp"),
        ("detector_d2", "d2_alarm", "d2_lead", "d2_cp"),
    ]

    # Individual regimes
    for r in range(18):
        sub = eval_df[eval_df["regime_id"] == r]
        meta = deadlines_meta[str(r)]
        for d_id, a_col, l_col, cp_col in detectors:
            summary_records.append(compute_metrics_for_slice(
                sub, d_id, a_col, l_col, cp_col,
                slice_id=r,
                wind_name=meta["wind_name"],
                clutter_name=meta["clutter_name"],
                contention_name=meta["contention_name"],
            ))

    # Pooled slice
    for d_id, a_col, l_col, cp_col in detectors:
        summary_records.append(compute_metrics_for_slice(
            eval_df, d_id, a_col, l_col, cp_col,
            slice_id="pooled",
            wind_name="all",
            clutter_name="all",
            contention_name="all",
        ))

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(args.output_dir, "gate_e_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved test summary (76 rows) -> {summary_path}")

    # Run Cluster-Bootstrapping
    print(f"\nExecuting {args.n_boot} episode-level cluster-bootstraps for paired difference CIs (seed={args.seed})...")
    t_b0 = time.perf_counter()
    boot_df = run_cluster_bootstrap(
        eval_df, deadlines_meta, n_boot=args.n_boot, seed=args.seed
    )
    t_boot = time.perf_counter()
    print(f"Cluster-bootstrapping completed in {t_boot - t_b0:.2f} s.")
    boot_path = os.path.join(args.output_dir, "gate_e_bootstrap_ci.csv")
    boot_df.to_csv(boot_path, index=False)
    print(f"Saved paired bootstrap CIs (19 rows) -> {boot_path}")

    # Audit Control Regimes
    control_df = audit_control_regimes(summary_df, deadlines_meta)
    control_path = os.path.join(args.output_dir, "gate_e_control_regimes.csv")
    control_df.to_csv(control_path, index=False)
    print(f"Saved control regimes audit (18 rows) -> {control_path}")

    # Console Reporting
    print("\n" + "=" * 80)
    print("GATE E POOLED TEST SUMMARY (7,200 frames, 144 episodes)")
    print("=" * 80)
    pooled_sub = summary_df[summary_df["regime_id"] == "pooled"]
    for _, row in pooled_sub.iterrows():
        print(
            f"Detector: {row['detector_id']:25s} | "
            f"Recall: {row['recall_pct']:5.2f}% | "
            f"Precision: {row['precision_pct']:5.2f}% | "
            f"FPR: {row['fpr_pct']:5.2f}% | "
            f"Alarm Rate: {row['alarm_rate_pct']:5.2f}% | "
            f"TP Lead: {row['tp_mean_lead_time_ms']:5.2f} ms"
        )

    print("\n" + "=" * 80)
    print(f"PAIRED CLUSTER-BOOTSTRAP 95% CIs (Episode-Clustered across 144 episodes, B={args.n_boot})")
    print("=" * 80)
    pooled_ci = boot_df[boot_df["regime_id"] == "pooled"].iloc[0]
    print(f"Delta Prec (D1 vs B1): {pooled_ci['delta_d1_vs_b1_mean']:+.2f} pp [95% CI: {pooled_ci['delta_d1_vs_b1_ci_lo']:+.2f}, {pooled_ci['delta_d1_vs_b1_ci_hi']:+.2f}] pp (Gate Pass: {pooled_ci['d1_vs_b1_passes_gate']})")
    print(f"Delta Prec (D2 vs B1): {pooled_ci['delta_d2_vs_b1_mean']:+.2f} pp [95% CI: {pooled_ci['delta_d2_vs_b1_ci_lo']:+.2f}, {pooled_ci['delta_d2_vs_b1_ci_hi']:+.2f}] pp (Gate Pass: {pooled_ci['d2_vs_b1_passes_gate']})")
    print(f"Delta Prec (D2 vs D1): {pooled_ci['delta_d2_vs_d1_mean']:+.2f} pp [95% CI: {pooled_ci['delta_d2_vs_d1_ci_lo']:+.2f}, {pooled_ci['delta_d2_vs_d1_ci_hi']:+.2f}] pp")
    print(f"Lead Degradation (D1 vs B2): {pooled_ci['lead_degradation_d1_vs_b2_pct']:+.2f}% | Lead Within Budget: {pooled_ci['lead_within_20pct_budget']}")

    print("\n" + "=" * 80)
    print("STATIONARY CONTROL REGIME AUDIT (Regime 0: Zero Wind / Low Clutter / Idle)")
    print("=" * 80)
    ctrl0 = control_df[control_df["regime_id"] == 0].iloc[0]
    print(f"Regime 0: B2 Prec = {ctrl0['b2_precision']:.2f}%, D1 Prec = {ctrl0['d1_precision']:.2f}%, D2 Prec = {ctrl0['d2_precision']:.2f}%")
    print(f"Delta D1 vs B2 = {ctrl0['delta_d1_vs_b2_prec']:+.2f} pp | Attribution: {ctrl0['attribution']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
