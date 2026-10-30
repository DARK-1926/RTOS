r"""
Comprehensive Test Suite for Gate E Confirmatory Test Split Evaluation.

Verifies:
1. Gate E CSV artifacts exist, are non-empty, and conform to exact schemas.
2. Test split integrity: row count is exactly 7,200 across 144 independent episodes.
3. Pre-registered success criteria: pooled Delta Precision (D1 vs B1) >= +5.0 pp
   with 95% cluster CI strictly excluding 0.
4. Candidate D2 performance: pooled Delta Precision (D2 vs B1) >= +5.0 pp with CI excluding 0.
5. Lead time preservation budget: lead degradation relative to Baseline 2 does not lose > 20%.
6. Stationary control regime audit: Regime 0 Delta Precision <= 1.0 pp (null control).
7. Physical alarm distribution soundness and mathematical confusion matrix accounting.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIB_PATH = os.path.join(REPO_ROOT, "results", "uav_calibration_traces.csv")
VAL_PATH = os.path.join(REPO_ROOT, "results", "uav_validation_traces.csv")
TEST_PATH = os.path.join(REPO_ROOT, "results", "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(REPO_ROOT, "results", "gate_b_deadlines.json")

GATE_E_SUMMARY_PATH = os.path.join(REPO_ROOT, "results", "gate_e_test_summary.csv")
GATE_E_BOOTSTRAP_PATH = os.path.join(REPO_ROOT, "results", "gate_e_bootstrap_ci.csv")
GATE_E_CONTROL_PATH = os.path.join(REPO_ROOT, "results", "gate_e_control_regimes.csv")


# ==============================================================================
# Test 1: Gate E Artifacts Existence and Schema Conformity
# ==============================================================================

def test_gate_e_artifacts_exist_and_conform_to_schema():
    """
    Verifies that all three Gate E output CSV files exist, are non-empty,
    contain exact row counts, and match the specified schemas.
    """
    assert os.path.exists(GATE_E_SUMMARY_PATH), f"Missing {GATE_E_SUMMARY_PATH}"
    assert os.path.exists(GATE_E_BOOTSTRAP_PATH), f"Missing {GATE_E_BOOTSTRAP_PATH}"
    assert os.path.exists(GATE_E_CONTROL_PATH), f"Missing {GATE_E_CONTROL_PATH}"

    summary_df = pd.read_csv(GATE_E_SUMMARY_PATH)
    bootstrap_df = pd.read_csv(GATE_E_BOOTSTRAP_PATH)
    control_df = pd.read_csv(GATE_E_CONTROL_PATH)

    # 1. Summary: 76 rows (19 slices x 4 detectors)
    assert len(summary_df) == 76, f"Expected 76 rows in summary, got {len(summary_df)}"
    summary_cols = [
        "detector_id", "regime_id", "wind_name", "clutter_name", "contention_name",
        "n_total", "n_miss", "n_safe", "tp", "fp", "tn", "fn",
        "recall_pct", "precision_pct", "fpr_pct", "alarm_rate_pct", "f1_score", "accuracy_pct",
        "mean_lead_time_ms", "lead_time_p50_ms", "tp_mean_lead_time_ms", "tp_lead_time_p50_ms",
        "first_alarm_n1", "first_alarm_n2", "first_alarm_n3_n4", "first_alarm_n5", "first_alarm_n6",
    ]
    assert list(summary_df.columns) == summary_cols, "Summary columns mismatch"
    assert summary_df.isnull().sum().sum() == 0, "Summary CSV contains NaNs"

    # 2. Bootstrap: 19 rows (18 regimes + pooled)
    assert len(bootstrap_df) == 19, f"Expected 19 rows in bootstrap, got {len(bootstrap_df)}"
    bootstrap_cols = [
        "regime_id",
        "b1_prec_mean", "b1_prec_ci_lo", "b1_prec_ci_hi",
        "b2_prec_mean", "b2_prec_ci_lo", "b2_prec_ci_hi",
        "d1_prec_mean", "d1_prec_ci_lo", "d1_prec_ci_hi",
        "d2_prec_mean", "d2_prec_ci_lo", "d2_prec_ci_hi",
        "delta_d1_vs_b1_mean", "delta_d1_vs_b1_ci_lo", "delta_d1_vs_b1_ci_hi", "d1_vs_b1_passes_gate",
        "delta_d2_vs_b1_mean", "delta_d2_vs_b1_ci_lo", "delta_d2_vs_b1_ci_hi", "d2_vs_b1_passes_gate",
        "delta_d2_vs_d1_mean", "delta_d2_vs_d1_ci_lo", "delta_d2_vs_d1_ci_hi",
        "lead_degradation_d1_vs_b2_pct", "lead_degradation_d2_vs_b2_pct", "lead_within_20pct_budget",
    ]
    assert list(bootstrap_df.columns) == bootstrap_cols, "Bootstrap columns mismatch"
    assert bootstrap_df.isnull().sum().sum() == 0, "Bootstrap CSV contains NaNs"

    # 3. Control Regimes: 18 rows
    assert len(control_df) == 18, f"Expected 18 rows in control, got {len(control_df)}"
    control_cols = [
        "regime_id", "wind_name", "clutter_name", "contention_name",
        "b2_precision", "d1_precision", "d2_precision",
        "delta_d1_vs_b2_prec", "delta_d2_vs_b2_prec", "attribution",
    ]
    assert list(control_df.columns) == control_cols, "Control columns mismatch"
    assert control_df.isnull().sum().sum() == 0, "Control CSV contains NaNs"


# ==============================================================================
# Test 2: Fresh Test Split Integrity & Mutual Episode Disjointness
# ==============================================================================

def test_gate_e_test_split_integrity():
    """
    Verifies that the test split traces file contains exactly 7,200 rows across
    144 independent episodes and has zero episode overlap with calib or val splits.
    """
    assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
    test_df = pd.read_csv(TEST_PATH)
    calib_df = pd.read_csv(CALIB_PATH)
    val_df = pd.read_csv(VAL_PATH)

    assert len(test_df) == 7200, f"Expected 7200 test rows, got {len(test_df)}"
    assert test_df["episode_id"].nunique() == 144, f"Expected 144 episodes, got {test_df['episode_id'].nunique()}"
    assert (test_df["split"] == "test").all(), "Test traces contain invalid split tags"
    assert test_df["regime_id"].nunique() == 18, "Test traces do not contain all 18 regimes"

    # Check for NaN / Inf
    assert test_df.isnull().sum().sum() == 0, "Test traces contain NaN values"
    num_cols = test_df.select_dtypes(include=[np.number])
    assert np.isinf(num_cols).sum().sum() == 0, "Test traces contain Inf values"

    # Strict episode isolation
    calib_eps = set(calib_df["episode_id"].unique())
    val_eps = set(val_df["episode_id"].unique())
    test_eps = set(test_df["episode_id"].unique())

    assert len(calib_eps.intersection(test_eps)) == 0, "Episode leakage between calib and test splits!"
    assert len(val_eps.intersection(test_eps)) == 0, "Episode leakage between val and test splits!"


# ==============================================================================
# Test 3: Pre-Registered Success Criteria Confirmation (PREREG.md §9.6)
# ==============================================================================

def test_gate_e_preregistered_success_criteria():
    """
    Confirms pre-registered success criteria on the fresh test split:
    - Candidate D1 pooled Delta Precision (D1 vs B1) >= +5.0 pp.
    - 95% cluster-bootstrap CI strictly excludes 0.
    - d1_vs_b1_passes_gate is True.
    - Candidate D2 pooled Delta Precision (D2 vs B1) >= +5.0 pp with CI excluding 0.
    - Lead time preservation budget: lead degradation relative to Baseline 2 does not lose > 20%.
    """
    bootstrap_df = pd.read_csv(GATE_E_BOOTSTRAP_PATH)
    pooled_ci = bootstrap_df[bootstrap_df["regime_id"] == "pooled"].iloc[0]

    # Primary Success Criterion: D1 vs B1
    assert pooled_ci["delta_d1_vs_b1_mean"] >= 5.0, (
        f"D1 precision gain failed: got {pooled_ci['delta_d1_vs_b1_mean']} pp, required >= 5.0 pp"
    )
    assert pooled_ci["delta_d1_vs_b1_ci_lo"] > 0.0, (
        f"D1 95% CI includes 0: [{pooled_ci['delta_d1_vs_b1_ci_lo']}, {pooled_ci['delta_d1_vs_b1_ci_hi']}]"
    )
    assert bool(pooled_ci["d1_vs_b1_passes_gate"]) is True

    # Candidate D2 Success Criterion: D2 vs B1
    assert pooled_ci["delta_d2_vs_b1_mean"] >= 5.0, (
        f"D2 precision gain failed: got {pooled_ci['delta_d2_vs_b1_mean']} pp, required >= 5.0 pp"
    )
    assert pooled_ci["delta_d2_vs_b1_ci_lo"] > 0.0, (
        f"D2 95% CI includes 0: [{pooled_ci['delta_d2_vs_b1_ci_lo']}, {pooled_ci['delta_d2_vs_b1_ci_hi']}]"
    )
    assert bool(pooled_ci["d2_vs_b1_passes_gate"]) is True

    # Lead time degradation budget check (PREREG.md §9.6: "without losing lead time > 20%")
    assert bool(pooled_ci["lead_within_20pct_budget"]) is True, (
        f"Lead time lost > 20%: D1 deg = {pooled_ci['lead_degradation_d1_vs_b2_pct']}%, "
        f"D2 deg = {pooled_ci['lead_degradation_d2_vs_b2_pct']}%"
    )


# ==============================================================================
# Test 4: Stationary Control Regime Audit (Regime 0 Attribution)
# ==============================================================================

def test_gate_e_stationary_control_regime_audit():
    """
    Verifies the stationary control regime requirement (PREREG.md §9.6):
    In Regime 0 (zero wind, low clutter, idle contention), Candidate D1 must not
    diverge from oracle Baseline 2 (|Delta Prec| <= 1.0 pp), confirming that
    the precision gains in non-stationary regimes are 100% attributable to
    environment awareness rather than artificial threshold shifts.
    """
    control_df = pd.read_csv(GATE_E_CONTROL_PATH)
    r0 = control_df[control_df["regime_id"] == 0].iloc[0]

    assert r0["wind_name"] == "zero"
    assert r0["clutter_name"] == "low"
    assert r0["contention_name"] == "idle"
    assert r0["attribution"] == "null_control"

    assert abs(r0["delta_d1_vs_b2_prec"]) <= 1.0, (
        f"Regime 0 null control violated: Delta Prec = {r0['delta_d1_vs_b2_prec']} pp (must be <= 1.0 pp)"
    )
    assert r0["b2_precision"] == 100.0
    assert r0["d1_precision"] == 100.0


# ==============================================================================
# Test 5: Checkpoint Alarm Distribution Physical Soundness
# ==============================================================================

def test_gate_e_checkpoint_alarm_distribution_soundness():
    """
    Verifies that Baseline 1 suffers from premature alarms at Node 1 across non-stationary
    regimes, whereas D1 and D2 alarm primarily at later nodes (Node 5, Node 6), matching
    authentic DAG execution dynamics.
    """
    summary_df = pd.read_csv(GATE_E_SUMMARY_PATH)
    pooled_sub = summary_df[summary_df["regime_id"] == "pooled"]

    b1_row = pooled_sub[pooled_sub["detector_id"] == "pooled_toba"].iloc[0]
    b2_row = pooled_sub[pooled_sub["detector_id"] == "per_regime_toba_oracle"].iloc[0]
    d1_row = pooled_sub[pooled_sub["detector_id"] == "detector_d1"].iloc[0]
    d2_row = pooled_sub[pooled_sub["detector_id"] == "detector_d2"].iloc[0]

    # Baseline 1 failure mode: premature alarms at DAG entry (Node 1) on >= 6000 frames
    assert b1_row["first_alarm_n1"] >= 6000, (
        f"Expected >= 6000 premature N1 alarms for B1, got {b1_row['first_alarm_n1']}"
    )

    # Oracle B2, Candidate D1, and Candidate D2 have 0 alarms at Node 1
    assert b2_row["first_alarm_n1"] == 0, f"B2 alarmed at N1: {b2_row['first_alarm_n1']}"
    assert d1_row["first_alarm_n1"] == 0, f"D1 alarmed at N1: {d1_row['first_alarm_n1']}"
    assert d2_row["first_alarm_n1"] == 0, f"D2 alarmed at N1: {d2_row['first_alarm_n1']}"

    # D1 and D2 alarms concentrated at downstream join (N5) and terminal node (N6)
    assert (d1_row["first_alarm_n5"] + d1_row["first_alarm_n6"]) >= 1100
    assert (d2_row["first_alarm_n5"] + d2_row["first_alarm_n6"]) >= 1200


# ==============================================================================
# Test 6: Confusion Matrix and Arithmetic Accounting Soundness
# ==============================================================================

def test_gate_e_confusion_matrix_accounting_soundness():
    """
    Verifies strict internal consistency of confusion matrix counts and rate formulas
    for all 76 rows in gate_e_test_summary.csv.
    """
    summary_df = pd.read_csv(GATE_E_SUMMARY_PATH)

    for idx, row in summary_df.iterrows():
        n_total = row["n_total"]
        n_miss = row["n_miss"]
        n_safe = row["n_safe"]
        tp = row["tp"]
        fp = row["fp"]
        tn = row["tn"]
        fn = row["fn"]

        # Basic sum checks
        assert tp + fp + tn + fn == n_total, f"Row {idx}: counts do not sum to n_total"
        assert tp + fn == n_miss, f"Row {idx}: tp + fn != n_miss"
        assert fp + tn == n_safe, f"Row {idx}: fp + tn != n_safe"
        assert n_miss + n_safe == n_total, f"Row {idx}: n_miss + n_safe != n_total"

        # Rate calculations within rounding tolerance
        if n_miss > 0:
            calc_rec = tp / n_miss * 100.0
            assert abs(row["recall_pct"] - calc_rec) <= 0.01, f"Row {idx}: recall mismatch"
        if (tp + fp) > 0:
            calc_prec = tp / (tp + fp) * 100.0
            assert abs(row["precision_pct"] - calc_prec) <= 0.01, f"Row {idx}: precision mismatch"
        if n_safe > 0:
            calc_fpr = fp / n_safe * 100.0
            assert abs(row["fpr_pct"] - calc_fpr) <= 0.01, f"Row {idx}: fpr mismatch"
        if n_total > 0:
            calc_alarm_rate = (tp + fp) / n_total * 100.0
            assert abs(row["alarm_rate_pct"] - calc_alarm_rate) <= 0.01, f"Row {idx}: alarm rate mismatch"
            calc_acc = (tp + tn) / n_total * 100.0
            assert abs(row["accuracy_pct"] - calc_acc) <= 0.01, f"Row {idx}: accuracy mismatch"
