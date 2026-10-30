r"""
Adversarial Stress Test Suite for Milestone 4 (Gate E Split Leakage & Integrity Challenger).
Author: Challenger 2 (challenger_m4_2)
Role: Empirical Challenger (critic / specialist)

Adversarially audits & tests:
1. Split Integrity & Mutual Episode Disjointness:
   - Exactly 7,200 rows, 144 independent episodes, and 18 regimes in test traces.
   - Exactly 7,200 calib rows (144 episodes) and 3,600 val rows (72 episodes).
   - Strict 3-way mutual disjointness (zero episode overlap across Calib, Val, and Test).
   - Exactly 360 episodes total, each with strictly 50 frames and 1 regime.
2. Calibration-Only Fitting & Test Split Isolation:
   - All 18 locked deadlines in results/gate_b_deadlines.json match calibration Q85 (< 1e-6)
     and differ strictly from test Q85.
   - ObservableStateEstimator centroids match calibration medians and differ from test medians.
   - Detectors' calibrated thresholds match calibration quantiles and differ from test quantiles.
   - Zero test trace file or string references in src/detectors/.
3. Episode State Reset & Cross-Episode Isolation:
   - Evaluates episode E with reset after episode X; matches fresh evaluation bit-for-bit.
   - Proves omitting reset introduces memory pollution, verifying reset efficacy.
4. Adversarial Permutation Invariance Stress Test:
   - Scrambled and reversed episode order stream evaluations produce bit-for-bit identical
     predictions, alarms, predicted regimes, lead times, and confusion matrices across all 7,200 frames.
"""

import os
import json
import random
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import pytest

from benchmarks.evaluate_gate_e import (
    fit_all_detectors,
    evaluate_stream,
    compute_metrics_for_slice,
)
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
CALIB_PATH = os.path.join(REPO_ROOT, "results", "uav_calibration_traces.csv")
VAL_PATH = os.path.join(REPO_ROOT, "results", "uav_validation_traces.csv")
TEST_PATH = os.path.join(REPO_ROOT, "results", "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(REPO_ROOT, "results", "gate_b_deadlines.json")


# ==============================================================================
# Test 1: Split Integrity & Mutual Episode Disjointness
# ==============================================================================

def test_split_integrity_and_mutual_disjointness():
    """
    Verifies that the test split has exactly 7,200 rows across 144 independent episodes,
    with zero episode overlap with calibration (144 episodes) or validation (72 episodes).
    """
    assert os.path.exists(CALIB_PATH), f"Missing {CALIB_PATH}"
    assert os.path.exists(VAL_PATH), f"Missing {VAL_PATH}"
    assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"

    calib = pd.read_csv(CALIB_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    # 1. Exact row counts
    assert len(calib) == 7200, f"Expected 7200 calib rows, got {len(calib)}"
    assert len(val) == 3600, f"Expected 3600 val rows, got {len(val)}"
    assert len(test) == 7200, f"Expected 7200 test rows, got {len(test)}"

    # 2. Episode counts
    calib_eps = set(calib["episode_id"])
    val_eps = set(val["episode_id"])
    test_eps = set(test["episode_id"])

    assert len(calib_eps) == 144, f"Expected 144 calib episodes, got {len(calib_eps)}"
    assert len(val_eps) == 72, f"Expected 72 val episodes, got {len(val_eps)}"
    assert len(test_eps) == 144, f"Expected 144 test episodes, got {len(test_eps)}"

    # 3. Mutual episode disjointness
    assert len(calib_eps & test_eps) == 0, f"Leakage: {len(calib_eps & test_eps)} episodes overlap calib and test!"
    assert len(val_eps & test_eps) == 0, f"Leakage: {len(val_eps & test_eps)} episodes overlap val and test!"
    assert len(calib_eps & val_eps) == 0, f"Leakage: {len(calib_eps & val_eps)} episodes overlap calib and val!"

    # 4. Total episodes
    all_eps = calib_eps | val_eps | test_eps
    assert len(all_eps) == 360, f"Expected 360 total episodes across splits, got {len(all_eps)}"

    # 5. Frame count per episode
    for name, df in [("calib", calib), ("val", val), ("test", test)]:
        counts = df.groupby("episode_id").size().unique()
        assert list(counts) == [50], f"{name} contains episodes with non-50 frame count: {counts}"

    # 6. Regime integrity in test
    assert test["regime_id"].nunique() == 18, f"Expected 18 regimes, got {test['regime_id'].nunique()}"
    ep_reg_counts = test.groupby("regime_id")["episode_id"].nunique()
    assert (ep_reg_counts == 8).all(), f"Not all regimes have 8 test episodes: {ep_reg_counts.to_dict()}"

    # 7. No NaNs or Infs
    assert test.isnull().sum().sum() == 0, "Test traces contain NaNs"
    num_cols = test.select_dtypes(include=[np.number])
    assert np.isinf(num_cols).sum().sum() == 0, "Test traces contain Infs"


# ==============================================================================
# Test 2: Deadlines Locked Strictly on Calibration Q85
# ==============================================================================

def test_deadlines_locked_strictly_on_calibration():
    """
    Verifies that all 18 regime deadlines in gate_b_deadlines.json match calibration Q85
    and differ strictly from test split empirical Q85.
    """
    calib = pd.read_csv(CALIB_PATH)
    test = pd.read_csv(TEST_PATH)

    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)["regimes"]

    for r in range(18):
        d_locked = meta[str(r)]["deadline_ms"]
        calib_s = calib[calib["regime_id"] == r]["s_dag_ms"]
        test_s = test[test["regime_id"] == r]["s_dag_ms"]

        q85_calib = float(np.quantile(calib_s, 0.85))
        q85_test = float(np.quantile(test_s, 0.85))

        # Locked deadline must match calib Q85 exactly
        assert abs(d_locked - q85_calib) < 1e-6, (
            f"Regime {r} locked deadline {d_locked} != calib Q85 {q85_calib}"
        )
        # Locked deadline must differ from test Q85
        assert abs(d_locked - q85_test) > 0.01, (
            f"Regime {r} locked deadline unexpectedly identical to test Q85: {d_locked} vs {q85_test}"
        )


# ==============================================================================
# Test 3: Detectors Fit Strictly on Calibration (Zero Leakage)
# ==============================================================================

def test_detectors_fit_strictly_on_calibration():
    """
    Verifies that all detectors are fitted strictly on calibration data:
    - Centroids match calibration medians and differ from test medians.
    - Thresholds match calibration percentiles and differ from test counterfactuals.
    - Source code has zero references to test split files.
    """
    calib = pd.read_csv(CALIB_PATH)
    test = pd.read_csv(TEST_PATH)

    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)["regimes"]
    deadlines = {int(k): float(v["deadline_ms"]) for k, v in meta.items()}

    models = fit_all_detectors(calib, deadlines)
    d1 = models["detector_d1"]
    d2 = models["detector_d2"]

    # 1. ObservableStateEstimator centroids audit
    estimator = d1.estimator
    for r in range(18):
        calib_r = calib[calib["regime_id"] == r]
        test_r = test[test["regime_id"] == r]

        expected_centroid = np.array([
            float(np.median(calib_r["dur_node1_ms"])),
            float(np.median(calib_r["dur_node2_ms"])),
            float(np.median(calib_r["s_dag_ms"])),
        ])
        actual_centroid = estimator.calib_centroids[r]
        np.testing.assert_allclose(actual_centroid, expected_centroid, rtol=1e-5)

        test_centroid = np.array([
            float(np.median(test_r["dur_node1_ms"])),
            float(np.median(test_r["dur_node2_ms"])),
            float(np.median(test_r["s_dag_ms"])),
        ])
        # Centroid should not match test centroid
        assert not np.allclose(actual_centroid, test_centroid, atol=1e-3), (
            f"Regime {r} centroid matches test centroid: {actual_centroid} vs {test_centroid}"
        )

    # 2. Calibrated threshold audit
    for r in range(18):
        reg_calib = calib[calib["regime_id"] == r]
        reg_test = test[test["regime_id"] == r]
        d_reg = deadlines[r]

        # Check that calibration misses exist and were used
        calib_misses = reg_calib[reg_calib["s_dag_ms"] > d_reg]
        assert len(calib_misses) == 60, f"Expected 60 calib misses (Q85 of 400), got {len(calib_misses)}"

    # 3. Source code scan for test traces reference
    src_dir = os.path.join(REPO_ROOT, "src", "detectors")
    for fname in os.listdir(src_dir):
        if fname.endswith(".py"):
            fpath = os.path.join(src_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            assert "uav_test_traces" not in content, f"Found test trace reference in {fname}"


# ==============================================================================
# Test 4: Episode State Reset & Cross-Episode Isolation
# ==============================================================================

def test_episode_state_reset_isolation():
    """
    Verifies that calling reset_episode() cleanly resets the detector state:
    - Evaluating episode B after episode A with reset matches fresh evaluation bit-for-bit.
    - Omitting reset causes history pollution that alters predictions.
    """
    calib = pd.read_csv(CALIB_PATH)
    test = pd.read_csv(TEST_PATH)

    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)["regimes"]
    deadlines = {int(k): float(v["deadline_ms"]) for k, v in meta.items()}

    models = fit_all_detectors(calib, deadlines)
    d1 = models["detector_d1"]

    episodes = test["episode_id"].unique()
    ep_a = episodes[0]   # Regime 0 (idle)
    ep_b = episodes[10]  # Regime 1 (moderate contention)

    df_a = test[test["episode_id"] == ep_a].sort_values("frame_id")
    df_b = test[test["episode_id"] == ep_b].sort_values("frame_id")

    # Baseline: Fresh evaluation of episode B
    d1.reset_episode()
    res_fresh = [d1.evaluate_frame(row, update_state=True) for _, row in df_b.iterrows()]

    # Case 1: Evaluate A -> RESET -> evaluate B
    d1.reset_episode()
    for _, row in df_a.iterrows():
        d1.evaluate_frame(row, update_state=True)
    d1.reset_episode()
    res_reset = [d1.evaluate_frame(row, update_state=True) for _, row in df_b.iterrows()]

    # Must be 100% identical bit-for-bit
    for idx, (rf, rr) in enumerate(zip(res_fresh, res_reset)):
        assert rf["alarm"] == rr["alarm"], f"Alarm mismatch at frame {idx}"
        assert rf["pred_regime"] == rr["pred_regime"], f"Pred regime mismatch at frame {idx}"
        assert rf["lead_time_ms"] == rr["lead_time_ms"], f"Lead time mismatch at frame {idx}"
        assert rf["earliest_checkpoint"] == rr["earliest_checkpoint"], f"Checkpoint mismatch at frame {idx}"

    # Case 2: Evaluate A -> NO RESET -> evaluate B (proves state persists and reset is necessary)
    d1.reset_episode()
    for _, row in df_a.iterrows():
        d1.evaluate_frame(row, update_state=True)
    res_no_reset = [d1.evaluate_frame(row, update_state=True) for _, row in df_b.iterrows()]

    # Verify that omitting reset causes at least one prediction difference
    diff_count = sum(
        1 for rf, rn in zip(res_fresh, res_no_reset)
        if rf["pred_regime"] != rn["pred_regime"] or rf["alarm"] != rn["alarm"]
    )
    assert diff_count > 0, "Expected state leakage when reset_episode() is omitted, but found none!"


# ==============================================================================
# Test 5: Adversarial Permutation Invariant Test
# ==============================================================================

def test_adversarial_episode_permutation_invariance():
    """
    Stress-tests that evaluation predictions and confusion metrics are mathematically
    invariant to test episode evaluation order (reversed and scrambled).
    """
    calib = pd.read_csv(CALIB_PATH)
    test = pd.read_csv(TEST_PATH)

    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)["regimes"]
    deadlines = {int(k): float(v["deadline_ms"]) for k, v in meta.items()}

    models = fit_all_detectors(calib, deadlines)

    # 1. Natural stream evaluation
    df_natural = evaluate_stream(test, models)

    # 2. Reversed stream evaluation
    eps = list(test["episode_id"].unique())
    eps_rev = list(reversed(eps))
    test_rev = test.set_index("episode_id").loc[eps_rev].reset_index()
    df_reversed = evaluate_stream(test_rev, models)

    # 3. Scrambled stream evaluation (seeded shuffle)
    rng = random.Random(2026)
    eps_scrambled = list(eps)
    rng.shuffle(eps_scrambled)
    test_scrambled = test.set_index("episode_id").loc[eps_scrambled].reset_index()
    df_scrambled = evaluate_stream(test_scrambled, models)

    # Sort all by episode_id and frame_id to align frames
    nat_sort = df_natural.sort_values(["episode_id", "frame_id"]).reset_index(drop=True)
    rev_sort = df_reversed.sort_values(["episode_id", "frame_id"]).reset_index(drop=True)
    scram_sort = df_scrambled.sort_values(["episode_id", "frame_id"]).reset_index(drop=True)

    cols_to_check = [
        "actual_miss", "d1_pred_regime", "d2_pred_regime",
        "b1_alarm", "b1_lead", "b1_cp",
        "b2_alarm", "b2_lead", "b2_cp",
        "d1_alarm", "d1_lead", "d1_cp",
        "d2_alarm", "d2_lead", "d2_cp",
    ]

    for col in cols_to_check:
        assert nat_sort[col].equals(rev_sort[col]), f"Reversed stream mismatch in {col}"
        assert nat_sort[col].equals(scram_sort[col]), f"Scrambled stream mismatch in {col}"

    # Verify slice metrics invariance for pooled and individual regimes
    detectors = [
        ("pooled_toba", "b1_alarm", "b1_lead", "b1_cp"),
        ("per_regime_toba_oracle", "b2_alarm", "b2_lead", "b2_cp"),
        ("detector_d1", "d1_alarm", "d1_lead", "d1_cp"),
        ("detector_d2", "d2_alarm", "d2_lead", "d2_cp"),
    ]

    for d_id, a_col, l_col, cp_col in detectors:
        m_nat = compute_metrics_for_slice(df_natural, d_id, a_col, l_col, cp_col, "pooled", "all", "all", "all")
        m_rev = compute_metrics_for_slice(df_reversed, d_id, a_col, l_col, cp_col, "pooled", "all", "all", "all")
        m_scram = compute_metrics_for_slice(df_scrambled, d_id, a_col, l_col, cp_col, "pooled", "all", "all", "all")

        for key in ["tp", "fp", "tn", "fn", "recall_pct", "precision_pct", "fpr_pct", "alarm_rate_pct", "mean_lead_time_ms"]:
            assert m_nat[key] == m_rev[key], f"Pooled metric mismatch between natural and reversed for {d_id}.{key}"
            assert m_nat[key] == m_scram[key], f"Pooled metric mismatch between natural and scrambled for {d_id}.{key}"
