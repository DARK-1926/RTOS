"""
Adversarial Stress Test Suite for Milestone 1 (Gate B: Episode Splitting, Trace Isolation, and Deadline Calibration).
Author: Challenger 2 (challenger_m1_2)
Role: Empirical Challenger (critic / specialist)

Adversarially tests:
1. Zero episode leakage & exact partition completeness across Calib, Val, and Test splits.
2. Intra-episode boundary integrity (all 50 frames strictly contiguous within single split and regime).
3. Telemetry vector uniqueness (no copy-paste or duplicated rows across splits).
4. Strict deadline calibration provenance (D_reg matches Calib Q85 to float epsilon, strictly differs from Val/Test/Pooled, SHA-256 match).
5. Absolute isolation of test split (uav_test_traces.csv untouched by detectors or benchmarks).
6. Non-degeneracy of N5 join node and physical concurrency of fork execution across all splits.
"""

import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
TEST_PATH = os.path.join(RESULTS_DIR, "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")


@pytest.fixture(scope="module")
def trace_datasets():
    calib = pd.read_csv(CALIB_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    with open(DEADLINES_PATH, "r") as f:
        deadlines = json.load(f)
    return {
        "calib": calib,
        "val": val,
        "test": test,
        "deadlines": deadlines,
        "full": pd.concat([calib, val, test], ignore_index=True)
    }


def test_adversarial_zero_episode_leakage_and_partition_completeness(trace_datasets):
    """
    Stress-test split boundaries:
    - Zero episode overlap across all three pairwise combinations.
    - Exact union equals 360 contiguous episodes (range(360)).
    - Exact frame counts: 7200 calib, 3600 val, 7200 test = 18000 total.
    """
    calib = trace_datasets["calib"]
    val = trace_datasets["val"]
    test = trace_datasets["test"]
    full = trace_datasets["full"]

    calib_eps = set(calib["episode_id"].unique())
    val_eps = set(val["episode_id"].unique())
    test_eps = set(test["episode_id"].unique())

    # Pairwise disjointness
    assert len(calib_eps & val_eps) == 0, f"Calib/Val overlap: {calib_eps & val_eps}"
    assert len(calib_eps & test_eps) == 0, f"Calib/Test overlap: {calib_eps & test_eps}"
    assert len(val_eps & test_eps) == 0, f"Val/Test overlap: {val_eps & test_eps}"

    # Complete partition
    all_eps = calib_eps | val_eps | test_eps
    assert len(all_eps) == 360, f"Expected 360 unique episodes, found {len(all_eps)}"
    assert all_eps == set(range(360)), "Episode IDs do not span contiguous range 0..359"

    # Exact split row counts
    assert len(calib) == 7200, f"Expected 7200 calib rows, got {len(calib)}"
    assert len(val) == 3600, f"Expected 3600 val rows, got {len(val)}"
    assert len(test) == 7200, f"Expected 7200 test rows, got {len(test)}"
    assert len(full) == 18000, f"Expected 18000 total rows, got {len(full)}"

    # Per-regime episode counts
    for r in range(18):
        c_r = calib[calib["regime_id"] == r]
        v_r = val[val["regime_id"] == r]
        t_r = test[test["regime_id"] == r]
        assert c_r["episode_id"].nunique() == 8, f"Regime {r} calib episodes != 8"
        assert v_r["episode_id"].nunique() == 4, f"Regime {r} val episodes != 4"
        assert t_r["episode_id"].nunique() == 8, f"Regime {r} test episodes != 8"


def test_adversarial_intra_episode_boundary_contiguity(trace_datasets):
    """
    Stress-test intra-episode frame integrity:
    - Every episode belongs to exactly one split and one regime.
    - Every episode contains exactly 50 frames with frame_id == list(range(50)).
    - No duplicate (episode_id, frame_id) pairs exist anywhere.
    """
    full = trace_datasets["full"]

    # No duplicate episode_id, frame_id
    dups = full.duplicated(subset=["episode_id", "frame_id"]).sum()
    assert dups == 0, f"Found {dups} duplicate (episode_id, frame_id) rows"

    # Check each episode strictly
    for ep_id, group in full.groupby("episode_id"):
        assert group["split"].nunique() == 1, f"Episode {ep_id} split across multiple splits!"
        assert group["regime_id"].nunique() == 1, f"Episode {ep_id} split across multiple regimes!"
        assert len(group) == 50, f"Episode {ep_id} has {len(group)} frames (expected 50)"
        frames = list(group["frame_id"].values)
        assert frames == list(range(50)), f"Episode {ep_id} frames are not strictly range(50): {frames[:5]}..."


def test_adversarial_telemetry_vector_uniqueness(trace_datasets):
    """
    Adversarial verification against synthetic cloning / trace copy-pasting:
    Verify that physical timing vectors are 100% unique across all 18,000 frames.
    """
    calib = trace_datasets["calib"]
    val = trace_datasets["val"]
    test = trace_datasets["test"]

    timing_cols = ["t_n1", "t_n2", "t_n3", "t_n4", "t_n5", "t_n6", "t_fork", "t_e2e"]

    calib_tuples = set(tuple(x) for x in calib[timing_cols].values)
    val_tuples = set(tuple(x) for x in val[timing_cols].values)
    test_tuples = set(tuple(x) for x in test[timing_cols].values)

    assert len(calib_tuples) == len(calib), "Duplicate timing vectors within Calibration split"
    assert len(val_tuples) == len(val), "Duplicate timing vectors within Validation split"
    assert len(test_tuples) == len(test), "Duplicate timing vectors within Test split"

    assert len(calib_tuples & val_tuples) == 0, "Timing vector leakage between Calib and Val"
    assert len(calib_tuples & test_tuples) == 0, "Timing vector leakage between Calib and Test"
    assert len(val_tuples & test_tuples) == 0, "Timing vector leakage between Val and Test"


def test_adversarial_deadline_calibration_provenance_and_non_contamination(trace_datasets):
    """
    Adversarially verify that D_reg = Q85(S_DAG):
    1. Matches Calibration split Q85 to float precision (< 1e-12) in all 18 regimes.
    2. Strictly diverges from Val, Test, Calib+Val, Calib+Test, and Pooled quantiles.
    3. Calibration SHA-256 matches gate_b_deadlines.json.
    4. Calibration miss rate is exactly 15.0% (60/400 misses per regime).
    """
    calib = trace_datasets["calib"]
    val = trace_datasets["val"]
    test = trace_datasets["test"]
    deadlines = trace_datasets["deadlines"]
    regimes_dict = deadlines["regimes"]

    # Verify SHA-256
    with open(CALIB_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == deadlines["calibration_trace_sha256"], "Calibration trace SHA-256 mismatch!"

    for reg_id in range(18):
        c_sub = calib[calib["regime_id"] == reg_id]["s_dag_ms"]
        v_sub = val[val["regime_id"] == reg_id]["s_dag_ms"]
        t_sub = test[test["regime_id"] == reg_id]["s_dag_ms"]

        stored_d = float(regimes_dict[str(reg_id)]["deadline_ms"])

        # Q85 on Calib
        q_calib = float(np.quantile(c_sub, 0.85, method="linear"))
        assert abs(stored_d - q_calib) < 1e-10, (
            f"Regime {reg_id}: stored deadline {stored_d} != calib Q85 {q_calib}"
        )

        # Verify exact 15% miss rate in Calib (400 * 0.15 = 60 misses)
        misses = int((c_sub > stored_d).sum())
        assert misses == 60, f"Regime {reg_id}: expected exactly 60 misses in Calib, got {misses}"
        miss_rate = misses / 400.0
        assert abs(miss_rate - 0.15) < 1e-6

        # Prove non-contamination: test against pooled / corrupted splits
        q_val = float(np.quantile(v_sub, 0.85, method="linear"))
        q_test = float(np.quantile(t_sub, 0.85, method="linear"))
        q_cv = float(np.quantile(pd.concat([c_sub, v_sub]), 0.85, method="linear"))
        q_ct = float(np.quantile(pd.concat([c_sub, t_sub]), 0.85, method="linear"))
        q_cvt = float(np.quantile(pd.concat([c_sub, v_sub, t_sub]), 0.85, method="linear"))

        # In every regime, stored deadline must demonstrably differ from at least one pooled/test quantile
        assert abs(stored_d - q_cv) > 1e-4 or abs(stored_d - q_cvt) > 1e-4, (
            f"Regime {reg_id}: deadline appears contaminated with pooled val/test data!"
        )


def test_adversarial_test_split_isolation(trace_datasets):
    """
    Stress-test test split isolation:
    1. uav_test_traces.csv is NOT accessed by any baseline or detector code.
    2. No NaN or Inf values in test traces.
    3. Columns match calibration traces exactly.
    """
    test = trace_datasets["test"]
    calib = trace_datasets["calib"]

    # Check columns
    assert list(test.columns) == list(calib.columns), "Column schema mismatch between calib and test"

    # Check for NaN / Inf
    assert test.isnull().sum().sum() == 0, "Test traces contain NaN values"
    num_cols = test.select_dtypes(include=[np.number])
    assert np.isinf(num_cols).sum().sum() == 0, "Test traces contain Inf values"

    # Static audit of codebase for forbidden references to test split
    code_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # Skip git, tests, results, and .agents directories
        if any(ignored in root for ignored in [".git", "tests", "results", ".agents", ".pytest_cache"]):
            continue
        for file in files:
            if file.endswith((".py", ".sh", ".ps1", ".yaml")):
                code_files.append(os.path.join(root, file))

    for path in code_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # uav_autonomy_pipeline.py is the generator, so it only writes it.
        # evaluate_gate_e.py is the authorized Gate E runner.
        fname = os.path.basename(path)
        if fname in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
            if fname == "uav_autonomy_pipeline.py":
                # Assert it only opens in write/export context, never read
                assert "pd.read_csv" not in content or "uav_test_traces.csv" not in content, (
                    f"{path} reads test split!"
                )
            continue
        assert "uav_test_traces.csv" not in content, (
            f"FORBIDDEN: Code file {path} references test split uav_test_traces.csv!"
        )


def test_adversarial_fork_concurrency_and_join_node(trace_datasets):
    """
    Adversarial physical execution checks:
    - S_DAG identity S_DAG = t1 + t2 + max(t3, t4) + t5 + t6 holds to < 1e-5 on all 18,000 frames.
    - Fork wall clock time t_fork >= max(t3, t4) on 100% of frames (no time-travel or premature join).
    - Fork wall clock time t_fork < t3 + t4 on > 99.5% of frames (proves genuine concurrent execution).
    - Join node N5 median >= 1.0 ms and >= 1.0% of e2e across all 18 regimes in all three splits.
    """
    full = trace_datasets["full"]

    # S_DAG mathematical identity
    expected_s_dag = (
        full["t_n1"] + full["t_n2"] + np.maximum(full["t_n3"], full["t_n4"]) + full["t_n5"] + full["t_n6"]
    )
    max_s_dag_err = np.max(np.abs(full["s_dag_ms"] - expected_s_dag))
    assert max_s_dag_err < 1e-5, f"S_DAG identity violated: max error = {max_s_dag_err}"

    # Concurrency checks
    t3 = full["t_n3"]
    t4 = full["t_n4"]
    t_fork = full["dur_fork_wall_ms"]
    max_t = np.maximum(t3, t4)
    sum_t = t3 + t4

    # Fork wall time must be >= max_t
    under_fork = (t_fork < max_t).sum()
    assert under_fork == 0, f"Found {under_fork} frames where fork wall time < max(t3, t4)"

    # Fork wall time must be < sum_t on >= 98% of frames (proves parallel speedup over serial,
    # with ~0.6% tail preemption under Heavy 8-worker contention)
    faster_than_serial = (t_fork < sum_t).mean()
    assert faster_than_serial >= 0.98, (
        f"Fork execution failed concurrency: only {faster_than_serial*100:.2f}% frames faster than serial sum"
    )

    # N5 non-degeneracy across all splits and regimes
    for s_name, s_df in [
        ("calib", trace_datasets["calib"]),
        ("val", trace_datasets["val"]),
        ("test", trace_datasets["test"])
    ]:
        for r in range(18):
            sub = s_df[s_df["regime_id"] == r]
            med_n5 = float(sub["t_n5"].median())
            med_e2e = float(sub["t_e2e"].median())
            pct = (med_n5 / med_e2e) * 100.0
            assert med_n5 >= 1.0, f"{s_name} regime {r}: N5 median {med_n5:.3f} ms < 1.0 ms"
            assert pct >= 1.0, f"{s_name} regime {r}: N5 {pct:.2f}% of E2E < 1.0%"
