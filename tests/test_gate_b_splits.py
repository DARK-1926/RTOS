"""
Regression test suite for Milestone 1 (Gate B: Environment Regimes & Fresh Trace Collection).
Verifies:
1. File existence and schema integrity.
2. Strict episode-level independence (zero overlap between Calib, Val, and Test).
3. Non-degeneracy of N5 join node (median >= 1.0 ms and >= 1.0% of e2e across all 18 regimes).
4. Pre-registered deadline rule: exact 18 deadlines in JSON matching Q85 with ~15% miss rate on Calibration.
5. Test split containment and isolation.
6. Fork wall-clock timing and S_DAG mathematical soundness.
"""

import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
TEST_PATH = os.path.join(RESULTS_DIR, "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")

def test_files_exist():
    """Verify that all trace splits and deadline JSON files exist and are non-empty."""
    assert os.path.exists(CALIB_PATH), f"Missing {CALIB_PATH}"
    assert os.path.exists(VAL_PATH), f"Missing {VAL_PATH}"
    assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
    assert os.path.exists(DEADLINES_PATH), f"Missing {DEADLINES_PATH}"
    
    assert os.path.getsize(CALIB_PATH) > 0
    assert os.path.getsize(VAL_PATH) > 0
    assert os.path.getsize(TEST_PATH) > 0
    assert os.path.getsize(DEADLINES_PATH) > 0

def test_episode_independence():
    """
    Verify strict episode-level independence:
    No episode_id appears in more than one split (Calib, Val, Test).
    """
    calib_df = pd.read_csv(CALIB_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    calib_eps = set(calib_df["episode_id"].unique())
    val_eps = set(val_df["episode_id"].unique())
    test_eps = set(test_df["episode_id"].unique())
    
    # Assert zero overlap
    overlap_cv = calib_eps & val_eps
    overlap_ct = calib_eps & test_eps
    overlap_vt = val_eps & test_eps
    
    assert len(overlap_cv) == 0, f"Leakage between Calibration and Validation: {overlap_cv}"
    assert len(overlap_ct) == 0, f"Leakage between Calibration and Test: {overlap_ct}"
    assert len(overlap_vt) == 0, f"Leakage between Validation and Test: {overlap_vt}"
    
    # Assert each episode has only one split tag
    for name, df, expected_split in [("calib", calib_df, "calib"), ("val", val_df, "val"), ("test", test_df, "test")]:
        splits = df["split"].unique()
        assert len(splits) == 1 and splits[0] == expected_split, f"Inconsistent split tags in {name}: {splits}"

def test_n5_join_node_non_degeneracy():
    """
    Verify non-degeneracy of N5 join node:
    Median duration must be >= 1.0 ms and >= 1.0% of end-to-end latency across all 18 regimes.
    """
    calib_df = pd.read_csv(CALIB_PATH)
    for reg_id in range(18):
        sub = calib_df[calib_df["regime_id"] == reg_id]
        assert len(sub) > 0, f"Regime {reg_id} has no calibration frames"
        
        n5_col = "t_n5" if "t_n5" in sub.columns else "dur_node5_ms"
        e2e_col = "t_e2e" if "t_e2e" in sub.columns else "end_to_end_ms"
        
        med_n5 = float(sub[n5_col].median())
        med_e2e = float(sub[e2e_col].median())
        pct_e2e = (med_n5 / med_e2e) * 100.0
        
        assert med_n5 >= 1.0, f"Regime {reg_id}: N5 median {med_n5:.3f} ms is below 1.0 ms threshold"
        assert pct_e2e >= 1.0, f"Regime {reg_id}: N5 pct {pct_e2e:.1f}% is below 1.0% threshold"

def test_exact_18_deadlines_and_calibration_miss_rate():
    """
    Verify pre-registered deadline rule:
    Exact 18 deadlines in JSON matching Q85 of Calibration S_DAG per regime,
    yielding approximately 15% miss rate on Calibration data.
    """
    calib_df = pd.read_csv(CALIB_PATH)
    with open(DEADLINES_PATH, "r") as f:
        deadlines_data = json.load(f)
        
    assert "regimes" in deadlines_data
    regimes_dict = deadlines_data["regimes"]
    assert len(regimes_dict) == 18, f"Expected 18 regimes in JSON, found {len(regimes_dict)}"
    
    # Check SHA-256 hash if present
    if "calibration_trace_sha256" in deadlines_data:
        with open(CALIB_PATH, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        assert deadlines_data["calibration_trace_sha256"] == actual_sha, "Calibration trace hash mismatch in deadlines JSON"
        
    for reg_id in range(18):
        reg_key = str(reg_id)
        assert reg_key in regimes_dict, f"Regime {reg_id} missing from deadlines JSON"
        reg_info = regimes_dict[reg_key]
        
        d_val = float(reg_info["deadline_ms"])
        assert np.isfinite(d_val) and d_val > 0.0, f"Invalid deadline {d_val} for regime {reg_id}"
        
        sub = calib_df[calib_df["regime_id"] == reg_id]
        emp_q85 = float(np.quantile(sub["s_dag_ms"], 0.85, method="linear"))
        assert abs(emp_q85 - d_val) < 1e-4, f"Deadline mismatch for regime {reg_id}: {d_val} vs {emp_q85}"
        
        # Empirical miss rate should be ~15% (allow +/- 3% tolerance for finite sample discretization)
        emp_miss_rate = float((sub["s_dag_ms"] > d_val).mean())
        assert 0.12 <= emp_miss_rate <= 0.18, f"Regime {reg_id} miss rate out of range: {emp_miss_rate*100:.2f}%"

def test_test_split_containment_and_isolation():
    """
    Verify test split containment:
    Test traces exist, but NO baseline or detector script in src/detectors or benchmarks/
    imports or opens uav_test_traces.csv.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    detector_scripts = glob.glob(os.path.join(repo_root, "src", "detectors", "**", "*.py"), recursive=True)
    bench_scripts = glob.glob(os.path.join(repo_root, "benchmarks", "**", "*.py"), recursive=True)
    
    for script_path in detector_scripts + bench_scripts:
        fname = os.path.basename(script_path)
        if fname in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
            continue
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        assert "uav_test_traces.csv" not in code, f"FORBIDDEN: {script_path} accesses test split uav_test_traces.csv!"

def test_fork_timing_and_s_dag_soundness():
    """
    Verify fork wall-clock duration and S_DAG math soundness:
    S_DAG == t1 + t2 + max(t3, t4) + t5 + t6
    t_fork >= max(t3, t4) on >= 90% of frames (allowing scheduling noise)
    """
    calib_df = pd.read_csv(CALIB_PATH)
    t1 = calib_df["t_n1"]
    t2 = calib_df["t_n2"]
    t3 = calib_df["t_n3"]
    t4 = calib_df["t_n4"]
    t5 = calib_df["t_n5"]
    t6 = calib_df["t_n6"]
    
    expected_s_dag = t1 + t2 + np.maximum(t3, t4) + t5 + t6
    np.testing.assert_allclose(calib_df["s_dag_ms"].values, expected_s_dag.values, rtol=1e-5, atol=1e-5)
    
    t_fork = calib_df["t_fork"]
    max_fork_thread = np.maximum(t3, t4)
    valid_fraction = (t_fork >= max_fork_thread * 0.95).mean()
    assert valid_fraction >= 0.90, f"Fork wall time under-tracks thread times: {valid_fraction*100:.1f}%"
