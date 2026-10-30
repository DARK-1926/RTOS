"""
Adversarial Challenger Test Suite for Milestone 1 (Gate B).
Validates:
1. No synthetic sleep, mock delays, or artificial multipliers in compute nodes.
2. Contention workers genuinely saturate memory and cause physical slowdown.
3. N5 join node is non-degenerate across all 18 regimes and all splits (median >= 1.0 ms, >= 1% of E2E).
4. Fork wall-clock duration reflects authentic concurrent execution and bounds thread latencies.
5. Critical path S_DAG formula holds with exact mathematical fidelity.
6. Pre-registered deadlines are strictly regime-specific (no pooling) with exact 15% calibration miss rates.
7. Empirical distributions exhibit authentic physical execution characteristics.
"""

import os
import ast
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from benchmarks.uav_autonomy_pipeline import UAVAutonomyPipeline, ContentionManager

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
TEST_PATH = os.path.join(RESULTS_DIR, "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")


def test_no_synthetic_sleep_or_delay_in_nodes():
    """Adversarial check: ensure no time.sleep or artificial delay loops inside compute nodes."""
    pipeline_file = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "uav_autonomy_pipeline.py")
    with open(pipeline_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=pipeline_file)

    node_methods = [
        "node1_sensor_ingest",
        "node2_state_estimation",
        "node3_obstacle_detection",
        "node4_local_costmap",
        "node5_motion_planner",
        "node6_command_output",
        "run_frame"
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in node_methods:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # Check for sleep calls
                    if isinstance(child.func, ast.Attribute) and child.func.attr == "sleep":
                        pytest.fail(f"FORBIDDEN: sleep() found inside compute node {node.name}")
                    if isinstance(child.func, ast.Name) and "sleep" in child.func.id.lower():
                        pytest.fail(f"FORBIDDEN: sleep-like call found inside compute node {node.name}")


def test_contention_workers_physically_active():
    """Empirical check: ContentionManager must cause measurable physical execution slowdown."""
    pipeline = UAVAutonomyPipeline(grid_size=40)
    # Warmup
    for _ in range(3):
        pipeline.run_frame(wind_level=1.0, clutter_level=1.0)

    # Measure idle
    idle_times = []
    with ContentionManager(level="idle") as cm:
        for _ in range(15):
            res = pipeline.run_frame(wind_level=1.0, clutter_level=1.0)
            idle_times.append(res["s_dag_ms"])
    med_idle = np.median(idle_times)

    # Measure moderate
    mod_times = []
    with ContentionManager(level="moderate") as cm:
        for _ in range(15):
            res = pipeline.run_frame(wind_level=1.0, clutter_level=1.0)
            mod_times.append(res["s_dag_ms"])
    med_mod = np.median(mod_times)

    # Measure heavy
    heavy_times = []
    with ContentionManager(level="heavy") as cm:
        for _ in range(15):
            res = pipeline.run_frame(wind_level=1.0, clutter_level=1.0)
            heavy_times.append(res["s_dag_ms"])
    med_heavy = np.median(heavy_times)

    # Contention workers must cause measurable, monotonic physical slowdown
    assert med_mod > med_idle, f"Moderate contention did not increase median latency: {med_mod:.2f} ms vs {med_idle:.2f} ms"
    assert med_heavy > med_idle * 1.20, f"Heavy contention did not cause significant slowdown: {med_heavy:.2f} ms vs {med_idle:.2f} ms ({med_heavy/med_idle:.2f}x)"


@pytest.mark.parametrize("split_name, path", [
    ("calib", CALIB_PATH),
    ("val", VAL_PATH),
    ("test", TEST_PATH)
])
def test_join_node_non_degeneracy_all_splits(split_name, path):
    """Verify join node N5 duration >= 1.0 ms and >= 1.0% of E2E across all 18 regimes in all splits."""
    df = pd.read_csv(path)
    for reg_id in range(18):
        sub = df[df["regime_id"] == reg_id]
        assert len(sub) > 0, f"Regime {reg_id} missing from {split_name}"
        
        med_n5 = float(sub["dur_node5_ms"].median())
        med_e2e = float(sub["end_to_end_ms"].median())
        pct = (med_n5 / med_e2e) * 100.0
        
        assert med_n5 >= 1.0, f"Split {split_name}, Regime {reg_id}: N5 median {med_n5:.2f} ms < 1.0 ms"
        assert pct >= 1.0, f"Split {split_name}, Regime {reg_id}: N5 pct {pct:.2f}% < 1.0%"
        assert sub["dur_node5_ms"].min() > 0.5, f"Split {split_name}, Regime {reg_id}: N5 min below 0.5 ms"


@pytest.mark.parametrize("split_name, path", [
    ("calib", CALIB_PATH),
    ("val", VAL_PATH),
    ("test", TEST_PATH)
])
def test_fork_wall_clock_and_concurrency(split_name, path):
    """
    Verify fork wall-clock duration bounds thread durations and demonstrates concurrency:
    1. dur_fork_wall_ms >= max(dur_node3_ms, dur_node4_ms) (>= 99.9% of frames)
    2. Parallel speedup (dur_node3_ms + dur_node4_ms) / dur_fork_wall_ms has median > 1.5x
    """
    df = pd.read_csv(path)
    t_fork = df["dur_fork_wall_ms"]
    t3 = df["dur_node3_ms"]
    t4 = df["dur_node4_ms"]
    max_t = np.maximum(t3, t4)
    sum_t = t3 + t4

    # Verification 1: Wall-clock time >= max branch time
    valid_fraction = (t_fork >= max_t - 0.05).mean()
    assert valid_fraction >= 0.999, f"Split {split_name}: Fork wall time under-tracks max branch: {valid_fraction*100:.2f}%"

    # Verification 2: Parallel speedup
    speedup = sum_t / t_fork
    assert speedup.median() > 1.5, f"Split {split_name}: Parallel concurrency speedup median {speedup.median():.2f}x <= 1.5x"


@pytest.mark.parametrize("split_name, path", [
    ("calib", CALIB_PATH),
    ("val", VAL_PATH),
    ("test", TEST_PATH)
])
def test_critical_path_mathematical_identity(split_name, path):
    """Verify S_DAG = t1 + t2 + max(t3, t4) + t5 + t6 exact mathematical identity."""
    df = pd.read_csv(path)
    t1 = df["dur_node1_ms"]
    t2 = df["dur_node2_ms"]
    t3 = df["dur_node3_ms"]
    t4 = df["dur_node4_ms"]
    t5 = df["dur_node5_ms"]
    t6 = df["dur_node6_ms"]
    expected_s_dag = t1 + t2 + np.maximum(t3, t4) + t5 + t6

    np.testing.assert_allclose(df["s_dag_ms"].values, expected_s_dag.values, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(df["t_s_dag"].values, expected_s_dag.values, rtol=1e-5, atol=1e-5)


def test_pre_registered_deadlines_uniqueness_and_calibration():
    """Verify pre-registered deadline rules: 18 unique deadlines, 15% miss rate, SHA-256 integrity."""
    with open(DEADLINES_PATH, "r") as f:
        deadlines_data = json.load(f)

    calib_df = pd.read_csv(CALIB_PATH)
    with open(CALIB_PATH, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()

    assert deadlines_data.get("calibration_trace_sha256") == actual_sha, "Calibration trace hash mismatch"

    regimes_dict = deadlines_data["regimes"]
    assert len(regimes_dict) == 18, f"Expected 18 regimes, found {len(regimes_dict)}"

    deadlines = []
    for reg_id in range(18):
        reg_info = regimes_dict[str(reg_id)]
        d_val = float(reg_info["deadline_ms"])
        deadlines.append(d_val)
        sub = calib_df[calib_df["regime_id"] == reg_id]
        emp_q85 = float(np.quantile(sub["s_dag_ms"], 0.85, method="linear"))
        assert abs(d_val - emp_q85) < 1e-4, f"Regime {reg_id} deadline mismatch: {d_val} vs {emp_q85}"
        miss_count = (sub["s_dag_ms"] > d_val).sum()
        assert miss_count == 60, f"Regime {reg_id}: Expected 60 misses (15%), got {miss_count}"

    # Confirm deadlines are not pooled
    assert len(set(deadlines)) == 18, f"Pooled deadlines detected: only {len(set(deadlines))} unique values"


def test_empirical_latency_authenticity():
    """Verify that execution times exhibit authentic physical hardware distributions (positive skewness)."""
    calib_df = pd.read_csv(CALIB_PATH)
    for col in ["dur_node1_ms", "dur_node3_ms", "dur_node5_ms", "s_dag_ms"]:
        sk = skew(calib_df[col].values)
        assert sk > 0.5, f"Column {col} lacks positive skewness ({sk:.2f}); suspicious for synthetic/uniform data"
