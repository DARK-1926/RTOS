"""
Formal Sanity Checks for Gate C (Baselines & Sanity Checks).

Tests:
1. Sanity Check 1 (Serial Toy Chain): Discrete convolutional subtraction vs. brute-force
   enumeration across all permutations matches exactly (< 10^-12 error).
2. Sanity Check 2 (Fork-Join Toy): Quantifies and verifies the Eq. 11 independence discrepancy
   vs. brute-force joint distribution, and verifies exact match under genuine independence.
3. AST Arrival-Time Checking and Linear Variable Threshold Schedules (ACM TECS 2026 Eq. 13-18).
4. WCET Laxity Equivalence at Threshold theta = 1.0.
5. Boundary Handling & Negative-Slack (Guaranteed Miss) Conservation.
6. Split Isolation & Non-Contamination Verification (zero references to test split).
"""

import os
import glob
import itertools
import pytest
import numpy as np
import pandas as pd

from src.detectors.toba_baseline import DiscretePMF, TobaPlaxityBaseline, TobaDetectorSuite


# ==============================================================================
# Sanity Check 1: Serial Toy Chain Discrete Convolution vs Brute Force
# ==============================================================================

def test_sanity_check_1_serial_toy_chain_forward_and_backward():
    """
    Sanity Check 1: 3-node serial DAG (N1 -> N2 -> N3).
    Verifies that:
    (a) Forward discrete convolution matches Cartesian brute-force sum distribution (< 10^-12).
    (b) Convolutional subtraction backward plaxity matches brute-force survival (< 10^-12).
    (c) All 3! = 6 permutations yield identical distributions (< 10^-12).
    """
    scale = 10
    # Discrete execution PMFs:
    # X1 in {1, 2, 3} ms
    p1 = np.array([0.0, 0.2, 0.5, 0.3])
    # X2 in {1, 2, 3} ms
    p2 = np.array([0.0, 0.3, 0.4, 0.3])
    # X3 in {1, 2, 4} ms
    p3 = np.array([0.0, 0.1, 0.6, 0.0, 0.3])

    pmf1 = DiscretePMF(p1, scale=scale)
    pmf2 = DiscretePMF(p2, scale=scale)
    pmf3 = DiscretePMF(p3, scale=scale)

    # 1. Forward Brute-Force Cartesian Product
    max_len = len(p1) + len(p2) + len(p3)
    p_brute = np.zeros(max_len, dtype=np.float64)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            for k, c in enumerate(p3):
                p_brute[i + j + k] += a * b * c

    # 2. Forward Linear Convolution
    p_conv = np.convolve(np.convolve(p1, p2), p3)
    max_err_forward = np.max(np.abs(p_conv - p_brute[:len(p_conv)]))
    assert max_err_forward < 1e-12, f"Forward convolution failed: error {max_err_forward:.2e}"

    # 3. Permutation Invariance across all 6 orderings
    for perm in itertools.permutations([p1, p2, p3]):
        p_perm = np.convolve(np.convolve(perm[0], perm[1]), perm[2])
        err_perm = np.max(np.abs(p_perm - p_conv))
        assert err_perm < 1e-12, f"Permutation invariance failed: error {err_perm:.2e}"

    # 4. Backward Convolutional Subtraction vs Brute Force across multiple deadlines
    for D_int in [5, 6, 7, 8, 10]:
        D_ms = D_int / float(scale)
        # N3 exit node: L3 = D - X3
        l3 = DiscretePMF.from_exit_node(pmf3, deadline_ms=D_ms, scale=scale)
        # N2: L2 = L3 \ominus X2
        l2 = l3.convolutional_subtraction(pmf2)
        # N1: L1 = L2 \ominus X1
        l1 = l2.convolutional_subtraction(pmf1)

        # For every arrival time AST in [0, D_ms], P(L1 >= AST) must match P(S <= D_int - AST_int)
        for t_int in range(D_int + 1):
            ast_ms = t_int / float(scale)
            p_meet_toba = l1.query_meeting_prob(ast_ms)

            # Brute force probability that S <= D_int - t_int
            rem_d = D_int - t_int
            p_meet_brute = float(np.sum(p_brute[: rem_d + 1])) if rem_d >= 0 else 0.0

            diff = abs(p_meet_toba - p_meet_brute)
            assert diff < 1e-12, (
                f"Backward plaxity mismatch at D={D_int}, t={t_int}: "
                f"toba={p_meet_toba:.8f}, brute={p_meet_brute:.8f}, diff={diff:.2e}"
            )


# ==============================================================================
# Sanity Check 2: Fork-Join Toy with Real Join vs. Eq. (11) Independence Discrepancy
# ==============================================================================

def test_sanity_check_2_fork_join_independent_exact_match():
    """
    Sanity Check 2 (Part A): 4-node DAG (N1 -> Fork {N2 || N3} -> N4) under TRUE independence.
    Verifies that when branches are statistically independent, discrete convolution
    and fork survival multiplication match 4D Cartesian brute-force within < 10^-12.
    """
    scale = 10
    # X1 in {1, 2} ms
    p1 = np.array([0.0, 0.4, 0.6])
    # X2 in {2, 4} ms
    p2 = np.array([0.0, 0.0, 0.5, 0.0, 0.5])
    # X3 in {2, 4} ms
    p3 = np.array([0.0, 0.0, 0.5, 0.0, 0.5])
    # X4 (Join) in {3, 5} ms
    p4 = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.5])

    # 4D Cartesian Brute Force: S = X1 + max(X2, X3) + X4
    p_brute = np.zeros(25, dtype=np.float64)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            for k, c in enumerate(p3):
                for m, d in enumerate(p4):
                    s = i + max(j, k) + m
                    p_brute[s] += a * b * c * d

    # Forward Independent Fork:
    # F_fork(t) = F2(t) * F3(t)
    f2 = np.cumsum(p2)
    f3 = np.cumsum(p3)
    f_fork = f2 * f3
    p_fork = np.zeros(len(f_fork))
    p_fork[0] = f_fork[0]
    p_fork[1:] = np.diff(f_fork)

    p_s = np.convolve(np.convolve(p1, p_fork), p4)
    max_err = np.max(np.abs(p_s - p_brute[:len(p_s)]))
    assert max_err < 1e-12, f"Independent fork-join forward mismatch: error {max_err:.2e}"


def test_sanity_check_2_fork_join_eq11_discrepancy_analytical_and_numerical():
    """
    Sanity Check 2 (Part B): Formally proves and numerically verifies the exact
    Eq. 11 independence discrepancy caused by downstream join node coupling.
    Setup from ACM TECS 2026 specification:
    X2 in {2, 6} (prob 0.5, 0.5), X3 in {2, 6} (prob 0.5, 0.5), X4 in {1, 5} (prob 0.5, 0.5), D=12.
    Predicted discrepancies:
    lambda = 1: +0.0625 (+6.25 pp)
    lambda = 5:  0.0000 ( 0.0 pp)
    lambda = 9: -0.0625 (-6.25 pp)
    """
    scale = 1
    D = 12

    p4_arr = np.zeros(6); p4_arr[1] = 0.5; p4_arr[5] = 0.5
    p2_arr = np.zeros(7); p2_arr[2] = 0.5; p2_arr[6] = 0.5
    p3_arr = np.zeros(7); p3_arr[2] = 0.5; p3_arr[6] = 0.5

    pmf4 = DiscretePMF(p4_arr, scale=scale)
    pmf2 = DiscretePMF(p2_arr, scale=scale)
    pmf3 = DiscretePMF(p3_arr, scale=scale)

    # Backward Plaxity Analysis using Eq. 9, 10, 11
    # Exit N4:
    l4 = DiscretePMF.from_exit_node(pmf4, deadline_ms=D, scale=scale)
    # Branches N2, N3:
    l2 = l4.convolutional_subtraction(pmf2)
    l3 = l4.convolutional_subtraction(pmf3)
    # Toba Eq. 11 Merge:
    l1_toba = DiscretePMF.independent_fork_min(l2, l3)

    # Exact PMF checks
    assert abs(l1_toba.probs[1] - 0.4375) < 1e-12, f"Expected 0.4375 at lam=1, got {l1_toba.probs[1]}"
    assert abs(l1_toba.probs[5] - 0.5000) < 1e-12, f"Expected 0.5000 at lam=5, got {l1_toba.probs[5]}"
    assert abs(l1_toba.probs[9] - 0.0625) < 1e-12, f"Expected 0.0625 at lam=9, got {l1_toba.probs[9]}"

    # True Joint Plaxity PMF:
    # S = max(X2, X3) + X4
    # max(X2, X3) in {2: 0.25, 6: 0.75}
    # S in {3: 0.125, 7: 0.500, 11: 0.375} -> L = 12 - S in {9: 0.125, 5: 0.500, 1: 0.375}
    true_probs = {1: 0.3750, 5: 0.5000, 9: 0.1250}

    # Numerical Discrepancy Verification
    disc_1 = l1_toba.probs[1] - true_probs[1]
    disc_5 = l1_toba.probs[5] - true_probs[5]
    disc_9 = l1_toba.probs[9] - true_probs[9]

    assert abs(disc_1 - (+0.0625)) < 1e-12, f"Discrepancy at lam=1 failed: {disc_1}"
    assert abs(disc_5 - (0.0000)) < 1e-12, f"Discrepancy at lam=5 failed: {disc_5}"
    assert abs(disc_9 - (-0.0625)) < 1e-12, f"Discrepancy at lam=9 failed: {disc_9}"


def test_sanity_check_2_fork_join_contention_correlation_discrepancy():
    """
    Sanity Check 2 (Part C): Evaluates 4-node DAG under positive branch correlation
    (mimicking shared memory-bus contention). Asserts divergence between true joint
    distribution and independent Eq. 11 prediction exceeds 4.0 pp.
    """
    p_J = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.5])  # X_join in {3, 5}
    # Independent fork branches: X3 in {2, 4}, X4 in {2, 4}
    c3 = np.cumsum(np.array([0.0, 0.0, 0.5, 0.0, 0.5]))
    c4 = np.cumsum(np.array([0.0, 0.0, 0.5, 0.0, 0.5]))
    p_fork_indep = np.diff(np.concatenate(([0.0], c3 * c4)))
    p_tot_indep = np.convolve(p_fork_indep, p_J)

    # Positively correlated branches sharing the identical marginals
    joint_dep = {(2, 2): 0.45, (2, 4): 0.05, (4, 2): 0.05, (4, 4): 0.45}
    p_tot_dep = np.zeros(len(p_tot_indep))
    for (a, b), pab in joint_dep.items():
        for j, pj in enumerate(p_J):
            if pj > 0:
                p_tot_dep[max(a, b) + j] += pab * pj

    # Discrepancy at tail threshold D = 8 (index 9)
    tail_indep = np.sum(p_tot_indep[9:])
    tail_dep = np.sum(p_tot_dep[9:])
    discrepancy = abs(tail_indep - tail_dep)

    assert discrepancy >= 0.04, f"Tail discrepancy under correlation was {discrepancy:.4f} (< 0.04)"


# ==============================================================================
# AST Arrival Checks and Linear Variable Threshold Schedules
# ==============================================================================

def test_variable_threshold_schedules_monotonicity_and_step_size():
    """
    Verifies ACM TECS 2026 Eq. 13-18 on the 6-node UAV autonomy DAG:
    N1 -> N2 -> Fork {N3 || N4} -> N5 -> N6
    For uniform job count (w_v = 1):
    rem = [5, 4, 3, 3, 2, 1]
    Ramp 0.8 -> 1.0: step size exactly +0.05
    Ramp 0.9 -> 1.0: step size exactly +0.025
    Ramp 0.7 -> 1.0: step size exactly +0.075
    """
    # Create synthetic calibration traces
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "regime_id": np.zeros(n, dtype=int),
        "dur_node1_ms": np.random.uniform(1.0, 2.0, n),
        "dur_node2_ms": np.random.uniform(0.1, 0.3, n),
        "dur_node3_ms": np.random.uniform(5.0, 7.0, n),
        "dur_node4_ms": np.random.uniform(5.0, 7.0, n),
        "dur_node5_ms": np.random.uniform(10.0, 15.0, n),
        "dur_node6_ms": np.random.uniform(0.1, 0.2, n),
        "s_dag_ms": np.random.uniform(18.0, 26.0, n),
        "miss": np.zeros(n, dtype=int),
    })

    # Test Ramp 0.8 -> 1.0
    det8 = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.80, th_max=1.00)
    det8.fit(df, deadline_ms=25.0)
    th8 = det8.thresholds
    assert abs(th8[1] - 0.800) < 1e-12
    assert abs(th8[2] - 0.850) < 1e-12
    assert abs(th8[3] - 0.900) < 1e-12
    assert abs(th8[4] - 0.900) < 1e-12
    assert abs(th8[5] - 0.950) < 1e-12
    assert abs(th8[6] - 1.000) < 1e-12
    assert th8[1] < th8[2] < th8[3] == th8[4] < th8[5] < th8[6]

    # Test Ramp 0.9 -> 1.0
    det9 = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.90, th_max=1.00)
    det9.fit(df, deadline_ms=25.0)
    th9 = det9.thresholds
    assert abs(th9[1] - 0.900) < 1e-12
    assert abs(th9[2] - 0.925) < 1e-12
    assert abs(th9[3] - 0.950) < 1e-12
    assert abs(th9[4] - 0.950) < 1e-12
    assert abs(th9[5] - 0.975) < 1e-12
    assert abs(th9[6] - 1.000) < 1e-12

    # Test Ramp 0.7 -> 1.0
    det7 = TobaPlaxityBaseline(scale=10, threshold_mode="linear_job", th_min=0.70, th_max=1.00)
    det7.fit(df, deadline_ms=25.0)
    th7 = det7.thresholds
    assert abs(th7[1] - 0.700) < 1e-12
    assert abs(th7[2] - 0.775) < 1e-12
    assert abs(th7[3] - 0.850) < 1e-12
    assert abs(th7[4] - 0.850) < 1e-12
    assert abs(th7[5] - 0.925) < 1e-12
    assert abs(th7[6] - 1.000) < 1e-12


def test_wcet_laxity_equivalence_at_theta_1():
    """
    Verifies that when threshold theta = 1.0, early detection triggers if and only
    if the actual start time AST exceeds the deterministic WCET laxity cutoff:
    AST_v > D - sum_{downstream} WCET.
    """
    scale = 10
    D = 30.0

    # Deterministic task durations:
    # N1: 2.0 ms, N2: 1.0 ms, N3: 5.0 ms, N4: 6.0 ms, N5: 10.0 ms, N6: 1.0 ms
    # S_DAG = 2 + 1 + max(5, 6) + 10 + 1 = 20.0 ms.
    row_nominal = {
        "dur_node1_ms": 2.0, "dur_node2_ms": 1.0, "dur_node3_ms": 5.0,
        "dur_node4_ms": 6.0, "dur_node5_ms": 10.0, "dur_node6_ms": 1.0,
    }
    df = pd.DataFrame([row_nominal] * 50)
    det = TobaPlaxityBaseline(scale=scale, threshold_mode="fixed", fixed_threshold=1.00)
    det.fit(df, deadline_ms=D)

    # Nominal frame: all ASTs well within laxity -> NO alarm
    res_nom = det.evaluate_frame(row_nominal)
    assert not res_nom["alarm"], "Nominal frame falsely triggered alarm at theta=1.0"

    # Overrun at Node 1: t1 = 15.0 ms -> AST(N2) = 15.0 ms.
    # Downstream WCET from N2 is 1 + 6 + 10 + 1 = 18.0 ms.
    # Total finish = 15 + 18 = 33 > 30 -> MUST alarm at Node 2!
    row_delayed = dict(row_nominal)
    row_delayed["dur_node1_ms"] = 15.0
    res_delayed = det.evaluate_frame(row_delayed)
    assert res_delayed["alarm"], "Delayed frame failed to alarm at theta=1.0"
    assert res_delayed["earliest_checkpoint"] == 2, f"Alarmed at wrong node: {res_delayed['earliest_checkpoint']}"


def test_boundary_handling_and_negative_slack_conservation():
    """
    Verifies that guaranteed misses (LST < 0) are strictly conserved in probability
    and reported at AST = 0 rather than being corrupted to P(meet) = 1.0.
    """
    scale = 10
    D = 5.0

    # Node with 50% chance of exceeding deadline: X in {3.0, 8.0} ms
    p = np.zeros(int(round(9.0 * scale)))
    p[int(round(3.0 * scale))] = 0.5
    p[int(round(8.0 * scale))] = 0.5
    pmf = DiscretePMF(p, scale=scale)

    l_exit = DiscretePMF.from_exit_node(pmf, deadline_ms=D, scale=scale)

    # Total mass must equal 1.0
    total_mass = l_exit.p_miss_already + np.sum(l_exit.probs)
    assert abs(total_mass - 1.0) < 1e-12, f"Probability mass not conserved: {total_mass}"

    # P(miss_already) must equal 0.5
    assert abs(l_exit.p_miss_already - 0.5) < 1e-12, f"p_miss_already was {l_exit.p_miss_already}"

    # At AST = 0, P(meet) must equal 0.5 (NOT 1.0!)
    p_meet_0 = l_exit.query_meeting_prob(0.0)
    assert abs(p_meet_0 - 0.5) < 1e-12, f"P(meet | AST=0) was {p_meet_0}, expected 0.5"


# ==============================================================================
# Static & Runtime Split Isolation Verification
# ==============================================================================

def test_split_isolation_no_test_split_references():
    """
    Static code audit across all files in src/ and benchmarks/.
    Strictly forbids any references, open calls, or strings of uav_test_traces.csv!
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scan_patterns = [
        os.path.join(repo_root, "src", "**", "*.py"),
        os.path.join(repo_root, "benchmarks", "**", "*.py"),
    ]

    checked_files = []
    for pattern in scan_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            # Skip uav_autonomy_pipeline.py as it generates the dataset, and evaluate_gate_e.py as it evaluates test split
            if os.path.basename(file_path) in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
                continue
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Forbidden string assertion
            assert "uav_test_traces.csv" not in content, (
                f"STRICT ISOLATION VIOLATION: {file_path} contains forbidden test split reference!"
            )
            checked_files.append(file_path)

    assert len(checked_files) > 0, "No files scanned for split isolation"


def test_toba_suite_execution_on_authentic_calib_and_val_traces():
    """
    Verifies that TobaDetectorSuite runs cleanly on authentic calibration and validation
    traces without errors, warnings, or NaN predictions.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    calib_path = os.path.join(repo_root, "results", "uav_calibration_traces.csv")
    val_path = os.path.join(repo_root, "results", "uav_validation_traces.csv")

    assert os.path.exists(calib_path), f"Missing {calib_path}"
    assert os.path.exists(val_path), f"Missing {val_path}"

    calib_df = pd.read_csv(calib_path)
    val_df = pd.read_csv(val_path)

    # Subsample 1 regime (Regime 0) for fast smoke test
    calib_r0 = calib_df[calib_df["regime_id"] == 0]
    val_r0 = val_df[val_df["regime_id"] == 0]
    d_r0 = float(calib_r0["deadline_ms"].iloc[0])

    suite = TobaDetectorSuite(scale=10, policy="linear_job_0.8_1.0")
    suite.fit(calib_r0, deadlines={0: d_r0})

    eval_results = suite.evaluate_variant(val_r0, variant="per_regime")
    assert len(eval_results) == len(val_r0)
    for r in eval_results:
        assert isinstance(r["alarm"], bool)
        assert 0.0 <= r["min_p_meet"] <= 1.0
        assert 0.0 <= r["miss_risk"] <= 1.0
        assert r["lead_time_ms"] >= 0.0
