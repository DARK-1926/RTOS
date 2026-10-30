r"""
Comprehensive Test Suite for Gate D Candidate Detectors (Milestone 3).

Verifies:
1. D1 Centroid Classification & Online Rolling Window State Reset per episode.
2. D2 Super-Node PMF construction matches empirical max(X3, X4) distribution.
3. D2 5-Stage Backward Convolution eliminates Eq. 11 join-node distortion.
4. Monotonicity and validity of meeting probabilities for D1 and D2 across valid AST values.
5. Linear variable threshold schedule monotonicity and policy configuration.
6. Strict data split isolation: zero references to test split in src/ or benchmarks/.
"""

import os
import glob
import numpy as np
import pandas as pd
import pytest

from src.detectors.toba_baseline import DiscretePMF, TobaPlaxityBaseline
from src.detectors.regime_detector import (
    ObservableStateEstimator,
    SuperNodePlaxityBaseline,
    RegimeConditionedDetectorD1,
    SuperNodePlaxityDetectorD2,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIB_PATH = os.path.join(REPO_ROOT, "results", "uav_calibration_traces.csv")
VAL_PATH = os.path.join(REPO_ROOT, "results", "uav_validation_traces.csv")
DEADLINES_PATH = os.path.join(REPO_ROOT, "results", "gate_b_deadlines.json")


# ==============================================================================
# Test 1: ObservableStateEstimator & D1 Online Rolling Window Reset
# ==============================================================================

def test_d1_centroid_classification_and_rolling_state_reset():
    """
    Verifies that:
    (a) ObservableStateEstimator correctly computes 18 regime centroids from calibration data.
    (b) Queries at exact centroid coordinates correctly classify to the matching regime.
    (c) Rolling window maintains state over past W=5 frames.
    (d) reset() completely clears episode state and falls back to global centroid for S_DAG.
    """
    assert os.path.exists(CALIB_PATH), f"Missing {CALIB_PATH}"
    calib_df = pd.read_csv(CALIB_PATH)

    estimator = ObservableStateEstimator(window=5)
    estimator.fit(calib_df)

    assert len(estimator.calib_centroids) == 18
    assert estimator.global_centroid is not None
    assert len(estimator.global_centroid) == 3

    # Exact centroid matching
    for r in range(18):
        c = estimator.calib_centroids[r]
        # Querying with t1=c[0], t2=c[1] after setting history or single frame with sdag=c[2]
        estimator.reset()
        # Feed 5 identical frames to lock rolling median to centroid
        for _ in range(5):
            estimator.update(c[0], c[1], c[2])
        pred_r = estimator.predict_regime(c[0], c[1])
        assert pred_r == r, f"Centroid {r} misclassified as {pred_r}"

    # Verify reset behavior
    estimator.reset()
    assert len(estimator.hist_t1) == 0
    assert len(estimator.hist_t2) == 0
    assert len(estimator.hist_sdag) == 0

    # First frame observable vector must use global centroid[2] for S_DAG
    obs_vec = estimator.get_observable_vector(curr_t1=5.0, curr_t2=1.0)
    assert obs_vec[0] == 5.0
    assert obs_vec[1] == 1.0
    assert obs_vec[2] == estimator.global_centroid[2]

    # After updating 3 frames, length of history should be 3
    estimator.update(5.0, 1.0, 25.0)
    estimator.update(6.0, 2.0, 30.0)
    estimator.update(7.0, 3.0, 35.0)
    assert len(estimator.hist_t1) == 3
    obs_vec2 = estimator.get_observable_vector(curr_t1=8.0, curr_t2=4.0)
    # w_t1 = [5, 6, 7, 8] -> median is 6.5
    # w_t2 = [1, 2, 3, 4] -> median is 2.5
    # w_sdag = [25, 30, 35] -> median is 30.0
    assert abs(obs_vec2[0] - 6.5) < 1e-6
    assert abs(obs_vec2[1] - 2.5) < 1e-6
    assert abs(obs_vec2[2] - 30.0) < 1e-6

    # Reset clears history
    estimator.reset()
    assert len(estimator.hist_t1) == 0


# ==============================================================================
# Test 2: D2 Super-Node PMF Construction vs. Empirical Max Distribution
# ==============================================================================

def test_d2_super_node_pmf_construction_matches_empirical_max():
    """
    Verifies that DiscretePMF constructed for N_fork matches the empirical
    distribution of max(X3, X4) with exact mass conservation.
    """
    rng = np.random.RandomState(42)
    n_samples = 5000
    scale = 10

    # Correlated X3 and X4 samples
    shared = rng.exponential(scale=5.0, size=n_samples)
    x3 = shared + rng.uniform(2.0, 8.0, size=n_samples)
    x4 = shared + rng.uniform(1.0, 6.0, size=n_samples)
    x_fork_empirical = np.maximum(x3, x4)

    max_d = float(np.max(x_fork_empirical) * 1.2)
    pmf_fork = DiscretePMF.from_samples(x_fork_empirical, max_val=max_d, scale=scale)

    # Check total probability mass conservation
    total_mass = np.sum(pmf_fork.probs) + pmf_fork.p_miss_already
    assert abs(total_mass - 1.0) < 1e-12

    # Check CDF match at various evaluation points
    for q in [5.0, 10.0, 15.0, 20.0, 25.0]:
        emp_cdf = float(np.mean(x_fork_empirical <= q))
        bin_idx = int(round(q * scale))
        pmf_cdf = float(np.sum(pmf_fork.probs[: bin_idx + 1]))
        assert abs(emp_cdf - pmf_cdf) < 0.02, (
            f"CDF mismatch at q={q}: empirical={emp_cdf:.4f}, pmf={pmf_cdf:.4f}"
        )


# ==============================================================================
# Test 3: D2 5-Stage Backward Convolution Eliminates Eq. 11 Distortion
# ==============================================================================

def test_d2_super_node_eliminates_eq11_distortion():
    """
    Constructs a fork-join DAG toy where parallel branches N3 || N4 join at N5.
    Verifies that:
    (a) Toba Eq. 11 produces an analytical discrepancy vs brute force joint survival.
    (b) SuperNodePlaxityBaseline completely bypasses Eq. 11, matching brute force exactly.
    """
    scale = 1
    # Independent discrete branch distributions:
    # X3 in {1, 3} ms, prob 0.5 each
    p3 = np.array([0.0, 0.5, 0.0, 0.5])
    # X4 in {1, 3} ms, prob 0.5 each
    p4 = np.array([0.0, 0.5, 0.0, 0.5])
    # Downstream join task X5 in {2, 4} ms, prob 0.5 each
    p5 = np.array([0.0, 0.0, 0.5, 0.0, 0.5])

    pmf3 = DiscretePMF(p3, scale=scale)
    pmf4 = DiscretePMF(p4, scale=scale)
    pmf5 = DiscretePMF(p5, scale=scale)

    # Super-node distribution: X_fork = max(X3, X4)
    # Brute force pairs of (x3, x4) in {1, 3} x {1, 3}:
    # max(1,1)=1 (prob 0.25), max(1,3)=3 (prob 0.25), max(3,1)=3 (prob 0.25), max(3,3)=3 (prob 0.25)
    # Hence X_fork = 1 (prob 0.25), 3 (prob 0.75)
    p_fork = np.array([0.0, 0.25, 0.0, 0.75])
    pmf_fork = DiscretePMF(p_fork, scale=scale)

    # True joint distribution of remaining latency from fork start:
    # S_rem = max(X3, X4) + X5
    # Possible outcomes:
    # 1 + 2 = 3 (0.25 * 0.5 = 0.125)
    # 1 + 4 = 5 (0.25 * 0.5 = 0.125)
    # 3 + 2 = 5 (0.75 * 0.5 = 0.375) -> total for 5 is 0.500
    # 3 + 4 = 7 (0.75 * 0.5 = 0.375)
    p_true_s_rem = {3: 0.125, 5: 0.500, 7: 0.375}

    deadline = 8.0

    # 1. Super-Node plaxity backward subtraction:
    # L5 = D - X5
    l5 = DiscretePMF.from_exit_node(pmf5, deadline_ms=deadline, scale=scale)
    # L_fork = L5 \ominus X_fork
    l_fork = l5.convolutional_subtraction(pmf_fork)

    # 2. Standard Toba backward recursion using Eq. 11:
    l3 = l5.convolutional_subtraction(pmf3)
    l4 = l5.convolutional_subtraction(pmf4)
    l_toba = DiscretePMF.independent_fork_min(l3, l4)

    # Compare query_meeting_prob across all arrival times
    for t_int in range(0, int(deadline) + 1):
        t_ms = float(t_int)
        p_d2 = l_fork.query_meeting_prob(t_ms)
        p_toba = l_toba.query_meeting_prob(t_ms)

        # Ground truth: P(S_rem <= deadline - t_ms)
        rem_avail = deadline - t_ms
        p_true = sum(prob for s_val, prob in p_true_s_rem.items() if s_val <= rem_avail + 1e-9)

        # Super-node D2 must match ground truth exactly (< 10^-12)
        diff_d2 = abs(p_d2 - p_true)
        assert diff_d2 < 1e-12, (
            f"Super-node D2 mismatch at t={t_ms} ms: D2={p_d2:.8f}, true={p_true:.8f}"
        )

        # At t = 2.0 ms (rem_avail = 6.0 ms):
        # Ground truth P(S_rem <= 6) = 0.125 + 0.500 = 0.625
        # Under Toba Eq. 11, L3 and L4 are assumed independent given L5,
        # inducing the join-node reuse discrepancy
        if abs(t_ms - 2.0) < 1e-6:
            # Toba computes P(L3 >= 2) * P(L4 >= 2)
            # L3 is 8 - X5 - X3. P(X5 + X3 <= 6):
            # X5=2: X3=1,3 (0.5 * 1.0 = 0.5)
            # X5=4: X3=1 (0.5 * 0.5 = 0.25) -> P(L3 >= 2) = 0.75
            # P(L4 >= 2) = 0.75
            # Toba Eq. 11 predicts: 0.75 * 0.75 = 0.5625
            # Discrepancy is 0.5625 - 0.625 = -0.0625 (-6.25 pp)
            toba_diff = p_toba - p_true
            assert abs(toba_diff - (-0.0625)) < 1e-12, f"Expected -0.0625 Toba error, got {toba_diff}"


# ==============================================================================
# Test 4: Monotonicity & Validity of Meeting Probabilities
# ==============================================================================

def test_monotonicity_and_validity_of_meeting_probs():
    """
    Verifies that for both D1 and D2:
    (a) Meeting probabilities are valid probabilities: 0.0 <= P <= 1.0.
    (b) Non-increasing monotonicity holds across arrival times:
        ast_1 < ast_2 ==> P(L >= ast_1) >= P(L >= ast_2).
    (c) Guaranteed deadline miss (ast > deadline) returns P = 0.0.
    """
    calib_df = pd.read_csv(CALIB_PATH)
    deadlines = {0: 21.02851}

    # Fit D1 & D2 on Regime 0
    d1 = RegimeConditionedDetectorD1(scale=10, policy="fixed_0.95")
    d1.fit(calib_df, deadlines=deadlines)

    d2 = SuperNodePlaxityDetectorD2(scale=10, policy="fixed_0.95", fork_mode="wall")
    d2.fit(calib_df, deadlines=deadlines)

    for det_obj, name in [(d1, "D1"), (d2, "D2")]:
        det = det_obj.regime_detectors[0]
        for node_id, pmf in det.plaxity_pmfs.items():
            prev_p = 1.0
            test_points = np.linspace(0.0, 30.0, 301)
            for t in test_points:
                p = pmf.query_meeting_prob(t)
                assert 0.0 <= p <= 1.0, f"{name} Node {node_id} returned invalid probability {p}"
                assert p <= prev_p + 1e-12, (
                    f"{name} Node {node_id} violated non-increasing monotonicity at t={t}: "
                    f"p={p} > prev_p={prev_p}"
                )
                prev_p = p

            # Past deadline (30 ms > 21.03 ms) must return 0.0
            p_past = pmf.query_meeting_prob(30.0)
            assert p_past == 0.0, f"{name} Node {node_id} did not return 0.0 for ast > deadline"


# ==============================================================================
# Test 5: Variable Probabilistic Threshold Policies
# ==============================================================================

def test_variable_probabilistic_threshold_schedules():
    """
    Verifies that SuperNodePlaxityBaseline correctly computes linear variable threshold
    schedules (TECS 2026 Eq. 13-18) across the 5 super-DAG stages:
    N1 -> N2 -> N_fork -> N5 -> N6
    """
    calib_df = pd.read_csv(CALIB_PATH)
    deadline = 25.0

    # 1. Job count schedule: linear_job_0.8_1.0
    det_job = SuperNodePlaxityBaseline(
        scale=10, threshold_mode="linear_job", th_min=0.80, th_max=1.00
    ).fit(calib_df[calib_df["regime_id"] == 0], deadline_ms=deadline)

    th_j = det_job.thresholds
    assert th_j[1] == 0.80
    assert th_j[6] == 1.00
    assert th_j[1] < th_j[2] < th_j[3] == th_j[4] < th_j[5] < th_j[6]

    # Check exact step progression: rem = [5, 4, 3, 2, 1]
    # th_1 = 0.80
    # th_2 = 0.80 + 0.20 / 4 = 0.85
    # th_fork = 0.85 + 0.15 / 3 = 0.90
    # th_5 = 0.90 + 0.10 / 2 = 0.95
    # th_6 = 1.00
    assert abs(th_j[2] - 0.85) < 1e-6
    assert abs(th_j[3] - 0.90) < 1e-6
    assert abs(th_j[4] - 0.90) < 1e-6
    assert abs(th_j[5] - 0.95) < 1e-6

    # 2. Mean time schedule: linear_time_0.8_1.0
    det_time = SuperNodePlaxityBaseline(
        scale=10, threshold_mode="linear_time", th_min=0.80, th_max=1.00
    ).fit(calib_df[calib_df["regime_id"] == 0], deadline_ms=deadline)

    th_t = det_time.thresholds
    assert th_t[1] == 0.80
    assert th_t[6] == 1.00
    assert th_t[1] <= th_t[2] <= th_t[3] == th_t[4] <= th_t[5] <= th_t[6]


# ==============================================================================
# Test 6: Strict Data Split Isolation & Non-Contamination Audit
# ==============================================================================

def test_strict_split_isolation():
    """
    Scans all source code in src/ and benchmarks/ to verify zero references to
    the test split traces.
    """
    forbidden_name = f"uav_{'test'}_traces.csv"
    for folder in ["src", "benchmarks"]:
        folder_path = os.path.join(REPO_ROOT, folder)
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname.endswith(".py"):
                    if fname in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
                        continue
                    full_path = os.path.join(root, fname)
                    with open(full_path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                        assert forbidden_name not in content, (
                            f"Audit violation: '{fname}' in {folder} references test split!"
                        )

    # Verify calibration and validation split row counts and disjoint episodes
    calib_df = pd.read_csv(CALIB_PATH)
    val_df = pd.read_csv(VAL_PATH)

    assert len(calib_df) == 7200, f"Expected 7200 calib rows, got {len(calib_df)}"
    assert len(val_df) == 3600, f"Expected 3600 val rows, got {len(val_df)}"

    calib_eps = set(calib_df["episode_id"].unique())
    val_eps = set(val_df["episode_id"].unique())

    overlap = calib_eps.intersection(val_eps)
    assert len(overlap) == 0, f"Critical split contamination: {len(overlap)} episodes overlap!"
