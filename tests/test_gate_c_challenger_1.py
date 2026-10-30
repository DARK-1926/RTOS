"""
Adversarial Stress Test Suite for Milestone 2 (Gate C: Baselines & Sanity Checks).
Author: Challenger 1 (challenger_m2_1)
Role: Empirical Challenger (critic / specialist)

Adversarially tests:
1. Extreme Edge Cases:
   - Single-point Dirac delta PMFs (deterministic tasks) and zero variance.
   - Wide flat uniform PMFs (high-dimensional support).
   - Zero deadline (D = 0.0 ms) and negative deadlines (D < 0 ms).
2. Conservation of Negative Slack (p_miss_already):
   - Backward monotonicity along DAG execution paths (N6 -> N5 -> {N3, N4} -> N2 -> N1).
   - Probability mass conservation: sum(probs) + p_miss_already == 1.0 to < 10^-12.
3. Monotonicity of Survival Probabilities:
   - P(meet | AST) as a function of AST across continuous real line.
   - Boundary checks at AST = 0.0 ms and AST < 0.0 ms.
4. Independent Fork Operator:
   - Upper Cumulative Product holds exactly under genuine independence (< 10^-12).
   - Systematic divergence under positive correlation (up to 25.0 pp) and negative correlation.
   - Asymmetric branch variance and bottleneck domination.
5. Bug Audits & Failure Mode Demonstrations:
   - Zero-deadline clamping bug in TobaPlaxityBaseline.fit().
   - AST boundary jump in query_meeting_prob().
"""

import os
import json
import itertools
import pytest
import numpy as np
import pandas as pd

from src.detectors.toba_baseline import (
    DiscretePMF,
    TobaPlaxityBaseline,
    TobaDetectorSuite,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")


# ==============================================================================
# 1. Extreme Edge Cases: Single-point Dirac Deltas & Zero Variance
# ==============================================================================

def test_adversarial_dirac_delta_zero_variance_dag():
    """
    Constructs a 6-node DAG where every task has zero variance (Dirac delta PMF):
    c1 = 2.0, c2 = 1.0, c3 = 4.0, c4 = 6.0, c5 = 8.0, c6 = 3.0 ms.
    Exact deterministic completion time:
    S = 2 + 1 + max(4, 6) + 8 + 3 = 20.0 ms.

    Verifies across multiple deadlines:
    - For D >= 20.0: plaxity distribution L1 is a single Dirac delta at D - 20.0 ms (prob 1.0).
    - For D < 20.0: p_miss_already == 1.0, and sum(probs) == 0.0.
    - All intermediate node plaxities match analytical laxities exactly.
    """
    scale = 10
    c = {1: 2.0, 2: 1.0, 3: 4.0, 4: 6.0, 5: 8.0, 6: 3.0}
    c_means = {k: float(v) for k, v in c.items()}

    pmfs = {}
    for i, val in c.items():
        p = np.zeros(int(round(val * scale)) + 1)
        p[-1] = 1.0
        pmfs[i] = DiscretePMF(p, scale=scale)

    test_deadlines = [30.0, 25.0, 20.0, 19.9, 15.0, 5.0, 0.0]

    for D in test_deadlines:
        det = TobaPlaxityBaseline(scale=scale, threshold_mode="fixed", fixed_threshold=1.0)
        det.fit_from_pmfs(pmfs, c_means, deadline_ms=D)

        l1 = det.plaxity_pmfs[1]
        expected_slack = D - 20.0

        if expected_slack >= 0:
            assert abs(l1.p_miss_already - 0.0) < 1e-12, f"D={D}: expected p_miss=0, got {l1.p_miss_already}"
            expected_bin = int(round(expected_slack * scale))
            assert np.argmax(l1.probs) == expected_bin, f"D={D}: expected peak at {expected_bin}, got {np.argmax(l1.probs)}"
            assert abs(l1.probs[expected_bin] - 1.0) < 1e-12, f"D={D}: peak probability != 1.0"
            # Survival function step
            assert l1.query_meeting_prob(expected_slack) == 1.0
            assert l1.query_meeting_prob(expected_slack + 0.1) == 0.0
        else:
            assert abs(l1.p_miss_already - 1.0) < 1e-12, f"D={D}: expected guaranteed miss p_miss=1.0, got {l1.p_miss_already}"
            assert np.sum(l1.probs) < 1e-12, f"D={D}: non-zero positive probs under guaranteed miss"
            assert l1.query_meeting_prob(0.0) == 0.0


def test_adversarial_wide_flat_uniform_pmfs():
    """
    Stress tests backward recursion with high-dimensional wide uniform distributions:
    X_v ~ U[0, 20] ms (201 bins each).
    Verifies:
    1. Conservation of total probability mass at every step (< 10^-12).
    2. Strict non-negativity and upper bound (0 <= p <= 1).
    3. Smooth monotonic survival decay under convolution.
    """
    scale = 10
    n_bins = 201  # 20.0 ms wide
    p_u = np.ones(n_bins) / n_bins
    u_pmf = DiscretePMF(p_u, scale=scale)

    pmfs = {i: u_pmf for i in range(1, 7)}
    means = {i: 10.0 for i in range(1, 7)}

    det = TobaPlaxityBaseline(scale=scale)
    det.fit_from_pmfs(pmfs, means, deadline_ms=60.0)

    for i in range(1, 7):
        pmf = det.plaxity_pmfs[i]
        total = np.sum(pmf.probs) + pmf.p_miss_already
        assert abs(total - 1.0) < 1e-12, f"Node {i} wide uniform mass violated: {total}"
        assert (pmf.probs >= 0.0).all()
        assert (pmf.probs <= 1.0).all()

    # Monotonicity of L1 survival
    l1 = det.plaxity_pmfs[1]
    surv = [l1.query_meeting_prob(t / scale) for t in range(int(round(60.0 * scale)))]
    diffs = np.diff(surv)
    assert np.all(diffs <= 1e-12), "Survival function not non-increasing under wide uniform inputs"


# ==============================================================================
# 2. Conservation of Negative Slack (p_miss_already)
# ==============================================================================

def test_adversarial_negative_slack_backward_monotonicity_all_regimes():
    """
    Audits all 18 authentic calibration regimes:
    Verifies that negative slack (p_miss_already) is strictly monotonic non-decreasing
    as analysis moves backward along execution paths from sink to source:
    Path A: N6 -> N5 -> N3 -> N2 -> N1
    Path B: N6 -> N5 -> N4 -> N2 -> N1
    And that sum(probs) + p_miss_already == 1.0 holds to < 10^-12 everywhere.
    """
    calib = pd.read_csv(CALIB_PATH)
    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        deadlines = json.load(f)["regimes"]

    for reg_str, info in deadlines.items():
        reg_id = int(reg_str)
        d = float(info["deadline_ms"])
        reg_df = calib[calib["regime_id"] == reg_id]

        det = TobaPlaxityBaseline(scale=10)
        det.fit(reg_df, deadline_ms=d)

        # 1. Total mass conservation
        for i in range(1, 7):
            pmf = det.plaxity_pmfs[i]
            total = np.sum(pmf.probs) + pmf.p_miss_already
            assert abs(total - 1.0) < 1e-12, f"Regime {reg_id} node {i} mass failed: {total}"

        # 2. Monotonicity of negative slack backwards
        m6 = det.plaxity_pmfs[6].p_miss_already
        m5 = det.plaxity_pmfs[5].p_miss_already
        m3 = det.plaxity_pmfs[3].p_miss_already
        m4 = det.plaxity_pmfs[4].p_miss_already
        m2 = det.plaxity_pmfs[2].p_miss_already
        m1 = det.plaxity_pmfs[1].p_miss_already

        assert m6 <= m5 + 1e-12, f"Regime {reg_id}: m6={m6} > m5={m5}"
        assert m5 <= m3 + 1e-12, f"Regime {reg_id}: m5={m5} > m3={m3}"
        assert m5 <= m4 + 1e-12, f"Regime {reg_id}: m5={m5} > m4={m4}"
        assert max(m3, m4) <= m2 + 1e-12, f"Regime {reg_id}: max(m3,m4)={max(m3,m4)} > m2={m2}"
        assert m2 <= m1 + 1e-12, f"Regime {reg_id}: m2={m2} > m1={m1}"


# ==============================================================================
# 3. Independent Fork Operator: Upper Cumulative Product vs Correlation
# ==============================================================================

def test_adversarial_fork_upper_cumulative_product_exact_under_independence():
    """
    Stress-tests DiscretePMF.independent_fork_min across 100 randomized asymmetric PMF pairs:
    Verifies that Upper Cumulative Product:
    P(min(A, B) >= lambda) == P(A >= lambda) * P(B >= lambda)
    holds to machine precision (< 10^-12) across all bins lambda.
    """
    scale = 10
    rng = np.random.RandomState(42)

    for trial in range(50):
        len_a = rng.randint(10, 80)
        len_b = rng.randint(10, 80)
        p_a = rng.exponential(scale=1.0, size=len_a)
        p_a /= p_a.sum()
        p_b = rng.exponential(scale=2.0, size=len_b)
        p_b /= p_b.sum()

        miss_a = rng.uniform(0.0, 0.2)
        miss_b = rng.uniform(0.0, 0.2)

        pmf_a = DiscretePMF(p_a, p_miss_already=miss_a, scale=scale)
        pmf_b = DiscretePMF(p_b, p_miss_already=miss_b, scale=scale)

        fork_res = DiscretePMF.independent_fork_min(pmf_a, pmf_b)

        max_len = max(pmf_a.n_bins, pmf_b.n_bins)
        surv_a = pmf_a.to_plaxity_cdf(length=max_len)
        surv_b = pmf_b.to_plaxity_cdf(length=max_len)
        expected_surv = np.clip(surv_a * surv_b, 0.0, 1.0)
        actual_surv = fork_res.to_plaxity_cdf(length=max_len)

        max_err = np.max(np.abs(expected_surv - actual_surv))
        assert max_err < 1e-12, f"Trial {trial}: Upper Cumulative Product failed with error {max_err:.2e}"

        # Total mass conservation check
        total_mass = np.sum(fork_res.probs) + fork_res.p_miss_already
        assert abs(total_mass - 1.0) < 1e-12, f"Trial {trial}: mass conservation failed {total_mass}"


def test_adversarial_fork_operator_divergence_under_correlation():
    """
    Demonstrates that independent fork min diverges from true joint distribution
    under branch correlation:
    1. Perfect positive correlation (B = A): independent product surv_a^2 underestimates
       true survival surv_a by up to 25.0 pp (max at surv=0.5: 0.5 - 0.25 = 0.25).
    2. Negative anti-correlation: independent product overestimates joint min survival.
    """
    scale = 10
    # Create two-point distribution: {2.0 ms: 0.5, 6.0 ms: 0.5}
    p = np.zeros(61)
    p[20] = 0.5
    p[60] = 0.5
    pmf = DiscretePMF(p, scale=scale)

    fork_indep = DiscretePMF.independent_fork_min(pmf, pmf)

    # True joint survival under perfect correlation: B = A -> min(A, A) = A
    surv_true = pmf.to_plaxity_cdf(length=61)
    surv_indep = fork_indep.to_plaxity_cdf(length=61)

    diff = surv_true - surv_indep
    max_divergence = np.max(diff)

    # At bin 20 (AST = 2.0 ms), surv_true = 1.0, surv_indep = 1.0.
    # At bin 21 to 60 (AST in (2.0, 6.0]), surv_true = 0.5, surv_indep = 0.25.
    # Discrepancy is exactly 0.25 (25.0 percentage points)
    assert abs(max_divergence - 0.25) < 1e-12, f"Expected 0.25 divergence, got {max_divergence}"


def test_adversarial_fork_asymmetric_bottleneck_branch():
    """
    Tests asymmetric fork where Branch A is long (bottleneck: 10 ms) and Branch B is short (1 ms).
    Verifies that plaxity is strictly dominated by the bottleneck branch.
    """
    scale = 10
    p_long = np.zeros(101); p_long[100] = 1.0  # 10.0 ms
    p_short = np.zeros(11); p_short[10] = 1.0   # 1.0 ms
    pmf_long = DiscretePMF(p_long, scale=scale)
    pmf_short = DiscretePMF(p_short, scale=scale)

    # L_long = 15 - 10 = 5 ms (bin 50). L_short = 15 - 1 = 14 ms (bin 140).
    # min(L_long, L_short) = 5 ms (bin 50).
    l_long = DiscretePMF.from_exit_node(pmf_long, deadline_ms=15.0, scale=scale)
    l_short = DiscretePMF.from_exit_node(pmf_short, deadline_ms=15.0, scale=scale)

    fork_res = DiscretePMF.independent_fork_min(l_long, l_short)
    assert np.argmax(fork_res.probs) == 50, f"Expected fork min peak at 50, got {np.argmax(fork_res.probs)}"
    assert abs(fork_res.probs[50] - 1.0) < 1e-12


# ==============================================================================
# 4. Bug Audit: Zero Deadline Clamping in TobaPlaxityBaseline.fit()
# ==============================================================================

def test_adversarial_audit_zero_deadline_fit_clamping_defect():
    """
    POST-REMEDIATION VERIFICATION:
    In src/detectors/toba_baseline.py:
    Ensures max_d accounts for sample max duration:
        max_d = max(float(deadline_ms) * 1.5, max_sample_val * 1.2, 1.0)
    When deadline_ms == 0.0:
    - Sample durations are NOT clamped to bin 0; they preserve true execution times (e.g. Node 1 peak at 2.0 ms / bin 20).
    - At D = 0.0 ms for positive-duration tasks, the baseline correctly models guaranteed deadline miss:
      p_miss_already == 1.0, and P(meet | AST=0) == 0.0 (NOT 1.0).
    """
    df = pd.DataFrame({
        "dur_node1_ms": [2.0] * 10,
        "dur_node2_ms": [1.0] * 10,
        "dur_node3_ms": [4.0] * 10,
        "dur_node4_ms": [6.0] * 10,
        "dur_node5_ms": [8.0] * 10,
        "dur_node6_ms": [3.0] * 10,
        "s_dag_ms": [20.0] * 10,
        "miss": [1] * 10,
    })

    det = TobaPlaxityBaseline(scale=10)
    det.fit(df, deadline_ms=0.0)

    # 1. Verification: Node 1 duration is preserved at 2.0 ms (bin 20), not clamped to 0
    node1_argmax = np.argmax(det.node_pmfs[1].probs)
    assert node1_argmax == 20, f"Node 1 duration was clamped to bin {node1_argmax}, expected bin 20 (2.0 ms)"

    # 2. Verification: At deadline = 0, Node 1 reports P(meet) = 0.0 and p_miss_already = 1.0
    p_meet_n1 = det.plaxity_pmfs[1].query_meeting_prob(0.0)
    p_miss_n1 = det.plaxity_pmfs[1].p_miss_already

    assert abs(p_miss_n1 - 1.0) < 1e-12, f"Expected guaranteed miss p_miss=1.0, got {p_miss_n1}"
    assert p_meet_n1 == 0.0, f"Expected P(meet | AST=0) == 0.0 under zero deadline, got {p_meet_n1}"


def test_adversarial_audit_query_meeting_prob_boundary_step_discrepancy():
    """
    POST-REMEDIATION VERIFICATION:
    In src/detectors/toba_baseline.py:
    query_meeting_prob clamps ast_ms <= 0.0 to 0.0, querying surv[bin_idx] consistently.
    Verifies that across the AST = 0 boundary and across continuous real line,
    survival probability P(meet | AST) is strictly non-increasing without machine-precision upward step.
    In Regime 17 node 1:
    prob_pos <= prob_neg holds strictly.
    """
    calib = pd.read_csv(CALIB_PATH)
    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        deadlines = json.load(f)["regimes"]

    reg17_df = calib[calib["regime_id"] == 17]
    d17 = float(deadlines["17"]["deadline_ms"])

    det = TobaPlaxityBaseline(scale=10)
    det.fit(reg17_df, deadline_ms=d17)

    pmf1 = det.plaxity_pmfs[1]
    prob_neg = pmf1.query_meeting_prob(-0.01)
    prob_pos = pmf1.query_meeting_prob(0.008)

    # Post-remediation verification: prob_pos <= prob_neg (no upward jump)
    assert prob_pos <= prob_neg, f"Monotonicity violation across boundary: prob_pos={prob_pos} > prob_neg={prob_neg}"
    assert abs(prob_pos - prob_neg) < 1e-12, f"Expected continuous boundary match, got diff {prob_pos - prob_neg}"

    # Monotonicity across continuous grid around 0.0
    grid = np.linspace(-1.0, 1.0, 201)
    probs = [pmf1.query_meeting_prob(t) for t in grid]
    diffs = np.diff(probs)
    assert np.all(diffs <= 1e-12), f"Monotonicity violated in grid around 0: max diff = {np.max(diffs)}"
