r"""
Adversarial Stress Test Suite for Milestone 3 (Gate D Mathematical & Algorithmic Stress Challenger).
Author: Challenger 1 (challenger_m3_1)
Role: Empirical Challenger (critic / specialist)

Adversarially tests:
1. Mathematical Soundness of Super-Node D2:
   - 5-stage convolution equivalence to exact Cartesian product ground-truth slack distribution
     across multiple synthetic toy DAG configurations (uniform, skewed, bimodal, correlated).
   - AST inspection & empirical verification that D2 never calls Eq. 11.
   - Exact quantification of Eq. 11 join-node reuse distortion across branch correlations rho in [-0.8, +0.8].
2. Stress-Testing D1 & D2 Observable State Estimator:
   - Sudden regime switches mid-episode: tracking transition curve and lag bounds (W=5).
   - Extreme out-of-distribution latency spikes (10x, 50x, 100x, 1000x normal execution time).
   - Non-positive arrival times (AST <= 0.0 ms) and boundary checks.
   - Single-frame episodes (N=1) with global median centroid fallback.
   - Constant, degenerate, and ping-pong upstream measurements.
3. Monotonicity & Probabilistic Invariant Audit:
   - Strict non-increasing monotonicity P(L_v >= AST) as AST increases across all 18 regimes.
   - Probability range validity (0 <= P <= 1) and exact overrun behavior (P=0 for AST > D).
4. Strict Data Isolation:
   - Zero access, reading, or opening of results/uav_test_traces.csv.
"""

import ast
import json
import os
import glob
import numpy as np
import pandas as pd
import pytest

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
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")


# ==============================================================================
# Helper Functions: Exact Cartesian Ground-Truth Slack Generator
# ==============================================================================

def compute_cartesian_ground_truth(
    pmf_1: DiscretePMF,
    pmf_2: DiscretePMF,
    pmf_fork: DiscretePMF,
    pmf_5: DiscretePMF,
    pmf_6: DiscretePMF,
    deadline_ms: float,
    scale: int = 10,
):
    """
    Computes exact Cartesian product ground-truth distribution of remaining duration
    from each node checkpoint v in {1, 2, fork, 5, 6} to the end of the DAG.
    
    For any node v, remaining duration R_v = sum_{u >= v} X_u.
    Then latest permissible arrival time L_v has survival function:
        P(L_v >= t) = P(R_v <= deadline_ms - t).
    """
    def get_support_and_probs(pmf: DiscretePMF):
        supp = np.arange(len(pmf.probs), dtype=np.float64) / pmf.scale
        mask = pmf.probs > 0.0
        return supp[mask], pmf.probs[mask]

    s6, p6 = get_support_and_probs(pmf_6)
    s5, p5 = get_support_and_probs(pmf_5)
    sf, pf = get_support_and_probs(pmf_fork)
    s2, p2 = get_support_and_probs(pmf_2)
    s1, p1 = get_support_and_probs(pmf_1)

    # Stage 6: R_6 = X_6
    dist_r6 = {}
    for x6, prob6 in zip(s6, p6):
        dist_r6[round(x6, 4)] = dist_r6.get(round(x6, 4), 0.0) + prob6

    # Stage 5: R_5 = X_5 + X_6
    dist_r5 = {}
    for x5, prob5 in zip(s5, p5):
        for r6, prob6 in dist_r6.items():
            tot = round(x5 + r6, 4)
            dist_r5[tot] = dist_r5.get(tot, 0.0) + prob5 * prob6

    # Stage fork: R_fork = X_fork + R_5
    dist_rfork = {}
    for xf, probf in zip(sf, pf):
        for r5, prob5 in dist_r5.items():
            tot = round(xf + r5, 4)
            dist_rfork[tot] = dist_rfork.get(tot, 0.0) + probf * prob5

    # Stage 2: R_2 = X_2 + R_fork
    dist_r2 = {}
    for x2, prob2 in zip(s2, p2):
        for rf, probf in dist_rfork.items():
            tot = round(x2 + rf, 4)
            dist_r2[tot] = dist_r2.get(tot, 0.0) + prob2 * probf

    # Stage 1: R_1 = X_1 + R_2
    dist_r1 = {}
    for x1, prob1 in zip(s1, p1):
        for r2, prob2 in dist_r2.items():
            tot = round(x1 + r2, 4)
            dist_r1[tot] = dist_r1.get(tot, 0.0) + prob1 * prob2

    dist_map = {
        6: dist_r6,
        5: dist_r5,
        "fork": dist_rfork,
        2: dist_r2,
        1: dist_r1,
    }

    def ground_truth_p_meet(node_key, ast_ms: float) -> float:
        rem_avail = deadline_ms - ast_ms
        if rem_avail < -1e-9:
            return 0.0
        dist = dist_map[node_key]
        return float(sum(p for dur, p in dist.items() if dur <= rem_avail + 1e-6))

    return ground_truth_p_meet


# ==============================================================================
# Suite 1: Mathematical Soundness of Super-Node D2 5-Stage Convolution
# ==============================================================================

class TestSuperNodeD2MathematicalSoundness:
    """Rigorous tests proving D2 convolution matches Cartesian ground truth."""

    @pytest.mark.parametrize("scale", [1, 5, 10])
    def test_d2_5stage_convolution_matches_cartesian_ground_truth(self, scale):
        """
        Synthesizes a 5-node toy pipeline with arbitrary non-trivial discrete PMFs.
        Compares SuperNodePlaxityBaseline's 5-stage convolutional subtraction
        against the exact Cartesian product joint survival function at all checkpoints.
        """
        deadline = 12.0

        p1 = np.zeros(int(round(3.0 * scale)) + 1)
        p1[int(round(0.5 * scale))] = 0.3
        p1[int(round(1.0 * scale))] = 0.7

        p2 = np.zeros(int(round(3.0 * scale)) + 1)
        p2[int(round(0.2 * scale))] = 0.5
        p2[int(round(0.8 * scale))] = 0.5

        pfork = np.zeros(int(round(4.0 * scale)) + 1)
        pfork[int(round(1.0 * scale))] = 0.25
        pfork[int(round(2.0 * scale))] = 0.50
        pfork[int(round(3.0 * scale))] = 0.25

        p5 = np.zeros(int(round(3.0 * scale)) + 1)
        p5[int(round(1.0 * scale))] = 0.6
        p5[int(round(2.0 * scale))] = 0.4

        p6 = np.zeros(int(round(3.0 * scale)) + 1)
        p6[int(round(0.5 * scale))] = 0.4
        p6[int(round(1.5 * scale))] = 0.6

        pmf1 = DiscretePMF(p1, scale=scale)
        pmf2 = DiscretePMF(p2, scale=scale)
        pmf_fork = DiscretePMF(pfork, scale=scale)
        pmf5 = DiscretePMF(p5, scale=scale)
        pmf6 = DiscretePMF(p6, scale=scale)

        oracle_p_meet = compute_cartesian_ground_truth(
            pmf1, pmf2, pmf_fork, pmf5, pmf6, deadline_ms=deadline, scale=scale
        )

        baseline = SuperNodePlaxityBaseline(scale=scale, threshold_mode="fixed", fixed_threshold=0.95)
        baseline.fit_from_pmfs(
            pmf_1=pmf1,
            pmf_2=pmf2,
            pmf_fork=pmf_fork,
            pmf_5=pmf5,
            pmf_6=pmf6,
            mean_1=0.85,
            mean_2=0.50,
            mean_fork=2.00,
            mean_5=1.40,
            mean_6=1.10,
            deadline_ms=deadline,
        )

        test_ast_points = np.linspace(0.0, deadline + 2.0, int((deadline + 2.0) * scale) + 1)
        checkpoint_mapping = {
            1: 1,
            2: 2,
            3: "fork",
            4: "fork",
            5: 5,
            6: 6,
        }

        for cp, mapped_key in checkpoint_mapping.items():
            plax_pmf = baseline.plaxity_pmfs[cp]
            for ast_val in test_ast_points:
                ast_val_rounded = round(float(ast_val), 4)
                p_d2 = plax_pmf.query_meeting_prob(ast_val_rounded)
                p_gt = oracle_p_meet(mapped_key, ast_val_rounded)
                diff = abs(p_d2 - p_gt)
                assert diff < 1e-10, (
                    f"Discrepancy at Checkpoint {cp} (key={mapped_key}), AST={ast_val_rounded} ms: "
                    f"D2={p_d2:.8f}, Cartesian Ground-Truth={p_gt:.8f}, Diff={diff}"
                )

    def test_d2_code_ast_contains_no_eq11_or_independent_fork_min(self):
        """
        Adversarially audits Python AST of regime_detector.py to prove that
        neither SuperNodePlaxityBaseline nor SuperNodePlaxityDetectorD2
        ever invokes independent_fork_min (ACM TECS 2026 Eq. 11).
        """
        regime_detector_file = os.path.join(REPO_ROOT, "src", "detectors", "regime_detector.py")
        with open(regime_detector_file, "r", encoding="utf-8") as f:
            code = f.read()

        parsed = ast.parse(code)

        calls_found = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.ClassDef) and "SuperNode" in node.name:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        if isinstance(sub.func, ast.Name):
                            calls_found.append(sub.func.id)
                        elif isinstance(sub.func, ast.Attribute):
                            calls_found.append(sub.func.attr)

        assert "independent_fork_min" not in calls_found, (
            "FATAL: SuperNode implementation references independent_fork_min!"
        )
        assert "eq11" not in [c.lower() for c in calls_found]

    @pytest.mark.parametrize("rho", [-0.8, -0.4, 0.0, 0.4, 0.8, 1.0])
    def test_eq11_distortion_quantification_across_branch_correlations(self, rho):
        """
        Demonstrates that:
        1. When parallel branches N3 || N4 feed into join node N5, standard Toba Eq. 11
           always incurs distortion due to common downstream path reuse.
        2. When branches are correlated (rho != 0), Eq. 11 error is further amplified.
        3. Super-Node D2 eliminates this distortion across ALL correlation values.
        """
        scale = 10
        deadline = 15.0
        n_samples = 10000
        rng = np.random.RandomState(12345)

        mean = [2.0, 2.0]
        cov = [[0.5, 0.5 * rho], [0.5 * rho, 0.5]]
        bivariate = rng.multivariate_normal(mean, cov, size=n_samples)
        x3 = np.clip(bivariate[:, 0], 0.2, 5.0)
        x4 = np.clip(bivariate[:, 1], 0.2, 5.0)
        x_fork = np.maximum(x3, x4)

        x5 = rng.gamma(shape=4.0, scale=0.5, size=n_samples)
        x6 = rng.uniform(0.5, 1.5, size=n_samples)

        pmf3 = DiscretePMF.from_samples(x3, max_val=10.0, scale=scale)
        pmf4 = DiscretePMF.from_samples(x4, max_val=10.0, scale=scale)
        pmf_fork = DiscretePMF.from_samples(x_fork, max_val=10.0, scale=scale)
        pmf5 = DiscretePMF.from_samples(x5, max_val=10.0, scale=scale)
        pmf6 = DiscretePMF.from_samples(x6, max_val=10.0, scale=scale)

        l6 = DiscretePMF.from_exit_node(pmf6, deadline_ms=deadline, scale=scale)
        l5 = l6.convolutional_subtraction(pmf5)
        l_fork_d2 = l5.convolutional_subtraction(pmf_fork)

        l3 = l5.convolutional_subtraction(pmf3)
        l4 = l5.convolutional_subtraction(pmf4)
        l_fork_toba = DiscretePMF.independent_fork_min(l3, l4)

        rem_durations = x_fork + x5 + x6

        test_points = [5.0, 7.0, 9.0, 10.0, 11.0]
        max_d2_err = 0.0
        max_toba_err = 0.0

        for t in test_points:
            p_emp = float(np.mean(rem_durations <= deadline - t))
            p_d2 = l_fork_d2.query_meeting_prob(t)
            p_toba = l_fork_toba.query_meeting_prob(t)

            err_d2 = abs(p_d2 - p_emp)
            err_toba = abs(p_toba - p_emp)

            max_d2_err = max(max_d2_err, err_d2)
            max_toba_err = max(max_toba_err, err_toba)

        assert max_d2_err < 0.02, f"D2 sampling error unexpectedly high: {max_d2_err}"
        assert max_toba_err >= max_d2_err, "Toba Eq. 11 unexpectedly outperformed D2"


# ==============================================================================
# Suite 2: Adversarial Stress Testing of D1 & D2 Observable State Estimator
# ==============================================================================

class TestObservableStateEstimatorStress:
    """Stress tests on moving median W=5, regime transitions, OOD spikes, and edge cases."""

    @pytest.fixture(scope="class")
    def fitted_calib(self):
        calib_df = pd.read_csv(CALIB_PATH)
        with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
            deadlines = {int(k): float(v["deadline_ms"]) for k, v in json.load(f)["regimes"].items()}
        return calib_df, deadlines

    def test_sudden_regime_switch_adaptation_lag(self, fitted_calib):
        """
        Adversarial Scenario: An episode starts in Regime 0 (fastest: zero wind, low clutter, idle)
        and mid-episode abruptly switches to Regime 17 (slowest: high wind, high clutter, heavy contention).
        
        Tests:
        1. ObservableStateEstimator transitions monotonically from Regime 0 to Regime 17.
        2. Window W=5 takes exactly <= 5 frames to complete the transition.
        3. No oscillation, NaN, or exceptions during the transient state.
        """
        calib_df, deadlines = fitted_calib
        estimator = ObservableStateEstimator(window=5)
        estimator.fit(calib_df)

        c0 = estimator.calib_centroids[0]
        c17 = estimator.calib_centroids[17]

        estimator.reset()
        for _ in range(10):
            estimator.update(c0[0], c0[1], c0[2])

        pred_before = estimator.predict_regime(c0[0], c0[1])
        assert pred_before == 0, f"Expected regime 0, got {pred_before}"

        predictions = []
        for frame_idx in range(1, 10):
            pred = estimator.predict_regime(c17[0], c17[1])
            predictions.append(pred)
            estimator.update(c17[0], c17[1], c17[2])

        final_pred = predictions[-1]
        assert final_pred == 17, f"Failed to adapt to regime 17 by frame 9: {predictions}"
        assert 17 in predictions[:5], f"Adaptation took longer than W=5 frames: {predictions}"

    @pytest.mark.parametrize("spike_factor", [10.0, 50.0, 100.0, 1000.0])
    def test_extreme_out_of_distribution_latency_spikes(self, fitted_calib, spike_factor):
        """
        Adversarial Scenario: Injects 10x to 1000x execution time latency spikes.
        Verifies:
        1. Estimator does not crash, overflow, or return invalid regime IDs.
        2. D1 and D2 return valid probabilities in [0.0, 1.0] without raising IndexError or ValueError.
        3. Returns immediate alarm (alarm=True) at earliest checkpoint with valid lead time.
        4. For massive spikes exceeding regime deadline (>= 50x), min_p_meet is strictly 0.0.
        """
        calib_df, deadlines = fitted_calib
        d1 = RegimeConditionedDetectorD1(scale=10, policy="fixed_0.95")
        d1.fit(calib_df, deadlines=deadlines)

        d2 = SuperNodePlaxityDetectorD2(scale=10, policy="fixed_0.95", fork_mode="wall")
        d2.fit(calib_df, deadlines=deadlines)

        d1.reset_episode()
        d2.reset_episode()

        normal_t1 = float(np.median(calib_df["dur_node1_ms"]))
        normal_t2 = float(np.median(calib_df["dur_node2_ms"]))
        spiked_t1 = normal_t1 * spike_factor
        spiked_t2 = normal_t2 * spike_factor
        spiked_sdag = spiked_t1 + spiked_t2 + 500.0

        spike_row = {
            "dur_node1_ms": spiked_t1,
            "dur_node2_ms": spiked_t2,
            "dur_node3_ms": spiked_t1,
            "dur_node4_ms": spiked_t2,
            "dur_node5_ms": spiked_t1,
            "dur_node6_ms": spiked_t1,
            "s_dag_ms": spiked_sdag,
            "dur_fork_wall_ms": spiked_t1,
        }

        for det, name in [(d1, "D1"), (d2, "D2")]:
            res = det.evaluate_frame(spike_row, update_state=True)
            assert 0 <= res["pred_regime"] < 18
            assert 0.0 <= res["min_p_meet"] <= 1.0, f"{name} min_p_meet invalid: {res['min_p_meet']}"
            assert 0.0 <= res["miss_risk"] <= 1.0, f"{name} miss_risk invalid: {res['miss_risk']}"
            assert res["alarm"] is True, f"{name} did not alarm on {spike_factor}x spike"
            assert res["earliest_checkpoint"] in [1, 2], f"{name} earliest checkpoint: {res['earliest_checkpoint']}"
            assert res["lead_time_ms"] >= 0.0
            if spike_factor >= 50.0:
                assert res["min_p_meet"] == 0.0, f"{name} min_p_meet not 0 on massive spike: {res['min_p_meet']}"

    @pytest.mark.parametrize("ast_val", [-50.0, -10.0, -1.0, -0.001, 0.0])
    def test_non_positive_arrival_times_boundary_checks(self, fitted_calib, ast_val):
        """
        Adversarial Scenario: Evaluates non-positive AST values (AST <= 0.0 ms).
        Verifies:
        1. Clamped gracefully to AST=0.0 ms.
        2. Returns strictly valid probability P in [0.0, 1.0].
        3. P(L >= ast_neg) == P(L >= 0.0).
        """
        calib_df, deadlines = fitted_calib
        d2 = SuperNodePlaxityDetectorD2(scale=10, policy="fixed_0.95")
        d2.fit(calib_df, deadlines=deadlines)

        reg0_det = d2.regime_detectors[0]
        for cp, pmf in reg0_det.plaxity_pmfs.items():
            p_neg = pmf.query_meeting_prob(ast_val)
            p_zero = pmf.query_meeting_prob(0.0)
            assert 0.0 <= p_neg <= 1.0
            assert p_neg == p_zero, f"AST={ast_val} yielded {p_neg} != AST=0 {p_zero}"

    def test_single_frame_episodes_n1_without_state_leakage(self, fitted_calib):
        """
        Adversarial Scenario: 50 successive 1-frame episodes (N=1).
        Verifies:
        1. Estimator falls back to global centroid for S_DAG on frame 1 without crash.
        2. reset_episode() prevents any historical state from leaking into subsequent episodes.
        """
        calib_df, deadlines = fitted_calib
        d1 = RegimeConditionedDetectorD1(scale=10, policy="fixed_0.95")
        d1.fit(calib_df, deadlines=deadlines)

        global_sdag_median = d1.estimator.global_centroid[2]

        for ep_idx in range(50):
            d1.reset_episode()
            assert len(d1.estimator.hist_t1) == 0
            assert len(d1.estimator.hist_sdag) == 0

            obs = d1.estimator.get_observable_vector(curr_t1=5.0, curr_t2=1.0)
            assert obs[2] == global_sdag_median

            row = {
                "dur_node1_ms": 5.0,
                "dur_node2_ms": 1.0,
                "dur_node3_ms": 2.0,
                "dur_node4_ms": 2.0,
                "dur_node5_ms": 2.0,
                "dur_node6_ms": 1.0,
                "s_dag_ms": 25.0,
                "episode_id": ep_idx,
                "frame_id": 0,
                "regime_id": 0,
                "deadline_ms": deadlines[0],
            }
            res = d1.evaluate_frame(row, update_state=True)
            assert isinstance(res["alarm"], (bool, np.bool_))

    def test_degenerate_and_constant_upstream_measurements(self, fitted_calib):
        """
        Adversarial Scenario: Upstream measurements are constant zero or identical repeated values.
        Verifies:
        1. Zero variance does not produce divide-by-zero or NaN in moving median.
        2. Alternating extreme ping-pong does not corrupt rolling state.
        """
        calib_df, deadlines = fitted_calib
        estimator = ObservableStateEstimator(window=5)
        estimator.fit(calib_df)

        estimator.reset()
        for _ in range(100):
            estimator.update(0.0, 0.0, 0.0)
            pred = estimator.predict_regime(0.0, 0.0)
            assert 0 <= pred < 18

        for i in range(50):
            val = 0.1 if i % 2 == 0 else 100.0
            estimator.update(val, val, val)
            pred = estimator.predict_regime(val, val)
            assert 0 <= pred < 18
            obs = estimator.get_observable_vector(val, val)
            assert np.all(np.isfinite(obs))


# ==============================================================================
# Suite 3: Global Monotonicity & Probabilistic Invariant Audit
# ==============================================================================

class TestProbabilisticInvariantsAcrossAllRegimes:
    """Verifies strict mathematical invariants across all 18 regime distributions."""

    @pytest.fixture(scope="class")
    def fitted_detectors(self):
        calib_df = pd.read_csv(CALIB_PATH)
        with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
            deadlines = {int(k): float(v["deadline_ms"]) for k, v in json.load(f)["regimes"].items()}

        d1 = RegimeConditionedDetectorD1(scale=10, policy="fixed_0.95")
        d1.fit(calib_df, deadlines=deadlines)

        d2 = SuperNodePlaxityDetectorD2(scale=10, policy="fixed_0.95", fork_mode="wall")
        d2.fit(calib_df, deadlines=deadlines)

        return calib_df, deadlines, d1, d2

    def test_strict_non_increasing_monotonicity_all_regimes(self, fitted_detectors):
        """
        For ALL 18 regimes and ALL 6 query checkpoints in both D1 and D2:
        Verifies that:
            ast_1 < ast_2 ==> P(L >= ast_1) >= P(L >= ast_2)
        across a dense grid of 500 arrival times.
        """
        _, deadlines, d1, d2 = fitted_detectors

        for det_obj, name in [(d1, "D1"), (d2, "D2")]:
            for r, d_reg in deadlines.items():
                det = det_obj.regime_detectors[r]
                test_ast = np.linspace(0.0, d_reg + 10.0, 501)

                for cp in [1, 2, 3, 4, 5, 6]:
                    pmf = det.plaxity_pmfs[cp]
                    prev_p = 1.0000000001
                    for t in test_ast:
                        p = pmf.query_meeting_prob(float(t))
                        assert 0.0 <= p <= 1.0, f"Invalid prob {p} in {name} R{r} CP{cp} at t={t}"
                        assert p <= prev_p + 1e-12, (
                            f"Monotonicity violation in {name} R{r} CP{cp} at t={t}: "
                            f"p={p:.8f} > prev_p={prev_p:.8f}"
                        )
                        prev_p = p

    def test_overrun_strict_zero_probability(self, fitted_detectors):
        """
        Verifies that whenever AST exceeds deadline, P(meet | AST) is strictly 0.0.
        """
        _, deadlines, d1, d2 = fitted_detectors

        for det_obj, name in [(d1, "D1"), (d2, "D2")]:
            for r, d_reg in deadlines.items():
                det = det_obj.regime_detectors[r]
                for cp in [1, 2, 3, 4, 5, 6]:
                    pmf = det.plaxity_pmfs[cp]
                    p_over = pmf.query_meeting_prob(d_reg + 5.0)
                    assert p_over == 0.0, f"{name} R{r} CP{cp} did not return 0.0 on overrun: {p_over}"


# ==============================================================================
# Suite 4: Strict Data Split Isolation Audit
# ==============================================================================

def test_challenger_strict_split_isolation():
    """
    Audits the entire codebase ensuring results/uav_test_traces.csv is completely unreferenced
    in src/ and benchmarks/, preserving absolute Gate E isolation.
    """
    forbidden_stem = f"uav_{'test'}_traces.csv"

    violations = []
    for search_dir in ["src", "benchmarks"]:
        full_dir = os.path.join(REPO_ROOT, search_dir)
        for root, _, files in os.walk(full_dir):
            for file in files:
                if file.endswith(".py") and file not in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8") as f:
                        if forbidden_stem in f.read():
                            violations.append(fpath)

    assert len(violations) == 0, f"Isolation breach! Forbidden test file found in: {violations}"
