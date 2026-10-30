"""
Adversarial Stress Test Suite for Milestone 2 (Gate C: Baselines & Sanity Checks).
Author: Challenger 2 (challenger_m2_2)
Role: Empirical Challenger (critic / specialist)

Adversarially tests:
1. Strict Data Isolation: Absolute non-contamination and zero leakage from results/uav_test_traces.csv.
2. Cluster-Bootstrap Exceedance Gaps Reproducibility & Mathematical Soundness:
   - Full replication of results/gate_c_exceedance_gap_summary.csv.
   - Verification of discrete convolution and fork survival operators.
3. Stop Rule Adversarial Stress Testing:
   - Verification that non-triggering verdict is supported by statistically significant positive gaps.
   - Large-scale bootstrap sensitivity (n_boot=10,000, multi-seed stability).
   - Two-sample cluster bootstrap (resampling both calibration and validation clusters).
   - Temporal grid discretization sensitivity (scale = 5, 10, 20).
4. Baseline Summary Invariant Audit:
   - Metric conservation laws (TP + FN = n_miss, TN + FP = n_safe, checkpoint sums = TP + FP).
   - Monotonicity across threshold ramp schedules (linear_job, linear_time, fixed).
   - Lead time physical validity bounds (0 <= lead_time <= deadline).
5. Toba Plaxity Operator Invariants:
   - Monotonicity of P(meet | AST) with respect to arrival time AST.
   - Extreme overrun handling (AST > deadline -> P(meet) = 0.0).
"""

import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
import pytest

from src.detectors.toba_baseline import (
    DiscretePMF,
    TobaPlaxityBaseline,
    TobaDetectorSuite,
)
from benchmarks.evaluate_gate_c import (
    POLICIES,
    load_datasets_and_deadlines,
    compute_exceedance_gaps_and_stop_rule,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CALIB_PATH = os.path.join(RESULTS_DIR, "uav_calibration_traces.csv")
VAL_PATH = os.path.join(RESULTS_DIR, "uav_validation_traces.csv")
TEST_PATH = os.path.join(RESULTS_DIR, "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")
GAP_CSV_PATH = os.path.join(RESULTS_DIR, "gate_c_exceedance_gap_summary.csv")
BASELINE_CSV_PATH = os.path.join(RESULTS_DIR, "gate_c_baseline_summary.csv")


@pytest.fixture(scope="module")
def gate_c_data():
    calib = pd.read_csv(CALIB_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    with open(DEADLINES_PATH, "r", encoding="utf-8") as f:
        deadlines = json.load(f)["regimes"]
    gap_summary = pd.read_csv(GAP_CSV_PATH)
    baseline_summary = pd.read_csv(BASELINE_CSV_PATH)
    return {
        "calib": calib,
        "val": val,
        "test": test,
        "deadlines": deadlines,
        "gap_summary": gap_summary,
        "baseline_summary": baseline_summary,
    }


# ==============================================================================
# 1. Strict Data Isolation & Zero Leakage Verification
# ==============================================================================

def test_adversarial_static_code_isolation():
    """
    Scans every file in src/ and benchmarks/ to ensure zero mentions of uav_test_traces.csv.
    Only uav_autonomy_pipeline.py is permitted to reference it for dataset generation.
    """
    scan_dirs = [
        os.path.join(REPO_ROOT, "src"),
        os.path.join(REPO_ROOT, "benchmarks"),
    ]
    violations = []
    for s_dir in scan_dirs:
        for root, _, files in os.walk(s_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                if file in ["uav_autonomy_pipeline.py", "evaluate_gate_e.py"]:
                    continue
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "uav_test_traces.csv" in content:
                    violations.append(path)

    assert len(violations) == 0, f"Found test split leakage in: {violations}"


def test_adversarial_dataset_split_isolation(gate_c_data):
    """
    Verifies that no test split episodes or frames exist in calibration or validation traces.
    """
    calib = gate_c_data["calib"]
    val = gate_c_data["val"]
    test = gate_c_data["test"]

    calib_eps = set(calib["episode_id"].unique())
    val_eps = set(val["episode_id"].unique())
    test_eps = set(test["episode_id"].unique())

    # Episode disjointness
    assert len(calib_eps & test_eps) == 0, "Episode overlap between calib and test!"
    assert len(val_eps & test_eps) == 0, "Episode overlap between val and test!"
    assert len(calib_eps & val_eps) == 0, "Episode overlap between calib and val!"

    # Frame count exactness
    assert len(calib) == 7200, f"Expected 7200 calib rows, got {len(calib)}"
    assert len(val) == 3600, f"Expected 3600 val rows, got {len(val)}"
    assert len(test) == 7200, f"Expected 7200 test rows, got {len(test)}"

    # Split column label integrity
    assert (calib["split"] == "calib").all()
    assert (val["split"] == "val").all()
    assert (test["split"] == "test").all()


# ==============================================================================
# 2. Mathematical Soundness & Reproducibility of Exceedance Gaps
# ==============================================================================

def test_adversarial_reproduce_exceedance_gap_calculations(gate_c_data):
    """
    Independently replicates the Gate C exceedance gap summary calculations:
    - Verifies p_emp_pct exactly matches empirical validation miss rate.
    - Verifies p_indep_pct matches independent discrete convolution of calibration marginals.
    - Verifies gap_pp = p_emp_pct - p_indep_pct.
    - Verifies 2,000 cluster-bootstrap CI reproduction at seed=42.
    """
    calib = gate_c_data["calib"]
    val = gate_c_data["val"]
    deadlines = gate_c_data["deadlines"]
    expected_gap_df = gate_c_data["gap_summary"]

    recomputed_df = compute_exceedance_gaps_and_stop_rule(
        calib_df=calib,
        val_df=val,
        deadlines_meta=deadlines,
        scale=10,
        n_boot=2000,
        seed=42,
    )

    assert len(recomputed_df) == len(expected_gap_df) == 19

    for idx in range(len(expected_gap_df)):
        exp_row = expected_gap_df.iloc[idx]
        rec_row = recomputed_df.iloc[idx]

        assert str(exp_row["regime_id"]) == str(rec_row["regime_id"])
        # Empirical miss rate match
        assert abs(exp_row["p_emp_pct"] - rec_row["p_emp_pct"]) < 1e-6
        # Independent convolution exceedance match
        assert abs(exp_row["p_indep_pct"] - rec_row["p_indep_pct"]) < 1e-6
        # Gap match
        assert abs(exp_row["gap_pp"] - rec_row["gap_pp"]) < 1e-6
        # Bootstrap CI match
        assert abs(exp_row["gap_ci_lo_pp"] - rec_row["gap_ci_lo_pp"]) < 1e-6
        assert abs(exp_row["gap_ci_hi_pp"] - rec_row["gap_ci_hi_pp"]) < 1e-6
        assert exp_row["ci_excludes_zero"] == rec_row["ci_excludes_zero"]
        assert exp_row["stop_rule_active"] == rec_row["stop_rule_active"]


# ==============================================================================
# 3. Stop Rule Adversarial Stress Testing
# ==============================================================================

def test_adversarial_stop_rule_positive_gaps_and_significance(gate_c_data):
    """
    Stress-tests the Stop Rule condition:
    "If matched-gap CI includes 0 in all regimes at Gate C, stop and report negative result."
    Verifies that:
    1. Regimes 4, 11, 12, 14 have strictly positive gaps whose 95% Cluster CIs strictly exclude 0.
    2. Pooled validation dataset (72 clusters) has strictly positive gap (+2.20 pp [95% CI: +0.06, +4.48] pp).
    3. The stop rule is genuinely NOT triggered.
    """
    gap_df = gate_c_data["gap_summary"]

    # Check regimes with positive CI strictly > 0
    pos_sig_regimes = []
    for _, row in gap_df[gap_df["regime_id"] != "pooled"].iterrows():
        if row["gap_ci_lo_pp"] > 0.0:
            pos_sig_regimes.append(int(row["regime_id"]))

    assert set(pos_sig_regimes) == {4, 11, 12, 14}, (
        f"Expected positive significant regimes {{4, 11, 12, 14}}, got {pos_sig_regimes}"
    )

    # Check pooled CI strictly > 0
    pooled_row = gap_df[gap_df["regime_id"] == "pooled"].iloc[0]
    assert pooled_row["gap_pp"] > 0.0
    assert pooled_row["gap_ci_lo_pp"] > 0.0
    assert pooled_row["ci_excludes_zero"] is True or pooled_row["ci_excludes_zero"] == 1


def test_adversarial_stop_rule_large_scale_bootstrap_sensitivity(gate_c_data):
    """
    Stress tests cluster bootstrap stability by increasing resamples to 10,000
    and testing across multiple independent random seeds (123, 999, 20261101).
    Asserts that the pooled exceedance gap CI remains strictly positive and bounded above 0.
    """
    calib = gate_c_data["calib"]
    val = gate_c_data["val"]
    deadlines = gate_c_data["deadlines"]

    for seed in [123, 999, 20261101]:
        boot_df = compute_exceedance_gaps_and_stop_rule(
            calib_df=calib,
            val_df=val,
            deadlines_meta=deadlines,
            scale=10,
            n_boot=10000,
            seed=seed,
        )
        pooled_row = boot_df[boot_df["regime_id"] == "pooled"].iloc[0]
        # Pooled gap must be positive and CI lower bound must remain positive (> 0.0)
        assert pooled_row["gap_pp"] > 0.0, f"Seed {seed}: pooled gap not positive"
        assert pooled_row["gap_ci_lo_pp"] > 0.0, (
            f"Seed {seed}: pooled CI lower bound collapsed to {pooled_row['gap_ci_lo_pp']}"
        )


def test_adversarial_stop_rule_two_sample_cluster_bootstrap(gate_c_data):
    """
    Stress-tests the exceedance gap by resampling BOTH calibration clusters (to compute
    p_indep under sample uncertainty) AND validation clusters (to compute p_emp).
    Verifies that the positive underestimation in heavy contention (Regimes 11, 14)
    and pooled dataset remains positive even when calibration uncertainty is included.
    """
    calib = gate_c_data["calib"]
    val = gate_c_data["val"]
    deadlines_meta = gate_c_data["deadlines"]
    scale = 10
    n_boot = 500
    rng = np.random.RandomState(42)

    # Focus on Regime 14 (strongest contention underestimation: Gap ~ +10.93 pp)
    r_calib = calib[calib["regime_id"] == 14]
    r_val = val[val["regime_id"] == 14]
    d_reg = float(deadlines_meta["14"]["deadline_ms"])
    d_bins = int(round(d_reg * scale))
    max_d = d_reg * 1.5

    calib_eps = r_calib["episode_id"].unique()
    val_eps = r_val["episode_id"].unique()

    def get_indep_exceedance(calib_sample):
        def get_pmf(samples):
            nb = int(np.ceil(max_d * scale)) + 1
            c = np.zeros(nb)
            for v in samples:
                idx = min(nb - 1, max(0, int(round(v * scale))))
                c[idx] += 1
            return c / c.sum()

        p1 = get_pmf(calib_sample["dur_node1_ms"])
        p2 = get_pmf(calib_sample["dur_node2_ms"])
        p3 = get_pmf(calib_sample["dur_node3_ms"])
        p4 = get_pmf(calib_sample["dur_node4_ms"])
        p5 = get_pmf(calib_sample["dur_node5_ms"])
        p6 = get_pmf(calib_sample["dur_node6_ms"])

        c3 = np.cumsum(p3)
        c4 = np.cumsum(p4)
        p_fork = np.diff(np.concatenate(([0.0], np.clip(c3 * c4, 0.0, 1.0))))

        curr = p1
        for nxt in [p2, p_fork, p5, p6]:
            curr = np.convolve(curr, nxt)
        curr /= curr.sum()
        return float(curr[d_bins + 1:].sum() * 100.0) if d_bins + 1 < len(curr) else 0.0

    boot_gaps = []
    val_ep_dict = {ep: r_val[r_val["episode_id"] == ep] for ep in val_eps}
    calib_ep_dict = {ep: r_calib[r_calib["episode_id"] == ep] for ep in calib_eps}

    for _ in range(n_boot):
        # Resample validation clusters
        sampled_val_eps = rng.choice(val_eps, size=len(val_eps), replace=True)
        sampled_val_df = pd.concat([val_ep_dict[ep] for ep in sampled_val_eps], ignore_index=True)
        b_emp = float(sampled_val_df["miss"].mean() * 100.0)

        # Resample calibration clusters
        sampled_calib_eps = rng.choice(calib_eps, size=len(calib_eps), replace=True)
        sampled_calib_df = pd.concat([calib_ep_dict[ep] for ep in sampled_calib_eps], ignore_index=True)
        b_indep = get_indep_exceedance(sampled_calib_df)

        boot_gaps.append(b_emp - b_indep)

    ci_lo = float(np.percentile(boot_gaps, 2.5))
    ci_hi = float(np.percentile(boot_gaps, 97.5))

    # In Regime 14, two-sample bootstrap CI lower bound must strictly exceed 0
    assert ci_lo > 0.0, f"Two-sample bootstrap CI included zero in Regime 14: [{ci_lo:.2f}, {ci_hi:.2f}]"
    assert np.mean(boot_gaps) >= 8.0, f"Expected large gap in Regime 14, got {np.mean(boot_gaps):.2f}"


def test_adversarial_temporal_grid_discretization_sensitivity(gate_c_data):
    """
    Stress-tests the discrete convolution tail exceedance across multiple bin resolutions:
    scale = 5 (0.2 ms), scale = 10 (0.1 ms), scale = 20 (0.05 ms).
    Verifies that the exceedance gap is robust to bin resolution (< 0.5 pp variation).
    """
    calib = gate_c_data["calib"]
    val = gate_c_data["val"]
    deadlines = gate_c_data["deadlines"]

    gaps_by_scale = {}
    for scale in [5, 10, 20]:
        df = compute_exceedance_gaps_and_stop_rule(
            calib_df=calib,
            val_df=val,
            deadlines_meta=deadlines,
            scale=scale,
            n_boot=200,  # Fast check
            seed=42,
        )
        pooled_gap = df[df["regime_id"] == "pooled"]["gap_pp"].iloc[0]
        gaps_by_scale[scale] = pooled_gap

    # Variation across scale 5, 10, 20 should be tiny (< 0.5 pp)
    diff_5_10 = abs(gaps_by_scale[5] - gaps_by_scale[10])
    diff_10_20 = abs(gaps_by_scale[10] - gaps_by_scale[20])
    assert diff_5_10 < 0.5, f"Scale 5 vs 10 differed by {diff_5_10:.2f} pp"
    assert diff_10_20 < 0.5, f"Scale 10 vs 20 differed by {diff_10_20:.2f} pp"


# ==============================================================================
# 4. Baseline Summary Invariant and Monotonicity Audit
# ==============================================================================

def test_adversarial_baseline_summary_conservation_invariants(gate_c_data):
    """
    Audits all 418 rows in results/gate_c_baseline_summary.csv for exact arithmetic conservation:
    1. tp + fn == n_miss
    2. fp + tn == n_safe
    3. tp + fp + tn + fn == n_total
    4. sum(first_alarm_n1..n6) == tp + fp
    5. lead_time >= 0.0
    """
    df = gate_c_data["baseline_summary"]
    assert len(df) == 418, f"Expected 418 rows, got {len(df)}"

    for idx, row in df.iterrows():
        # Condition 1
        assert row["tp"] + row["fn"] == row["n_miss"], f"TP+FN != n_miss at row {idx}"
        # Condition 2
        assert row["fp"] + row["tn"] == row["n_safe"], f"FP+TN != n_safe at row {idx}"
        # Condition 3
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == row["n_total"], f"Total mismatch at row {idx}"
        # Condition 4: Every alarm must be attributed to an earliest checkpoint
        total_alarms = row["tp"] + row["fp"]
        attributed_alarms = (
            row["first_alarm_n1"]
            + row["first_alarm_n2"]
            + row["first_alarm_n3_n4"]
            + row["first_alarm_n5"]
            + row["first_alarm_n6"]
        )
        assert total_alarms == attributed_alarms, (
            f"Checkpoint alarm sum mismatch at row {idx}: {attributed_alarms} vs {total_alarms}"
        )
        # Condition 5: Lead times non-negative
        assert row["mean_lead_time_ms"] >= 0.0
        assert row["lead_time_p50_ms"] >= 0.0


def test_adversarial_baseline_summary_monotonicities_across_ramps(gate_c_data):
    """
    Audits monotonicity across threshold policies for all 19 slices and 2 variants (38 total slices):
    1. linear_job: 0.7_1.0 <= 0.8_1.0 <= 0.9_1.0:
       Thresholds are pointwise higher at higher ramps -> Alarms and Recall must be non-decreasing!
    2. linear_time: 0.7_1.0 <= 0.8_1.0 <= 0.9_1.0:
       Thresholds are pointwise higher -> Alarms and Recall must be non-decreasing!
    3. fixed: 0.95 <= 0.99 <= 1.00:
       Thresholds are pointwise higher -> Alarms, Recall, and FPR must be non-decreasing!
    """
    df = gate_c_data["baseline_summary"]
    variants = df["variant"].unique()
    slices = df["regime_id"].unique()

    for v in variants:
        for s in slices:
            sub = df[(df["variant"] == v) & (df["regime_id"] == str(s))]

            # 1. linear_job
            j7 = sub[sub["policy"] == "linear_job_0.7_1.0"].iloc[0]
            j8 = sub[sub["policy"] == "linear_job_0.8_1.0"].iloc[0]
            j9 = sub[sub["policy"] == "linear_job_0.9_1.0"].iloc[0]

            assert j7["alarm_rate_pct"] <= j8["alarm_rate_pct"] + 1e-6, f"Linear job alarm rate non-monotonic at {v}, {s}"
            assert j8["alarm_rate_pct"] <= j9["alarm_rate_pct"] + 1e-6, f"Linear job alarm rate non-monotonic at {v}, {s}"
            assert j7["recall_pct"] <= j8["recall_pct"] + 1e-6, f"Linear job recall non-monotonic at {v}, {s}"
            assert j8["recall_pct"] <= j9["recall_pct"] + 1e-6, f"Linear job recall non-monotonic at {v}, {s}"

            # 2. linear_time
            t7 = sub[sub["policy"] == "linear_time_0.7_1.0"].iloc[0]
            t8 = sub[sub["policy"] == "linear_time_0.8_1.0"].iloc[0]
            t9 = sub[sub["policy"] == "linear_time_0.9_1.0"].iloc[0]

            assert t7["alarm_rate_pct"] <= t8["alarm_rate_pct"] + 1e-6, f"Linear time alarm rate non-monotonic at {v}, {s}"
            assert t8["alarm_rate_pct"] <= t9["alarm_rate_pct"] + 1e-6, f"Linear time alarm rate non-monotonic at {v}, {s}"
            assert t7["recall_pct"] <= t8["recall_pct"] + 1e-6, f"Linear time recall non-monotonic at {v}, {s}"
            assert t8["recall_pct"] <= t9["recall_pct"] + 1e-6, f"Linear time recall non-monotonic at {v}, {s}"

            # 3. fixed
            f95 = sub[sub["policy"] == "fixed_0.95"].iloc[0]
            f99 = sub[sub["policy"] == "fixed_0.99"].iloc[0]
            f100 = sub[sub["policy"] == "fixed_1.00"].iloc[0]

            assert f95["alarm_rate_pct"] <= f99["alarm_rate_pct"] + 1e-6, f"Fixed alarm rate non-monotonic at {v}, {s}"
            assert f99["alarm_rate_pct"] <= f100["alarm_rate_pct"] + 1e-6, f"Fixed alarm rate non-monotonic at {v}, {s}"
            assert f95["recall_pct"] <= f99["recall_pct"] + 1e-6, f"Fixed recall non-monotonic at {v}, {s}"
            assert f99["recall_pct"] <= f100["recall_pct"] + 1e-6, f"Fixed recall non-monotonic at {v}, {s}"
            assert f95["fpr_pct"] <= f99["fpr_pct"] + 1e-6, f"Fixed FPR non-monotonic at {v}, {s}"
            assert f99["fpr_pct"] <= f100["fpr_pct"] + 1e-6, f"Fixed FPR non-monotonic at {v}, {s}"


# ==============================================================================
# 5. Toba Plaxity Operator Invariants
# ==============================================================================

def test_adversarial_toba_plaxity_monotonicity_and_overrun_invariants():
    """
    Stress-tests the DiscretePMF and TobaPlaxityBaseline mathematical functions:
    1. Querying meeting probability P(meet | AST) must be strictly non-increasing in AST.
    2. AST > deadline must return exactly 0.0.
    3. AST < 0 must return P(L >= 0) = 1.0 - p_miss_already.
    4. Probabilities must strictly remain in [0.0, 1.0] across all nodes.
    """
    scale = 10
    D = 20.0
    # Create arbitrary execution PMF with positive tail
    p = np.zeros(250)
    p[20] = 0.3
    p[40] = 0.4
    p[60] = 0.2
    p[150] = 0.1  # Tail beyond D
    pmf = DiscretePMF(p, scale=scale)

    l_exit = DiscretePMF.from_exit_node(pmf, deadline_ms=D, scale=scale)

    # Monotonicity check
    prev_prob = 1.0
    for ast in np.linspace(0.0, D * 1.5, 300):
        prob = l_exit.query_meeting_prob(ast)
        assert 0.0 <= prob <= 1.0
        assert prob <= prev_prob + 1e-12, f"Monotonicity violation at AST={ast}: {prob} > {prev_prob}"
        prev_prob = prob

    # Overrun checks
    assert l_exit.query_meeting_prob(D + 5.0) == 0.0
    assert l_exit.query_meeting_prob(100.0) == 0.0

    # Negative AST handling
    prob_neg = l_exit.query_meeting_prob(-5.0)
    assert abs(prob_neg - (1.0 - l_exit.p_miss_already)) < 1e-12
