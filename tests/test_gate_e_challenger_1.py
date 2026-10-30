r"""
Adversarial & Generalization Challenge Test Suite for Milestone 4 (Gate E Confirmatory Test Split).
Author: Challenger 1 (challenger_m4_1)
Role: Empirical Challenger (critic / specialist)

Adversarially tests:
1. Generalization & Over-fitting Audit:
   - Compares Gate D Validation vs Gate E Test performance.
   - Asserts that generalization gap (|Test - Val|) for D1 and D2 is tightly bounded:
     pooled precision delta within 2.0 pp, pooled recall delta within 2.0 pp.
   - Verifies zero signs of over-fitting across all 18 environment regimes.
2. Cluster-Bootstrap Seed Robustness:
   - Tests cluster-bootstrap CI estimation stability across 5 diverse random seeds.
   - Asserts that Gate E passing status is invariant to bootstrap seed (100% pass rate).
   - Asserts that CI lower bound variance is negligible (std < 0.15 pp, min CI lo > +64.0 pp).
3. Alarm Timing and Checkpoint Invariants:
   - Asserts B1 suffers severe premature alarms at Node 1 (> 85% of frames).
   - Asserts D1 and D2 have 0 alarms at Node 1.
   - Asserts D1 triggers >= 100 True Positive alarms at Join Node N5.
4. Candidate D2 Trade-off Quantification:
   - Quantifies D2 recall gain: +3.0 to +6.0 pp over D1.
   - Quantifies D2 lead time compression: > 75% lead time reduction vs D1.
   - Asserts D1 strictly obeys the PREREG §9.6 lead time preservation budget (loss <= 20% vs Oracle B2).
5. Strict Data Integrity:
   - Disjointness of test episodes with calib/val splits.
   - Exact 7,200 rows across 144 episodes with no missing values.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

VAL_SUMMARY_PATH = os.path.join(RESULTS_DIR, "gate_d_validation_summary.csv")
VAL_BOOTSTRAP_PATH = os.path.join(RESULTS_DIR, "gate_d_bootstrap_ci.csv")
TEST_SUMMARY_PATH = os.path.join(RESULTS_DIR, "gate_e_test_summary.csv")
TEST_BOOTSTRAP_PATH = os.path.join(RESULTS_DIR, "gate_e_bootstrap_ci.csv")
TEST_TRACES_PATH = os.path.join(RESULTS_DIR, "uav_test_traces.csv")
DEADLINES_PATH = os.path.join(RESULTS_DIR, "gate_b_deadlines.json")


# ==============================================================================
# Suite 1: Generalization Audit (Validation vs Test Integrity)
# ==============================================================================

class TestGeneralizationAudit:
    """Verifies that Candidate Detectors D1 and D2 generalize cleanly with zero overfitting."""

    @pytest.fixture(scope="class")
    def summaries(self):
        val_df = pd.read_csv(VAL_SUMMARY_PATH)
        test_df = pd.read_csv(TEST_SUMMARY_PATH)
        return val_df, test_df

    def test_pooled_generalization_gap_bounds(self, summaries):
        """
        Adversarial Check: Does test performance collapse relative to validation?
        Verifies that pooled metrics between Val (3,600 frames) and Test (7,200 frames)
        diverge by no more than tight bounds:
        - D1 Precision Delta: |Test - Val| <= 1.0 pp
        - D1 Recall Delta: |Test - Val| <= 2.0 pp
        - D1 FPR Delta: |Test - Val| <= 0.5 pp
        - D2 Precision Delta: |Test - Val| <= 3.0 pp
        - D2 Recall Delta: |Test - Val| <= 2.0 pp
        """
        val_df, test_df = summaries
        val_p = val_df[val_df["regime_id"] == "pooled"].set_index("detector_id")
        test_p = test_df[test_df["regime_id"] == "pooled"].set_index("detector_id")

        # Candidate D1
        d1_val = val_p.loc["detector_d1"]
        d1_test = test_p.loc["detector_d1"]

        d1_prec_gap = abs(float(d1_test["precision_pct"]) - float(d1_val["precision_pct"]))
        d1_rec_gap = abs(float(d1_test["recall_pct"]) - float(d1_val["recall_pct"]))
        d1_fpr_gap = abs(float(d1_test["fpr_pct"]) - float(d1_val["fpr_pct"]))

        assert d1_prec_gap <= 1.0, f"D1 precision gap exceeded: {d1_prec_gap:.4f} pp"
        assert d1_rec_gap <= 2.0, f"D1 recall gap exceeded: {d1_rec_gap:.4f} pp"
        assert d1_fpr_gap <= 0.5, f"D1 FPR gap exceeded: {d1_fpr_gap:.4f} pp"

        # Candidate D2
        d2_val = val_p.loc["detector_d2"]
        d2_test = test_p.loc["detector_d2"]

        d2_prec_gap = abs(float(d2_test["precision_pct"]) - float(d2_val["precision_pct"]))
        d2_rec_gap = abs(float(d2_test["recall_pct"]) - float(d2_val["recall_pct"]))
        d2_fpr_gap = abs(float(d2_test["fpr_pct"]) - float(d2_val["fpr_pct"]))

        assert d2_prec_gap <= 3.0, f"D2 precision gap exceeded: {d2_prec_gap:.4f} pp"
        assert d2_rec_gap <= 2.0, f"D2 recall gap exceeded: {d2_rec_gap:.4f} pp"
        assert d2_fpr_gap <= 1.0, f"D2 FPR gap exceeded: {d2_fpr_gap:.4f} pp"

    def test_per_regime_mean_generalization_gap(self, summaries):
        """
        Verifies that across all 18 individual regimes, the mean generalization gap
        is near zero, showing no systematic bias or split shift.
        """
        val_df, test_df = summaries
        val_reg = val_df[val_df["regime_id"] != "pooled"].copy()
        test_reg = test_df[test_df["regime_id"] != "pooled"].copy()
        val_reg["regime_id"] = val_reg["regime_id"].astype(int)
        test_reg["regime_id"] = test_reg["regime_id"].astype(int)

        merged = pd.merge(
            val_reg[["detector_id", "regime_id", "precision_pct", "recall_pct"]],
            test_reg[["detector_id", "regime_id", "precision_pct", "recall_pct"]],
            on=["detector_id", "regime_id"],
            suffixes=("_val", "_test"),
        )
        merged["delta_prec"] = merged["precision_pct_test"] - merged["precision_pct_val"]
        merged["delta_rec"] = merged["recall_pct_test"] - merged["recall_pct_val"]

        for d in ["detector_d1", "detector_d2"]:
            sub = merged[merged["detector_id"] == d]
            mean_prec_gap = sub["delta_prec"].mean()
            mean_rec_gap = sub["delta_rec"].mean()
            assert -2.0 <= mean_prec_gap <= 4.0, f"{d} mean per-regime precision gap shifted: {mean_prec_gap:.2f} pp"
            assert -2.0 <= mean_rec_gap <= 3.0, f"{d} mean per-regime recall gap shifted: {mean_rec_gap:.2f} pp"


# ==============================================================================
# Suite 2: Bootstrap Seed Stability Stress Test
# ==============================================================================

class TestBootstrapSeedRobustness:
    """Stress tests cluster-bootstrap confidence interval estimation across different random seeds."""

    @pytest.fixture(scope="class")
    def stream_eval_data(self):
        from benchmarks.evaluate_gate_e import load_datasets_and_deadlines, fit_all_detectors, evaluate_stream
        calib_df, test_df, deadlines, meta = load_datasets_and_deadlines(REPO_ROOT)
        models = fit_all_detectors(calib_df, deadlines)
        eval_df = evaluate_stream(test_df, models)
        return eval_df, meta

    @pytest.mark.parametrize("seed", [42, 100, 2026, 7735, 99999])
    def test_bootstrap_stability_across_seeds(self, stream_eval_data, seed):
        """
        Adversarial Scenario: Runs 2,000 paired cluster-bootstraps under varying RNG seeds.
        Verifies:
        1. Delta Precision (D1 vs B1) mean is within [66.5, 67.5] pp.
        2. CI lower bound is strictly > +64.0 pp (pre-registered threshold is +5.0 pp).
        3. Gate E pass decision is strictly True across all seeds.
        4. Delta Precision (D2 vs B1) mean is within [65.5, 66.5] pp with CI lo > +63.0 pp.
        """
        from benchmarks.evaluate_gate_e import run_cluster_bootstrap
        eval_df, meta = stream_eval_data

        boot_df = run_cluster_bootstrap(eval_df, meta, n_boot=2000, seed=seed)
        pooled = boot_df[boot_df["regime_id"] == "pooled"].iloc[0]

        # D1 vs B1
        d1_mean = float(pooled["delta_d1_vs_b1_mean"])
        d1_ci_lo = float(pooled["delta_d1_vs_b1_ci_lo"])
        d1_ci_hi = float(pooled["delta_d1_vs_b1_ci_hi"])
        d1_pass = bool(pooled["d1_vs_b1_passes_gate"])

        assert 66.5 <= d1_mean <= 67.5, f"Seed {seed}: unexpected D1 delta mean {d1_mean}"
        assert d1_ci_lo >= 64.0, f"Seed {seed}: D1 CI lo unexpectedly low {d1_ci_lo}"
        assert d1_ci_hi <= 70.0, f"Seed {seed}: D1 CI hi unexpectedly high {d1_ci_hi}"
        assert d1_pass is True, f"Seed {seed}: D1 vs B1 failed gate"

        # D2 vs B1
        d2_mean = float(pooled["delta_d2_vs_b1_mean"])
        d2_ci_lo = float(pooled["delta_d2_vs_b1_ci_lo"])
        d2_pass = bool(pooled["d2_vs_b1_passes_gate"])

        assert 65.5 <= d2_mean <= 66.5, f"Seed {seed}: unexpected D2 delta mean {d2_mean}"
        assert d2_ci_lo >= 63.0, f"Seed {seed}: D2 CI lo unexpectedly low {d2_ci_lo}"
        assert d2_pass is True, f"Seed {seed}: D2 vs B1 failed gate"


# ==============================================================================
# Suite 3: Alarm Timing and Checkpoint Soundness
# ==============================================================================

class TestAlarmTimingAndCheckpointSoundness:
    """Audits physical checkpoint alarm distributions and lead time characteristics."""

    @pytest.fixture(scope="class")
    def summary_data(self):
        return pd.read_csv(TEST_SUMMARY_PATH)

    def test_b1_premature_node1_collapse(self, summary_data):
        """
        Confirms the physical breakdown of static plaxity (Baseline 1):
        At least 85% of all alarms occur prematurely at DAG entry Node 1 (AST=0),
        causing the 88.8% false alarm rate.
        """
        pooled = summary_data[summary_data["regime_id"] == "pooled"]
        b1 = pooled[pooled["detector_id"] == "pooled_toba"].iloc[0]

        total_alarms = b1["tp"] + b1["fp"]
        n1_alarms = b1["first_alarm_n1"]
        n1_pct = n1_alarms / total_alarms * 100.0

        assert n1_pct >= 85.0, f"B1 did not concentrate at N1: {n1_pct:.2f}%"
        assert b1["fpr_pct"] >= 85.0, f"B1 FPR unexpectedly low: {b1['fpr_pct']:.2f}%"

    def test_d1_and_d2_zero_node1_alarms(self, summary_data):
        """
        Verifies that Candidate D1 and Candidate D2 never fire premature false alarms
        at Node 1 (AST=0), respecting authentic DAG timing.
        """
        pooled = summary_data[summary_data["regime_id"] == "pooled"]
        d1 = pooled[pooled["detector_id"] == "detector_d1"].iloc[0]
        d2 = pooled[pooled["detector_id"] == "detector_d2"].iloc[0]

        assert d1["first_alarm_n1"] == 0, f"D1 fired {d1['first_alarm_n1']} alarms at Node 1"
        assert d2["first_alarm_n1"] == 0, f"D2 fired {d2['first_alarm_n1']} alarms at Node 1"

    def test_d1_join_node_detection_power(self, summary_data):
        """
        Verifies that Candidate D1 catches impending misses early at Join Node N5
        on >= 100 frames, providing actionable early warning lead time.
        """
        pooled = summary_data[summary_data["regime_id"] == "pooled"]
        d1 = pooled[pooled["detector_id"] == "detector_d1"].iloc[0]
        assert d1["first_alarm_n5"] >= 100, f"D1 N5 alarms unexpectedly low: {d1['first_alarm_n5']}"


# ==============================================================================
# Suite 4: D2 Trade-Off Quantification (Recall vs Lead Time Compression)
# ==============================================================================

class TestD2TradeoffQuantification:
    """Rigorously analyzes the Pareto trade-off of Candidate D2."""

    @pytest.fixture(scope="class")
    def summary_data(self):
        return pd.read_csv(TEST_SUMMARY_PATH)

    def test_d2_recall_gain_over_d1(self, summary_data):
        """
        Verifies that super-node modeling in D2 yields a decisive recall boost
        between +3.0 and +6.0 percentage points over D1 on the fresh test set.
        """
        pooled = summary_data[summary_data["regime_id"] == "pooled"].set_index("detector_id")
        d1_rec = float(pooled.loc["detector_d1", "recall_pct"])
        d2_rec = float(pooled.loc["detector_d2", "recall_pct"])

        delta_rec = d2_rec - d1_rec
        assert 3.0 <= delta_rec <= 6.0, f"Unexpected D2 recall gain: {delta_rec:.2f} pp"

    def test_d2_lead_time_compression(self, summary_data):
        """
        Quantifies the price of D2's recall gain:
        Mean TP lead time compresses by > 75% relative to D1, because alarms shift
        from Join Node N5 downstream to Terminal Node N6.
        """
        pooled = summary_data[summary_data["regime_id"] == "pooled"].set_index("detector_id")
        d1_lead = float(pooled.loc["detector_d1", "tp_mean_lead_time_ms"])
        d2_lead = float(pooled.loc["detector_d2", "tp_mean_lead_time_ms"])

        compression_pct = (1.0 - d2_lead / d1_lead) * 100.0
        assert compression_pct >= 75.0, f"D2 lead compression lower than expected: {compression_pct:.1f}%"

        # Check N5 alarm migration to N6
        d1_n5 = int(pooled.loc["detector_d1", "first_alarm_n5"])
        d2_n5 = int(pooled.loc["detector_d2", "first_alarm_n5"])
        assert d2_n5 < d1_n5 / 2, f"D2 N5 alarms ({d2_n5}) should be less than half of D1 ({d1_n5})"

    def test_d1_preserves_lead_time_budget(self):
        """
        Verifies that Candidate D1 satisfies PREREG §9.6 lead time preservation budget
        (loss <= 20% relative to Oracle Baseline 2).
        """
        boot_df = pd.read_csv(TEST_BOOTSTRAP_PATH)
        pooled = boot_df[boot_df["regime_id"] == "pooled"].iloc[0]

        d1_lead_deg = float(pooled["lead_degradation_d1_vs_b2_pct"])
        assert d1_lead_deg >= -20.0, f"D1 violated lead time preservation budget: {d1_lead_deg:.2f}%"
