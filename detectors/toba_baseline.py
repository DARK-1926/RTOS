r"""
Authoritative implementation of Toba et al. plaxity analysis and deadline miss early detection.

References:
1. Hayate Toba, Sanjoy K. Baruah, and Takuya Azumi.
   "Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds."
   ACM Transactions on Embedded Computing Systems (TECS), Vol. 25, 2026. DOI: 10.1145/3808694
2. Hayate Toba and Takuya Azumi.
   "Deadline Miss Early Detection Method for DAG Tasks Considering Variable Execution Time."
   36th Euromicro Conference on Real-Time Systems (ECRTS 2024), LIPIcs Vol. 298, Article 8.
3. Pre-registered Protocol: research/PREREG.md §9.4.

Mathematical formulation:
- Exit node initialization (Eq. 9): P(L_exit = lambda) = P(X_exit = D - lambda)
- Discrete convolutional subtraction (Eq. 10): P(L_pred = lambda) = sum_x P(X = x) * P(L_succ = lambda + comm + x)
- Independent fork min operator (Eq. 11): P(min(A, B) = lambda) = P(A = lambda) P(lambda <= B) + P(lambda < A) P(B = lambda)
- Multi-successor plaxity recurrence (Eq. 12): L_pred = min_{succ} (L_succ \otimes^- X_pred)
- Plaxity-CDF survival function (Eq. 8): L^c(lambda) = P(L >= lambda)
- Variable probabilistic thresholds (Eq. 13-18):
    w_v = 1 (job count) or w_v = E[X_v] (mean execution time)
    rem_v = min_{succ} rem_succ + w_v
    theta_entry = th_min, theta_exit = th_max
    theta_v = max_{pred} (theta_pred + (th_max - theta_pred) / rem_v)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DiscretePMF:
    """
    Discrete probability mass function binned on a regular temporal grid.
    Grid resolution: Delta t = 1.0 / scale (default scale=10 -> 0.1 ms binning, 10^-4 s).
    Supports exact boundary tracking for guaranteed misses (plaxity < 0).
    """

    def __init__(
        self,
        probs: np.ndarray,
        p_miss_already: float = 0.0,
        scale: int = 10,
    ) -> None:
        """
        probs: 1D array of probabilities for non-negative bins [0, 1, ..., n_bins - 1].
        p_miss_already: Probability mass where plaxity is strictly negative (guaranteed miss).
        scale: Integer scale factor (bins per millisecond).
        """
        self.scale = int(scale)
        self.p_miss_already = max(0.0, float(p_miss_already))
        p = np.asarray(probs, dtype=np.float64)
        total = np.sum(p) + self.p_miss_already
        if total > 0.0 and abs(total - 1.0) > 1e-12:
            p = p / total
            self.p_miss_already = self.p_miss_already / total
        self.probs = np.maximum(0.0, p)
        self.n_bins = len(self.probs)

    @classmethod
    def from_samples(
        cls,
        samples: Union[np.ndarray, List[float], pd.Series],
        max_val: float,
        scale: int = 10,
    ) -> "DiscretePMF":
        """Constructs an empirical discrete PMF from sample durations."""
        arr = np.asarray(samples, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n_bins = max(1, int(np.ceil(max_val * scale)) + 1)
        counts = np.zeros(n_bins, dtype=np.float64)
        if len(arr) == 0:
            counts[0] = 1.0
            return cls(counts, p_miss_already=0.0, scale=scale)

        bin_indices = np.clip(np.round(arr * scale).astype(int), 0, n_bins - 1)
        np.add.at(counts, bin_indices, 1.0)
        probs = counts / float(len(arr))
        return cls(probs, p_miss_already=0.0, scale=scale)

    @classmethod
    def from_exit_node(
        cls,
        execution_pmf: "DiscretePMF",
        deadline_ms: float,
        scale: int = 10,
    ) -> "DiscretePMF":
        """
        Initializes plaxity for the exit node according to TECS 2026 Eq. 9:
        P(L_exit = lambda) = P(X_exit = D - lambda).
        """
        d_bins = max(0, int(round(deadline_ms * scale)))
        l_probs = np.zeros(d_bins + 1, dtype=np.float64)
        p_miss = 0.0

        for x_idx, p_x in enumerate(execution_pmf.probs):
            if p_x <= 0.0:
                continue
            lam_idx = d_bins - x_idx
            if lam_idx >= 0:
                l_probs[lam_idx] += p_x
            else:
                p_miss += p_x

        return cls(l_probs, p_miss_already=p_miss, scale=scale)

    def convolutional_subtraction(
        self,
        execution_pmf: "DiscretePMF",
        comm_ms: float = 0.0,
    ) -> "DiscretePMF":
        """
        TECS 2026 Eq. 10 / ECRTS 2024 Eq. 10:
        P(L_pred = lambda) = sum_x P(X = x) * P(L_succ = lambda + comm + x).
        Implemented via cross-correlation with exact negative slack tracking.
        """
        comm_bins = int(round(comm_ms * self.scale))
        corr = np.correlate(self.probs, execution_pmf.probs, mode="full")
        z = len(execution_pmf.probs) - 1 + comm_bins

        # Existing miss probability from successor propagates backwards additively,
        # plus any draws where execution exceeds available latest finish time (corr[:z])
        p_miss_pred = self.p_miss_already
        if z > 0:
            p_miss_pred += float(np.sum(corr[:min(len(corr), z)]))

        out_len = len(self.probs)
        probs_pred = np.zeros(out_len, dtype=np.float64)
        end = min(len(corr), z + out_len)
        if end > z:
            probs_pred[: end - z] = corr[z:end]

        return DiscretePMF(probs_pred, p_miss_already=p_miss_pred, scale=self.scale)

    @staticmethod
    def independent_fork_min(
        pmf_a: "DiscretePMF",
        pmf_b: "DiscretePMF",
    ) -> "DiscretePMF":
        """
        TECS 2026 Eq. 11 / ECRTS 2024 Eq. 11:
        P(min(A, B) = lambda) = P(A = lambda) P(lambda <= B) + P(lambda < A) P(B = lambda).
        Equivalently, upper cumulative survival:
        P(min(A, B) >= lambda) = P(A >= lambda) * P(B >= lambda).
        """
        max_len = max(len(pmf_a.probs), len(pmf_b.probs))
        a_p = np.pad(pmf_a.probs, (0, max_len - len(pmf_a.probs)))
        b_p = np.pad(pmf_b.probs, (0, max_len - len(pmf_b.probs)))

        # Plaxity-CDF survival arrays P(L >= lambda)
        surv_a = pmf_a.to_plaxity_cdf(length=max_len)
        surv_b = pmf_b.to_plaxity_cdf(length=max_len)
        surv_min = np.clip(surv_a * surv_b, 0.0, 1.0)

        # Reconstruct non-negative discrete PMF from survival difference
        min_probs = np.zeros(max_len, dtype=np.float64)
        if max_len > 1:
            min_probs[:-1] = np.maximum(0.0, surv_min[:-1] - surv_min[1:])
        min_probs[-1] = max(0.0, surv_min[-1])

        p_miss_min = max(0.0, 1.0 - float(surv_min[0]))
        return DiscretePMF(min_probs, p_miss_already=p_miss_min, scale=pmf_a.scale)

    def to_plaxity_cdf(self, length: Optional[int] = None) -> np.ndarray:
        """
        TECS 2026 Eq. 8 / ECRTS 2024 Eq. 8:
        l_pc[v] = sum_{h=v}^u l_p[h] = P(L >= lambda_v).
        """
        target_len = self.n_bins if length is None else max(self.n_bins, length)
        p = self.probs
        if target_len > len(p):
            p = np.pad(p, (0, target_len - len(p)))

        # S[k] = sum_{j=k}^{end} p[j]
        surv = np.cumsum(p[::-1])[::-1]
        return np.clip(surv, 0.0, 1.0)

    def query_meeting_prob(self, ast_ms: float) -> float:
        """
        Queries P(L >= AST) directly from plaxity-CDF.
        If AST <= 0, returns P(L >= 0) = 1.0 - p_miss_already.
        If AST > deadline / max support, returns 0.0.
        """
        if ast_ms <= 0.0:
            ast_ms = 0.0

        bin_idx = int(round(ast_ms * self.scale))
        if bin_idx >= self.n_bins:
            return 0.0

        surv = self.to_plaxity_cdf()
        return float(surv[bin_idx])


class TobaPlaxityBaseline:
    """
    Authoritative Toba et al. Plaxity Baseline Detector for DAG Tasks.
    
    Supports:
    - Discrete convolutional subtraction backward recursion (Eq. 9-12).
    - Arrival-time checking at the start of EVERY node in the DAG: AST(tau_{i, k}).
    - Linear variable thresholds theta(t) (Eq. 13-18) for job count and mean time workloads.
    - Fixed thresholds (0.95, 0.99, 1.00).
    - Matched recall threshold calibration.
    """

    def __init__(
        self,
        scale: int = 10,
        threshold_mode: str = "linear_job",
        th_min: float = 0.80,
        th_max: float = 1.00,
        fixed_threshold: float = 0.95,
    ) -> None:
        """
        scale: Bins per millisecond (10 -> 0.1 ms binning, 10^-4 s).
        threshold_mode:
            - 'linear_job': ACM TECS 2026 Eq. 13, 15-18 with w_v = 1.0
            - 'linear_time': ACM TECS 2026 Eq. 14, 15-18 with w_v = E[X_v]
            - 'linear_time_norm': Normalized linear ramp by remaining mean execution time
            - 'fixed': Static threshold fixed_threshold across all nodes
            - 'matched_recall': Calibrated to achieve target recall on calibration split
        th_min: Minimum threshold at DAG entry node (Eq. 15).
        th_max: Maximum threshold at DAG exit node (Eq. 16).
        fixed_threshold: Constant probability threshold when threshold_mode == 'fixed'.
        """
        self.scale = int(scale)
        self.threshold_mode = threshold_mode
        self.th_min = float(th_min)
        self.th_max = float(th_max)
        self.fixed_threshold = float(fixed_threshold)

        self.node_pmfs: Dict[int, DiscretePMF] = {}
        self.plaxity_pmfs: Dict[int, DiscretePMF] = {}
        self.thresholds: Dict[int, float] = {}
        self.node_means: Dict[int, float] = {}
        self.deadline_ms: float = 0.0
        self.calibrated_threshold: Optional[float] = None

    def fit_from_pmfs(
        self,
        node_pmfs: Dict[int, DiscretePMF],
        node_means: Dict[int, float],
        deadline_ms: float,
        calib_df: Optional[pd.DataFrame] = None,
        target_recall: Optional[float] = None,
    ) -> "TobaPlaxityBaseline":
        """Solves backward plaxity recursion and threshold schedule using provided PMFs."""
        self.deadline_ms = float(deadline_ms)
        self.node_pmfs = node_pmfs
        self.node_means = node_means

        # N6: Exit node plaxity (Eq. 9): L6 = D - X6
        plaxity_6 = DiscretePMF.from_exit_node(
            self.node_pmfs[6], deadline_ms=self.deadline_ms, scale=self.scale
        )

        # N5: Single successor N6 (Eq. 10): L5 = L6 \otimes^- X5
        plaxity_5 = plaxity_6.convolutional_subtraction(self.node_pmfs[5])

        # N3 & N4: Fork branches each with single successor N5 (Eq. 10):
        plaxity_3 = plaxity_5.convolutional_subtraction(self.node_pmfs[3])
        plaxity_4 = plaxity_5.convolutional_subtraction(self.node_pmfs[4])

        # N2: Fork source node with multiple successors {N3, N4} (Eq. 11, 12):
        l2_3 = plaxity_3.convolutional_subtraction(self.node_pmfs[2])
        l2_4 = plaxity_4.convolutional_subtraction(self.node_pmfs[2])
        plaxity_2 = DiscretePMF.independent_fork_min(l2_3, l2_4)

        # N1: Entry node with single successor N2 (Eq. 10):
        plaxity_1 = plaxity_2.convolutional_subtraction(self.node_pmfs[1])

        self.plaxity_pmfs = {
            1: plaxity_1,
            2: plaxity_2,
            3: plaxity_3,
            4: plaxity_4,
            5: plaxity_5,
            6: plaxity_6,
        }

        # Calculate Threshold Schedules
        self._calculate_thresholds()

        # Optional Calibration for Matched Recall
        if self.threshold_mode == "matched_recall" and target_recall is not None and calib_df is not None:
            self._calibrate_matched_recall(calib_df, target_recall=target_recall)

        return self

    def fit(
        self,
        calib_df: pd.DataFrame,
        deadline_ms: float,
        target_recall: Optional[float] = None,
    ) -> "TobaPlaxityBaseline":
        """
        Fits empirical node execution PMFs and solves backward plaxity recursion
        for the 6-node UAV autonomy DAG:
        N1 -> N2 -> Fork { N3 || N4 } -> N5 -> N6
        """
        node_cols = [f"dur_node{i}_ms" if f"dur_node{i}_ms" in calib_df.columns else f"t_n{i}" for i in range(1, 7)]
        max_sample_val = float(max(np.max(calib_df[c].values) for c in node_cols if c in calib_df.columns)) if any(c in calib_df.columns for c in node_cols) else 10.0
        max_d = max(float(deadline_ms) * 1.5, max_sample_val * 1.2, 1.0)
        node_pmfs = {}
        node_means = {}
        for i in range(1, 7):
            col = f"dur_node{i}_ms" if f"dur_node{i}_ms" in calib_df.columns else f"t_n{i}"
            vals = calib_df[col].values
            node_pmfs[i] = DiscretePMF.from_samples(vals, max_val=max_d, scale=self.scale)
            node_means[i] = float(np.mean(vals))

        return self.fit_from_pmfs(
            node_pmfs=node_pmfs,
            node_means=node_means,
            deadline_ms=deadline_ms,
            calib_df=calib_df,
            target_recall=target_recall,
        )

    def _calculate_thresholds(self) -> None:
        """
        Computes per-node detection thresholds theta_v according to Equations 13-18.
        """
        if self.threshold_mode == "fixed":
            self.thresholds = {i: float(self.fixed_threshold) for i in range(1, 7)}
            return

        if self.threshold_mode == "matched_recall":
            # Initialized with fixed default; overridden by _calibrate_matched_recall
            self.thresholds = {i: float(self.fixed_threshold) for i in range(1, 7)}
            return

        # Workload metrics w_v: Eq. 13 (job count) or Eq. 14 (mean execution time)
        w: Dict[int, float] = {}
        for i in range(1, 7):
            w[i] = 1.0 if "job" in self.threshold_mode else max(1e-6, self.node_means[i])

        # Remaining workload rem(v) backwards from exit node N6 (Eq. 17):
        # rem(N6) = w6
        # rem(N5) = rem(N6) + w5
        # rem(N3) = rem(N5) + w3, rem(N4) = rem(N5) + w4
        # rem(N2) = min(rem(N3), rem(N4)) + w2
        # rem(N1) = rem(N2) + w1
        rem: Dict[int, float] = {}
        rem[6] = w[6]
        rem[5] = rem[6] + w[5]
        rem[3] = rem[5] + w[3]
        rem[4] = rem[5] + w[4]
        rem[2] = min(rem[3], rem[4]) + w[2]
        rem[1] = rem[2] + w[1]

        if self.threshold_mode == "linear_time_norm":
            # Normalized linear ramp proportional to elapsed mean workload
            total_work = max(1e-6, rem[1])
            for i in range(1, 6):
                frac_elapsed = 1.0 - (rem[i] / total_work)
                self.thresholds[i] = self.th_min + (self.th_max - self.th_min) * frac_elapsed
            self.thresholds[6] = self.th_max
            return

        # Forward variable threshold propagation: Eq. 15, 18, 16
        self.thresholds[1] = self.th_min  # Eq. 15
        self.thresholds[2] = self.thresholds[1] + (self.th_max - self.thresholds[1]) / rem[2]  # Eq. 18
        self.thresholds[3] = self.thresholds[2] + (self.th_max - self.thresholds[2]) / rem[3]
        self.thresholds[4] = self.thresholds[2] + (self.th_max - self.thresholds[2]) / rem[4]
        self.thresholds[5] = max(
            self.thresholds[3] + (self.th_max - self.thresholds[3]) / rem[5],
            self.thresholds[4] + (self.th_max - self.thresholds[4]) / rem[5],
        )
        self.thresholds[6] = self.th_max  # Eq. 16

    def _calibrate_matched_recall(self, calib_df: pd.DataFrame, target_recall: float = 0.95) -> None:
        """
        Calibrates a uniform threshold theta on calibration traces to achieve target recall.
        """
        miss_mask = calib_df["miss"].values.astype(bool) if "miss" in calib_df.columns else (
            calib_df["s_dag_ms"].values > self.deadline_ms
        )
        if np.sum(miss_mask) == 0:
            self.calibrated_threshold = self.fixed_threshold
            self.thresholds = {i: self.fixed_threshold for i in range(1, 7)}
            return

        # Compute min_p_meet for all missed frames
        min_p_missed = []
        for _, row in calib_df[miss_mask].iterrows():
            eval_res = self._evaluate_raw_row(row)
            min_p_missed.append(eval_res["min_p_meet"])

        # To detect a frame, min_p_meet < theta.
        # Target recall R: fraction of missed frames where min_p_meet < theta is >= R.
        # Hence theta must be strictly greater than quantile(R) of min_p_meet.
        # Specifically, theta = np.percentile(min_p_missed, target_recall * 100).
        calib_theta = float(np.percentile(min_p_missed, target_recall * 100))
        # Ensure floating precision buffer so frames exactly at threshold trigger alarm
        calib_theta = min(1.0, calib_theta + 1e-6)
        self.calibrated_threshold = calib_theta
        self.thresholds = {i: calib_theta for i in range(1, 7)}

    def _evaluate_raw_row(self, row: Union[pd.Series, Dict[str, float]]) -> dict:
        """Helper to extract AST and meeting probabilities from a single row."""
        t1 = float(row.get("dur_node1_ms", row.get("t_n1", 0.0)))
        t2 = float(row.get("dur_node2_ms", row.get("t_n2", 0.0)))
        t3 = float(row.get("dur_node3_ms", row.get("t_n3", 0.0)))
        t4 = float(row.get("dur_node4_ms", row.get("t_n4", 0.0)))
        t5 = float(row.get("dur_node5_ms", row.get("t_n5", 0.0)))

        # Actual Start Time AST(tau_{i, k}) for each node
        ast = {
            1: 0.0,
            2: t1,
            3: t1 + t2,
            4: t1 + t2,
            5: t1 + t2 + max(t3, t4),
            6: t1 + t2 + max(t3, t4) + t5,
        }

        p_meet = {}
        for i in range(1, 7):
            p_meet[i] = self.plaxity_pmfs[i].query_meeting_prob(ast[i])

        min_p = min(p_meet.values())
        return {
            "ast": ast,
            "p_meet": p_meet,
            "min_p_meet": min_p,
            "miss_risk": 1.0 - min_p,
        }

    def evaluate_frame(self, row: Union[pd.Series, Dict[str, float]]) -> dict:
        """
        Evaluates frame arrival times AST(tau_{i, k}) against plaxity-CDFs and thresholds.
        Triggers early detection alarm if P(L_v >= AST_v) < theta_v for any node v.
        """
        raw = self._evaluate_raw_row(row)
        ast = raw["ast"]
        p_meet = raw["p_meet"]

        alarm_nodes = []
        for i in range(1, 7):
            if p_meet[i] < self.thresholds[i]:
                alarm_nodes.append(i)

        is_alarm = len(alarm_nodes) > 0
        earliest_cp = min(alarm_nodes) if is_alarm else None
        lead_time = max(0.0, self.deadline_ms - ast[earliest_cp]) if earliest_cp is not None else 0.0

        return {
            "alarm": is_alarm,
            "earliest_checkpoint": earliest_cp,
            "lead_time_ms": lead_time,
            "p_meet_nodes": p_meet,
            "thresholds": self.thresholds,
            "min_p_meet": raw["min_p_meet"],
            "miss_risk": raw["miss_risk"],
            "ast_nodes": ast,
        }


class TobaDetectorSuite:
    """
    Manager suite supporting both Gate C Baseline Variants:
    - Variant 1 (Pooled Toba): Fitted across pooled calibration traces.
    - Variant 2 (Per-Regime Toba): Fitted strictly per-regime on calibration traces.
    """

    def __init__(
        self,
        scale: int = 10,
        policy: str = "linear_job_0.8_1.0",
        deadlines: Optional[Dict[int, float]] = None,
    ) -> None:
        self.scale = int(scale)
        self.policy = policy
        self.deadlines = deadlines or {}
        self.pooled_detector: Optional[TobaPlaxityBaseline] = None
        self.regime_detectors: Dict[int, TobaPlaxityBaseline] = {}

    @staticmethod
    def parse_policy(policy: str) -> dict:
        """Parses policy string into TobaPlaxityBaseline configuration arguments."""
        if policy.startswith("linear_job_"):
            parts = policy.split("_")
            return {
                "threshold_mode": "linear_job",
                "th_min": float(parts[2]),
                "th_max": float(parts[3]),
            }
        elif policy.startswith("linear_time_norm_"):
            parts = policy.split("_")
            return {
                "threshold_mode": "linear_time_norm",
                "th_min": float(parts[3]),
                "th_max": float(parts[4]),
            }
        elif policy.startswith("linear_time_"):
            parts = policy.split("_")
            return {
                "threshold_mode": "linear_time",
                "th_min": float(parts[2]),
                "th_max": float(parts[3]),
            }
        elif policy.startswith("fixed_"):
            parts = policy.split("_")
            return {
                "threshold_mode": "fixed",
                "fixed_threshold": float(parts[1]),
            }
        elif policy.startswith("matched_recall_"):
            parts = policy.split("_")
            recall_val = float(parts[2]) / 100.0 if float(parts[2]) > 1.0 else float(parts[2])
            return {
                "threshold_mode": "matched_recall",
                "target_recall": recall_val,
            }
        else:
            raise ValueError(f"Unknown policy format: {policy}")

    def fit(self, calib_df: pd.DataFrame, deadlines: Dict[int, float]) -> "TobaDetectorSuite":
        """Fits both Variant 1 (Pooled) and Variant 2 (Per-regime) models."""
        self.deadlines = deadlines
        cfg = self.parse_policy(self.policy)
        target_rec = cfg.pop("target_recall", None)

        # 1. Fit Variant 1: Pooled Calibration Fit
        # Compute pooled marginal execution PMFs across all calibration traces
        node_cols = [f"dur_node{i}_ms" if f"dur_node{i}_ms" in calib_df.columns else f"t_n{i}" for i in range(1, 7)]
        max_sample_val = float(max(np.max(calib_df[c].values) for c in node_cols if c in calib_df.columns)) if any(c in calib_df.columns for c in node_cols) else 10.0
        max_d_pooled = max(max(deadlines.values()) * 1.5, max_sample_val * 1.2, 1.0)
        pooled_pmfs = {}
        pooled_means = {}
        for i in range(1, 7):
            col = f"dur_node{i}_ms" if f"dur_node{i}_ms" in calib_df.columns else f"t_n{i}"
            vals = calib_df[col].values
            pooled_pmfs[i] = DiscretePMF.from_samples(vals, max_val=max_d_pooled, scale=self.scale)
            pooled_means[i] = float(np.mean(vals))

        self.pooled_detectors: Dict[int, TobaPlaxityBaseline] = {}
        for reg_id, d_reg in deadlines.items():
            p_det = TobaPlaxityBaseline(scale=self.scale, **cfg)
            p_det.fit_from_pmfs(
                node_pmfs=pooled_pmfs,
                node_means=pooled_means,
                deadline_ms=d_reg,
                calib_df=calib_df,
                target_recall=target_rec,
            )
            self.pooled_detectors[reg_id] = p_det

        # 2. Fit Variant 2: Per-Regime Calibration Fit
        self.regime_detectors = {}
        for reg_id, d_reg in deadlines.items():
            reg_calib = calib_df[calib_df["regime_id"] == reg_id]
            if len(reg_calib) == 0:
                continue
            det = TobaPlaxityBaseline(scale=self.scale, **cfg)
            det.fit(reg_calib, deadline_ms=d_reg, target_recall=target_rec)
            self.regime_detectors[reg_id] = det

        return self

    def evaluate_variant(
        self,
        val_df: pd.DataFrame,
        variant: str = "per_regime",
    ) -> List[dict]:
        """
        Evaluates traces in val_df using the specified baseline variant:
        - 'per_regime': Uses the detector fit strictly on that frame's regime (Variant 2).
        - 'pooled': Uses the pooled calibration fit detector evaluated with regime deadline (Variant 1).
        """
        results = []
        for idx, row in val_df.iterrows():
            reg_id = int(row["regime_id"])
            if variant == "per_regime":
                detector = self.regime_detectors[reg_id]
            elif variant in ("pooled", "pooled_toba"):
                detector = self.pooled_detectors[reg_id]
            else:
                raise ValueError(f"Unknown variant: {variant}")

            res = detector.evaluate_frame(row)
            actual_miss = bool(row["miss"]) if "miss" in row else (
                float(row.get("s_dag_ms", row.get("t_s_dag", 0.0))) > float(row["deadline_ms"])
            )
            res["actual_miss"] = actual_miss
            res["regime_id"] = reg_id
            res["episode_id"] = int(row["episode_id"])
            res["frame_id"] = int(row["frame_id"])
            results.append(res)

        return results
