r"""
Gate D Candidate Early Deadline Miss Detectors for Non-Stationary Environments.

Implements:
1. ObservableStateEstimator:
   - Tracks online moving median of upstream node latencies (t1, t2) and recent end-to-end latency (S_DAG).
   - Classifies active regime via normalized Euclidean distance to calibration centroids.
   - Resets state at episode boundaries.
2. SuperNodePlaxityBaseline:
   - Extends Toba et al. plaxity framework by replacing parallel fork branches N3 || N4
     with an empirical composite super-node N_fork = max(X3, X4).
   - Solves the 5-node sequential super-DAG (N1 -> N2 -> N_fork -> N5 -> N6) via standard
     backward convolutional subtraction (ACM TECS 2026 Eq. 10).
   - Completely bypasses Eq. 11, eliminating join-node common-term reuse distortion and
     internalizing multi-core memory contention coupling.
3. RegimeConditionedDetectorD1:
   - Evaluates frame arrival times against plaxity models conditioned on the observable state.
4. SuperNodePlaxityDetectorD2:
   - Combines D1's observable-state regime conditioning with SuperNodePlaxityBaseline.

References:
- ACM Transactions on Embedded Computing Systems (TECS) 2026, DOI: 10.1145/3808694 (Section 6/7).
- Pre-registered Protocol: research/PREREG.md §9.5.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.detectors.toba_baseline import (
    DiscretePMF,
    TobaPlaxityBaseline,
    TobaDetectorSuite,
)


class ObservableStateEstimator:
    """
    Online observable-state estimator for environmental regime tracking.

    Maintains a rolling window (default W=5) of recent completed frame measurements
    (dur_node1_ms, dur_node2_ms, s_dag_ms) within an episode.
    Maps current frame observable feature vector z_k to calibration regime centroids
    using normalized Euclidean distance.
    """

    def __init__(
        self,
        window: int = 5,
        scale_factors: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        """
        window: Number of recent frames in episode for rolling median (W).
        scale_factors: Feature normalization scales [scale_t1, scale_t2, scale_sdag].
        """
        self.window = int(window)
        self.scale_factors = (
            np.asarray(scale_factors, dtype=np.float64)
            if scale_factors is not None
            else np.array([2.0, 0.3, 20.0], dtype=np.float64)
        )
        self.calib_centroids: Dict[int, np.ndarray] = {}
        self.global_centroid: Optional[np.ndarray] = None

        # Online per-episode history
        self.hist_t1: List[float] = []
        self.hist_t2: List[float] = []
        self.hist_sdag: List[float] = []

    def fit(self, calib_df: pd.DataFrame) -> "ObservableStateEstimator":
        """
        Computes per-regime centroids and global median centroid from calibration traces.
        """
        col_t1 = "dur_node1_ms" if "dur_node1_ms" in calib_df.columns else "t_n1"
        col_t2 = "dur_node2_ms" if "dur_node2_ms" in calib_df.columns else "t_n2"
        col_sdag = "s_dag_ms" if "s_dag_ms" in calib_df.columns else "t_s_dag"

        self.calib_centroids = {}
        regimes = sorted(calib_df["regime_id"].unique())
        for r in regimes:
            sub = calib_df[calib_df["regime_id"] == r]
            self.calib_centroids[int(r)] = np.array([
                float(np.median(sub[col_t1])),
                float(np.median(sub[col_t2])),
                float(np.median(sub[col_sdag])),
            ], dtype=np.float64)

        self.global_centroid = np.array([
            float(np.median(calib_df[col_t1])),
            float(np.median(calib_df[col_t2])),
            float(np.median(calib_df[col_sdag])),
        ], dtype=np.float64)

        return self

    def reset(self) -> None:
        """Resets online history. Must be called at the start of each episode."""
        self.hist_t1.clear()
        self.hist_t2.clear()
        self.hist_sdag.clear()

    def get_observable_vector(self, curr_t1: float, curr_t2: float) -> np.ndarray:
        """
        Constructs the observable state vector z_k for the current frame before DAG completion.
        Combines recent moving median with current frame upstream latencies t1, t2.
        """
        if self.global_centroid is None:
            raise RuntimeError("ObservableStateEstimator must be fit() before getting observable vector.")

        if len(self.hist_t1) == 0:
            # First frame of episode: use current t1, t2 and global median for end-to-end latency
            return np.array([float(curr_t1), float(curr_t2), float(self.global_centroid[2])], dtype=np.float64)

        w_t1 = self.hist_t1[-self.window:] + [float(curr_t1)]
        w_t2 = self.hist_t2[-self.window:] + [float(curr_t2)]
        w_sdag = self.hist_sdag[-self.window:]

        return np.array([
            float(np.median(w_t1)),
            float(np.median(w_t2)),
            float(np.median(w_sdag)),
        ], dtype=np.float64)

    def predict_regime(self, curr_t1: float, curr_t2: float) -> int:
        """
        Classifies current frame into nearest regime centroid using normalized Euclidean distance.
        """
        z = self.get_observable_vector(curr_t1, curr_t2)
        dists: Dict[int, float] = {}
        for r, centroid in self.calib_centroids.items():
            diff_norm = (z - centroid) / self.scale_factors
            dists[r] = float(np.sum(diff_norm ** 2))

        return min(dists, key=dists.get)

    def update(self, t1: float, t2: float, s_dag: float) -> None:
        """Updates rolling history with completed frame measurements."""
        self.hist_t1.append(float(t1))
        self.hist_t2.append(float(t2))
        self.hist_sdag.append(float(s_dag))


class SuperNodePlaxityBaseline:
    """
    Authoritative Super-Node Plaxity Baseline for DAG Tasks.

    Collapses parallel fork branches N3 || N4 into a single composite super-node
    N_fork with empirical distribution X_fork = max(X3, X4) (or wall-clock fork duration).
    Solves backward plaxity recursion across the resulting 5-node super-DAG:
        N1 -> N2 -> N_fork -> N5 -> N6
    using purely discrete convolutional subtraction (TECS 2026 Eq. 10).
    Completely bypasses Eq. 11 (independent fork min operator).
    """

    def __init__(
        self,
        scale: int = 10,
        threshold_mode: str = "linear_job",
        th_min: float = 0.80,
        th_max: float = 1.00,
        fixed_threshold: float = 0.95,
        fork_mode: str = "wall",
    ) -> None:
        """
        scale: Temporal discretization bins per millisecond (default 10 -> 0.1 ms).
        threshold_mode: 'linear_job', 'linear_time', 'linear_time_norm', 'fixed', 'matched_recall'.
        th_min: Minimum threshold at DAG entry node (Eq. 15).
        th_max: Maximum threshold at DAG exit node (Eq. 16).
        fixed_threshold: Constant probability threshold when threshold_mode == 'fixed'.
        fork_mode: 'wall' (uses dur_fork_wall_ms if present) or 'max' (uses max(X3, X4)).
        """
        self.scale = int(scale)
        self.threshold_mode = threshold_mode
        self.th_min = float(th_min)
        self.th_max = float(th_max)
        self.fixed_threshold = float(fixed_threshold)
        self.fork_mode = fork_mode

        self.node_pmfs: Dict[Union[int, str], DiscretePMF] = {}
        self.plaxity_pmfs: Dict[int, DiscretePMF] = {}
        self.thresholds: Dict[int, float] = {}
        self.node_means: Dict[Union[int, str], float] = {}
        self.deadline_ms: float = 0.0
        self.calibrated_threshold: Optional[float] = None

    def fit_from_pmfs(
        self,
        pmf_1: DiscretePMF,
        pmf_2: DiscretePMF,
        pmf_fork: DiscretePMF,
        pmf_5: DiscretePMF,
        pmf_6: DiscretePMF,
        mean_1: float,
        mean_2: float,
        mean_fork: float,
        mean_5: float,
        mean_6: float,
        deadline_ms: float,
        calib_df: Optional[pd.DataFrame] = None,
        target_recall: Optional[float] = None,
    ) -> "SuperNodePlaxityBaseline":
        """
        Solves 5-stage backward plaxity recursion and computes threshold schedule.
        """
        self.deadline_ms = float(deadline_ms)
        self.node_pmfs = {
            1: pmf_1,
            2: pmf_2,
            "fork": pmf_fork,
            5: pmf_5,
            6: pmf_6,
        }
        self.node_means = {
            1: float(mean_1),
            2: float(mean_2),
            "fork": float(mean_fork),
            5: float(mean_5),
            6: float(mean_6),
        }

        # Stage 5: Exit node plaxity (Eq. 9): L6 = D - X6
        l6 = DiscretePMF.from_exit_node(
            pmf_6, deadline_ms=self.deadline_ms, scale=self.scale
        )

        # Stage 4: Join node N5 (Eq. 10): L5 = L6 \ominus X5
        l5 = l6.convolutional_subtraction(pmf_5)

        # Stage 3: Super-Node N_fork (Eq. 10): L_fork = L5 \ominus X_fork
        l_fork = l5.convolutional_subtraction(pmf_fork)

        # Stage 2: Node N2 (Eq. 10): L2 = L_fork \ominus X2
        l2 = l_fork.convolutional_subtraction(pmf_2)

        # Stage 1: Entry node N1 (Eq. 10): L1 = L2 \ominus X1
        l1 = l2.convolutional_subtraction(pmf_1)

        # Map plaxities to the 6 physical query checkpoints:
        # Checkpoint 3 (N3 start) and Checkpoint 4 (N4 start) both query L_fork
        self.plaxity_pmfs = {
            1: l1,
            2: l2,
            3: l_fork,
            4: l_fork,
            5: l5,
            6: l6,
        }

        # Calculate threshold schedules
        self._calculate_thresholds()

        # Matched recall calibration if requested
        if self.threshold_mode == "matched_recall" and target_recall is not None and calib_df is not None:
            self._calibrate_matched_recall(calib_df, target_recall=target_recall)

        return self

    def fit(
        self,
        calib_df: pd.DataFrame,
        deadline_ms: float,
        target_recall: Optional[float] = None,
    ) -> "SuperNodePlaxityBaseline":
        """
        Fits empirical discrete PMFs and solves backward recursion on calibration traces.
        """
        col_t1 = "dur_node1_ms" if "dur_node1_ms" in calib_df.columns else "t_n1"
        col_t2 = "dur_node2_ms" if "dur_node2_ms" in calib_df.columns else "t_n2"
        col_t3 = "dur_node3_ms" if "dur_node3_ms" in calib_df.columns else "t_n3"
        col_t4 = "dur_node4_ms" if "dur_node4_ms" in calib_df.columns else "t_n4"
        col_t5 = "dur_node5_ms" if "dur_node5_ms" in calib_df.columns else "t_n5"
        col_t6 = "dur_node6_ms" if "dur_node6_ms" in calib_df.columns else "t_n6"

        vals1 = calib_df[col_t1].values
        vals2 = calib_df[col_t2].values
        vals5 = calib_df[col_t5].values
        vals6 = calib_df[col_t6].values

        if self.fork_mode == "wall" and "dur_fork_wall_ms" in calib_df.columns:
            vals_fork = calib_df["dur_fork_wall_ms"].values
        elif self.fork_mode == "wall" and "t_fork" in calib_df.columns:
            vals_fork = calib_df["t_fork"].values
        else:
            vals_fork = np.maximum(calib_df[col_t3].values, calib_df[col_t4].values)

        max_sample_val = float(max(
            np.max(vals1), np.max(vals2), np.max(vals_fork), np.max(vals5), np.max(vals6)
        ))
        max_d = max(float(deadline_ms) * 1.5, max_sample_val * 1.2, 10.0)

        pmf1 = DiscretePMF.from_samples(vals1, max_val=max_d, scale=self.scale)
        pmf2 = DiscretePMF.from_samples(vals2, max_val=max_d, scale=self.scale)
        pmf_fork = DiscretePMF.from_samples(vals_fork, max_val=max_d, scale=self.scale)
        pmf5 = DiscretePMF.from_samples(vals5, max_val=max_d, scale=self.scale)
        pmf6 = DiscretePMF.from_samples(vals6, max_val=max_d, scale=self.scale)

        return self.fit_from_pmfs(
            pmf_1=pmf1,
            pmf_2=pmf2,
            pmf_fork=pmf_fork,
            pmf_5=pmf5,
            pmf_6=pmf6,
            mean_1=float(np.mean(vals1)),
            mean_2=float(np.mean(vals2)),
            mean_fork=float(np.mean(vals_fork)),
            mean_5=float(np.mean(vals5)),
            mean_6=float(np.mean(vals6)),
            deadline_ms=deadline_ms,
            calib_df=calib_df,
            target_recall=target_recall,
        )

    def _calculate_thresholds(self) -> None:
        """
        Computes detection thresholds theta across the 5 super-DAG stages.
        """
        if self.threshold_mode == "fixed":
            self.thresholds = {i: float(self.fixed_threshold) for i in range(1, 7)}
            return

        if self.threshold_mode == "matched_recall":
            self.thresholds = {i: float(self.fixed_threshold) for i in range(1, 7)}
            return

        # Workload metrics w_v for the 5 stages: N1, N2, N_fork, N5, N6
        w: Dict[Union[int, str], float] = {}
        for k in [1, 2, "fork", 5, 6]:
            w[k] = 1.0 if "job" in self.threshold_mode else max(1e-6, self.node_means[k])

        # Remaining workload backwards from exit node N6:
        rem_6 = w[6]
        rem_5 = rem_6 + w[5]
        rem_fork = rem_5 + w["fork"]
        rem_2 = rem_fork + w[2]
        rem_1 = rem_2 + w[1]

        if self.threshold_mode == "linear_time_norm":
            total_work = max(1e-6, rem_1)
            self.thresholds[1] = self.th_min
            self.thresholds[2] = self.th_min + (self.th_max - self.th_min) * (1.0 - rem_2 / total_work)
            th_fork = self.th_min + (self.th_max - self.th_min) * (1.0 - rem_fork / total_work)
            self.thresholds[3] = th_fork
            self.thresholds[4] = th_fork
            self.thresholds[5] = self.th_min + (self.th_max - self.th_min) * (1.0 - rem_5 / total_work)
            self.thresholds[6] = self.th_max
            return

        # Forward variable threshold recurrence (Eq. 15, 18, 16):
        th_1 = self.th_min
        th_2 = th_1 + (self.th_max - th_1) / rem_2
        th_fork = th_2 + (self.th_max - th_2) / rem_fork
        th_5 = th_fork + (self.th_max - th_fork) / rem_5
        th_6 = self.th_max

        self.thresholds = {
            1: th_1,
            2: th_2,
            3: th_fork,
            4: th_fork,
            5: th_5,
            6: th_6,
        }

    def _calibrate_matched_recall(
        self,
        calib_df: pd.DataFrame,
        target_recall: float = 0.95,
    ) -> None:
        """
        Calibrates uniform threshold theta on calibration traces to achieve target recall.
        """
        miss_mask = calib_df["miss"].values.astype(bool) if "miss" in calib_df.columns else (
            calib_df["s_dag_ms"].values > self.deadline_ms
        )
        if np.sum(miss_mask) == 0:
            self.calibrated_threshold = self.fixed_threshold
            self.thresholds = {i: self.fixed_threshold for i in range(1, 7)}
            return

        min_p_missed = []
        for _, row in calib_df[miss_mask].iterrows():
            eval_res = self._evaluate_raw_row(row)
            min_p_missed.append(eval_res["min_p_meet"])

        calib_theta = float(np.percentile(min_p_missed, target_recall * 100))
        calib_theta = min(1.0, calib_theta + 1e-6)
        self.calibrated_threshold = calib_theta
        self.thresholds = {i: calib_theta for i in range(1, 7)}

    def _evaluate_raw_row(self, row: Union[pd.Series, Dict[str, float]]) -> dict:
        """Helper to extract AST and plaxity meeting probabilities from a single row."""
        t1 = float(row.get("dur_node1_ms", row.get("t_n1", 0.0)))
        t2 = float(row.get("dur_node2_ms", row.get("t_n2", 0.0)))
        t3 = float(row.get("dur_node3_ms", row.get("t_n3", 0.0)))
        t4 = float(row.get("dur_node4_ms", row.get("t_n4", 0.0)))
        t5 = float(row.get("dur_node5_ms", row.get("t_n5", 0.0)))

        if self.fork_mode == "wall" and "dur_fork_wall_ms" in row:
            f_dur = float(row["dur_fork_wall_ms"])
        elif self.fork_mode == "wall" and "t_fork" in row:
            f_dur = float(row["t_fork"])
        else:
            f_dur = max(t3, t4)

        ast = {
            1: 0.0,
            2: t1,
            3: t1 + t2,
            4: t1 + t2,
            5: t1 + t2 + f_dur,
            6: t1 + t2 + f_dur + t5,
        }

        p_meet = {}
        for i in [1, 2, 3, 4, 5, 6]:
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
        Evaluates frame arrival times AST against super-node plaxity-CDFs and thresholds.
        """
        raw = self._evaluate_raw_row(row)
        ast = raw["ast"]
        p_meet = raw["p_meet"]

        alarm_nodes = []
        # Check all unique physical checkpoints: 1, 2, 3 (fork), 5, 6
        for i in [1, 2, 3, 5, 6]:
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


class RegimeConditionedDetectorD1:
    """
    Candidate Detector D1: Observable-State Regime-Conditioned Plaxity Detector.

    Fits 18 per-regime TobaPlaxityBaseline detectors on calibration traces.
    At runtime, dynamically predicts active regime using ObservableStateEstimator
    and evaluates frame arrival times against that regime's plaxity model.
    """

    def __init__(
        self,
        scale: int = 10,
        policy: str = "matched_recall_95",
        window: int = 5,
        scale_factors: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> None:
        self.scale = int(scale)
        self.policy = policy
        self.window = int(window)
        self.scale_factors = scale_factors
        self.deadlines: Dict[int, float] = {}

        self.estimator = ObservableStateEstimator(
            window=self.window, scale_factors=self.scale_factors
        )
        self.regime_detectors: Dict[int, TobaPlaxityBaseline] = {}
        self.pooled_detector: Optional[TobaPlaxityBaseline] = None

    def fit(
        self,
        calib_df: pd.DataFrame,
        deadlines: Dict[int, float],
    ) -> "RegimeConditionedDetectorD1":
        """
        Fits ObservableStateEstimator and 18 per-regime TobaPlaxityBaseline detectors.
        """
        self.deadlines = {int(k): float(v) for k, v in deadlines.items()}
        self.estimator.fit(calib_df)

        cfg = TobaDetectorSuite.parse_policy(self.policy)
        target_rec = cfg.pop("target_recall", None)

        self.regime_detectors = {}
        for reg_id, d_reg in self.deadlines.items():
            reg_calib = calib_df[calib_df["regime_id"] == reg_id]
            if len(reg_calib) == 0:
                continue
            det = TobaPlaxityBaseline(scale=self.scale, **cfg)
            det.fit(reg_calib, deadline_ms=d_reg, target_recall=target_rec)
            self.regime_detectors[reg_id] = det

        return self

    def reset_episode(self) -> None:
        """Resets online rolling state for a new episode."""
        self.estimator.reset()

    def predict_regime(self, curr_t1: float, curr_t2: float) -> int:
        """Predicts active regime from observable state."""
        return self.estimator.predict_regime(curr_t1, curr_t2)

    def evaluate_frame(
        self,
        row: Union[pd.Series, Dict[str, float]],
        update_state: bool = True,
    ) -> dict:
        """
        Evaluates a frame:
        1. Predicts active regime using current upstream observations and history.
        2. Queries predicted regime's TobaPlaxityBaseline detector.
        3. Updates rolling history if update_state is True.
        """
        curr_t1 = float(row.get("dur_node1_ms", row.get("t_n1", 0.0)))
        curr_t2 = float(row.get("dur_node2_ms", row.get("t_n2", 0.0)))
        curr_sdag = float(row.get("s_dag_ms", row.get("t_s_dag", 0.0)))

        pred_r = self.predict_regime(curr_t1, curr_t2)
        detector = self.regime_detectors[pred_r]
        result = detector.evaluate_frame(row)
        result["pred_regime"] = pred_r

        if update_state:
            self.estimator.update(curr_t1, curr_t2, curr_sdag)

        return result

    def evaluate_episode(self, ep_df: pd.DataFrame) -> List[dict]:
        """Evaluates an entire episode sequentially, maintaining state across frames."""
        self.reset_episode()
        sorted_ep = ep_df.sort_values("frame_id")
        results = []
        for _, row in sorted_ep.iterrows():
            res = self.evaluate_frame(row, update_state=True)
            res["episode_id"] = int(row["episode_id"])
            res["frame_id"] = int(row["frame_id"])
            res["true_regime"] = int(row["regime_id"])
            actual_miss = bool(row["miss"]) if "miss" in row else (
                float(row.get("s_dag_ms", row.get("t_s_dag", 0.0))) > float(row["deadline_ms"])
            )
            res["actual_miss"] = actual_miss
            results.append(res)
        return results

    def evaluate_traces(self, traces_df: pd.DataFrame) -> List[dict]:
        """Evaluates traces across multiple episodes, resetting state at episode boundaries."""
        results = []
        for ep in traces_df["episode_id"].unique():
            ep_df = traces_df[traces_df["episode_id"] == ep]
            results.extend(self.evaluate_episode(ep_df))
        return results


class SuperNodePlaxityDetectorD2:
    """
    Candidate Detector D2: Super-Node Joint PMF + Observable-State Conditioning.

    Extends D1 by replacing fork branches N3 || N4 with empirical super-node
    N_fork = max(X3, X4) across 18 regimes, completely eliminating Eq. 11.
    Evaluates frame arrival times against the predicted regime's SuperNodePlaxityBaseline.
    """

    def __init__(
        self,
        scale: int = 10,
        policy: str = "matched_recall_95",
        window: int = 5,
        scale_factors: Optional[Union[np.ndarray, List[float]]] = None,
        fork_mode: str = "wall",
    ) -> None:
        self.scale = int(scale)
        self.policy = policy
        self.window = int(window)
        self.scale_factors = scale_factors
        self.fork_mode = fork_mode
        self.deadlines: Dict[int, float] = {}

        self.estimator = ObservableStateEstimator(
            window=self.window, scale_factors=self.scale_factors
        )
        self.regime_detectors: Dict[int, SuperNodePlaxityBaseline] = {}
        self.pooled_detector: Optional[SuperNodePlaxityBaseline] = None

    def fit(
        self,
        calib_df: pd.DataFrame,
        deadlines: Dict[int, float],
    ) -> "SuperNodePlaxityDetectorD2":
        """
        Fits ObservableStateEstimator and 18 per-regime SuperNodePlaxityBaseline detectors.
        """
        self.deadlines = {int(k): float(v) for k, v in deadlines.items()}
        self.estimator.fit(calib_df)

        cfg = TobaDetectorSuite.parse_policy(self.policy)
        target_rec = cfg.pop("target_recall", None)

        self.regime_detectors = {}
        for reg_id, d_reg in self.deadlines.items():
            reg_calib = calib_df[calib_df["regime_id"] == reg_id]
            if len(reg_calib) == 0:
                continue
            det = SuperNodePlaxityBaseline(
                scale=self.scale, fork_mode=self.fork_mode, **cfg
            )
            det.fit(reg_calib, deadline_ms=d_reg, target_recall=target_rec)
            self.regime_detectors[reg_id] = det

        return self

    def reset_episode(self) -> None:
        """Resets online rolling state for a new episode."""
        self.estimator.reset()

    def predict_regime(self, curr_t1: float, curr_t2: float) -> int:
        """Predicts active regime from observable state."""
        return self.estimator.predict_regime(curr_t1, curr_t2)

    def evaluate_frame(
        self,
        row: Union[pd.Series, Dict[str, float]],
        update_state: bool = True,
    ) -> dict:
        """
        Evaluates a frame:
        1. Predicts active regime using current upstream observations and history.
        2. Queries predicted regime's SuperNodePlaxityBaseline detector.
        3. Updates rolling history if update_state is True.
        """
        curr_t1 = float(row.get("dur_node1_ms", row.get("t_n1", 0.0)))
        curr_t2 = float(row.get("dur_node2_ms", row.get("t_n2", 0.0)))
        curr_sdag = float(row.get("s_dag_ms", row.get("t_s_dag", 0.0)))

        pred_r = self.predict_regime(curr_t1, curr_t2)
        detector = self.regime_detectors[pred_r]
        result = detector.evaluate_frame(row)
        result["pred_regime"] = pred_r

        if update_state:
            self.estimator.update(curr_t1, curr_t2, curr_sdag)

        return result

    def evaluate_episode(self, ep_df: pd.DataFrame) -> List[dict]:
        """Evaluates an entire episode sequentially, maintaining state across frames."""
        self.reset_episode()
        sorted_ep = ep_df.sort_values("frame_id")
        results = []
        for _, row in sorted_ep.iterrows():
            res = self.evaluate_frame(row, update_state=True)
            res["episode_id"] = int(row["episode_id"])
            res["frame_id"] = int(row["frame_id"])
            res["true_regime"] = int(row["regime_id"])
            actual_miss = bool(row["miss"]) if "miss" in row else (
                float(row.get("s_dag_ms", row.get("t_s_dag", 0.0))) > float(row["deadline_ms"])
            )
            res["actual_miss"] = actual_miss
            results.append(res)
        return results

    def evaluate_traces(self, traces_df: pd.DataFrame) -> List[dict]:
        """Evaluates traces across multiple episodes, resetting state at episode boundaries."""
        results = []
        for ep in traces_df["episode_id"].unique():
            ep_df = traces_df[traces_df["episode_id"] == ep]
            results.extend(self.evaluate_episode(ep_df))
        return results
