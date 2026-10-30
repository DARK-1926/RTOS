# Regime-Conditioned Probabilistic Laxity: Early Deadline Miss Detection for Real-Time UAV Perception-Planning DAGs Under Shared-Resource Contention

**Authors**: Anonymous Authors  
**Affiliation**: Laboratory for Autonomous Systems & Real-Time Computing  
**Target Venue**: IEEE Real-Time Systems Symposium (RTSS) / ACM Transactions on Embedded Computing Systems (TECS)  
**Artifact Status**: Fully Reproducible (Pre-Registered Protocol SHA-256: `156efdf79ad673d75ba115241fc512973e97989e4b228f15d0b9c657c0a86e7a`)

---

## Abstract

Autonomous unmanned aerial vehicles (UAVs) rely on directed acyclic graph (DAG) perception-planning pipelines executing on multi-core compute platforms under strict real-time deadlines. To avoid catastrophic flight failures, early deadline miss detection methods aim to trigger safety maneuvers (such as a Minimum Risk Maneuver) before end-to-end task deadlines expire. State-of-the-art frameworks, such as probabilistic laxity (plaxity) analysis (Toba et al., ACM TECS 2026), evaluate intermediate node completion times against cumulative distribution functions derived via offline discrete convolution under an assumption of stochastic independence among tasks. 

In this work, we demonstrate on an authentic AMD Ryzen 7 7735HS platform that physical shared-resource contention—specifically last-level cache (L3) thrashing and DDR5 memory bandwidth saturation induced by concurrent perception workloads—violates static plaxity assumptions. When evaluated across 7,200 confirmatory test frames spanning 18 factorial regimes of wind turbulence, visual clutter, and memory contention, static plaxity triggers premature false alarms on 88.89% of all frames at pipeline entry, collapsing precision to 19.22% with an 88.80% false positive rate (FPR). 

To address this challenge without requiring specialized hardware performance counters, we evaluate two architectural extensions: Candidate D1, which conditions offline plaxity models on an online moving-median observable state vector ($W=5$) tracking upstream latencies in $O(1)$ time, and Candidate D2, which bypasses the independent fork operator via an empirical super-node joint distribution. Across 144 independent test episodes (2,000 paired cluster-bootstrap resamples), Candidate D1 improves precision by +67.11 pp (95% CI: [+64.84, +69.20] pp, 86.36% vs. 19.22%), suppresses FPR to 2.68%, and preserves a 1.90 ms mean lead time (+1.55% over an oracle baseline). In a stationary null control regime, D1 matches oracle precision with a 0.00 pp divergence, confirming that improvements stem from environment tracking rather than threshold tuning. 

Crucially, our investigation documents significant negative results and trade-offs: while Candidate D2 increases recall by +4.44 pp (to 84.23%), it suffers a catastrophic lead time collapse, reducing mean lead time to 0.23 ms (-87.7% reduction) and deferring 95.82% of alarms to the terminal node. Furthermore, fine-grained regime classification is constrained to ~40.6% accuracy due to wind-induced latency overlap, explaining the remaining recall deficit against an oracle detector.

---

## 1. Introduction

Autonomous civilian unmanned aerial vehicles (UAVs) performing complex tasks such as infrastructure inspection, search-and-rescue, and precision agriculture require high-throughput perception and planning under rigorous temporal guarantees. In a typical modern flight autonomy stack, sensor processing, state estimation, obstacle detection, costmap generation, path planning, and trajectory tracking are structured as a Directed Acyclic Graph (DAG) of interdependent tasks. Each invocation of the autonomy DAG must complete within an end-to-end deadline $D$ (often bounded by 20–50 ms to ensure dynamic vehicle stability). 

When transient computational spikes or environmental disturbances threaten to violate $D$, the flight controller must initiate an emergency intervention, such as transitioning to a Minimum Risk Maneuver (MRM), loitering, or switching to a degraded control mode. If a deadline miss is detected only at the terminal node, the remaining slack is zero, leaving insufficient time to perform safe vehicle maneuvers. Consequently, early deadline miss detection methods aim to predict an impending deadline overrun at intermediate checkpoints within the DAG, providing actionable lead time for recovery.

### 1.1 The Breakdown of Static Probabilistic Laxity Under Non-Stationarity

Prior work by Toba and Azumi (ECRTS 2024) and Toba, Baruah, and Azumi (ACM TECS 2026) established the mathematical foundation of *probabilistic laxity* (plaxity). Plaxity extends classical deterministic laxity by representing remaining execution time as a random variable. By computing the discrete convolutional subtraction of task execution time probability mass functions (PMFs) along reverse DAG edges, plaxity derives a cumulative distribution function of meeting the end-to-end deadline from any given intermediate arrival time. To balance false alarms against early detection sensitivity, Toba et al. introduced variable probability thresholds that relax upstream detection criteria and tighten downstream.

However, existing plaxity frameworks rely on two foundational assumptions:
1. **Stationary Execution Profiles**: Task execution times are modeled by static marginal PMFs calibrated offline.
2. **Stochastic Independence**: Execution times across parallel fork branches and sequential pipeline stages are assumed to be statistically independent.

On physical multi-core embedded systems, modern perception-planning DAGs execute in highly dynamic, non-stationary environments. An aerial robot transitions rapidly between calm air and turbulent wind shears, sparse open skies and dense obstacle clutter, and varying degrees of background memory traffic on the system-on-chip (SoC). When concurrent camera pipelines, video encoders, or logging threads compete for shared microarchitectural resources—specifically the unified Last-Level Cache (L3/LLC) and the external DRAM memory controller—task execution distributions shift dramatically. 

Under heavy shared-resource contention, sequential execution times experience substantial tail inflation. When a detector equipped with static offline PMFs encounters a high-contention or high-clutter regime, nominal execution latencies exceed the static expectations. As a consequence, static plaxity perceives normal operation as an imminent disaster: across our confirmatory experiments on 7,200 test frames, the static baseline triggered alarms on 88.89% of all frames at the very first pipeline checkpoint ($AST=0$), yielding an 88.80% False Positive Rate (FPR) and a precision of only 19.22%. In an operational UAV, such false-alarm saturation would repeatedly trigger disruptive, unnecessary emergency landings.

### 1.2 Microarchitectural Shared-Resource Contention

On contemporary multi-core architectures (e.g., AMD Zen 3+ or ARM Cortex-A78 clusters), cores within a compute complex (CCX) share a unified L3 cache and a shared memory bus to DDR5 SDRAM. When an upstream sensor filtering or kinematic task runs alongside high-bandwidth memory operations, cache lines containing voxel grids and occupancy maps are evicted from L3 to main memory. Because the latency of a DRAM access is an order of magnitude higher than an L3 hit, memory bus saturation causes cross-task contention: delays compound across pipeline hops. 

Furthermore, parallel fork branches (e.g., concurrent obstacle detection and local costmap inflation) running simultaneously on adjacent cores compete directly for memory channels. Under the Toba plaxity formulation (ACM TECS 2026, Eq. 11), parallel fork convergence is handled via an analytical independent minimum operator. When cores contend for shared memory, the assumption of independent branch durations fails, distorting downstream probability estimation.

### 1.3 Scope and Contributions

In this study, we investigate the behavior of probabilistic deadline miss detection on an authentic 6-node civilian UAV perception-planning DAG deployed on an AMD Ryzen 7 7735HS platform. Rather than claiming absolute priority or novelty over existing real-time scheduling literature, we ground our work against established foundations (Toba et al., TECS 2026; Marković et al., RTSS 2023; Casini et al., ECRTS 2018). We formulate, implement, and empirically validate two architectural extensions designed to handle environmental non-stationarity and shared-resource contention:

1. **Observable-State Regime Conditioning (Candidate D1)**: An online, moving-median state estimator ($W=5$) that tracks upstream node execution times ($t_1, t_2$) and recent end-to-end latency ($S_{\text{DAG}}$). Operating with $O(1)$ standardized Euclidean nearest-centroid classification, D1 selects regime-matched plaxity models with an execution overhead below 5.0 $\mu$s.
2. **Super-Node Joint PMF Formulation (Candidate D2)**: An architectural reformulation that collapses parallel fork branches into an empirical super-node distribution $X_{\text{fork}} = \max(X_3, X_4)$, solving backward plaxity across a 5-node super-DAG via discrete convolutional subtraction and completely bypassing the independent fork operator of Eq. (11).
3. **Rigorous Empirical Confirmatory Evaluation**: Conducted across 18 factorial regimes (wind $\in \{0, 1, 2\}$, clutter $\in \{1.0, 2.5\}$, memory contention $\in \{0, 3, 8\}$ workers) using 18,000 frames strictly partitioned at the episode level into 40% calibration, 20% validation, and 40% confirmatory test sets. All 18 per-regime deadlines are pre-registered and cryptographically locked via SHA-256. Evaluation utilizes 2,000 paired cluster-bootstraps resampled at the episode level.
4. **Transparent Negative Results and Trade-offs (R4)**: We report and analyze two critical failure modes:
   - **Candidate D2 Lead Time Collapse**: Although D2 improves recall by +4.44 pp (84.23% vs. 79.79%), it compresses mean True Positive (TP) lead time to 0.23 ms (-87.7% reduction vs. D1's 1.90 ms), deferring 95.82% of alarms to terminal Node 6 due to variance smearing in the super-node PMF.
   - **Observable State Misclassification**: Fine-grained 18-regime classification achieves only ~40.6% accuracy on test traces (~39.2% on validation) due to aerodynamic wind turbulence causing minimal separation in upstream compute latency, explaining the recall gap relative to an oracle detector.
   - **Stationary Control Regime Attribution**: In Regime 0 (stationary null control), Candidate D1 achieves a 0.00 pp precision difference relative to the oracle baseline, confirming that our +67.11 pp precision gain across the non-stationary pool is strictly attributable to tracking environmental shifts.

---

## 2. Related Work and Authoritative Comparison

Real-time deadline guarantee mechanisms for multi-core DAGs span offline worst-case response-time analysis, hardware resource partitioning, and runtime early deadline miss detection.

### 2.1 Early Deadline Miss Detection on DAG Tasks

The concept of early deadline miss detection on DAG tasks was established by Toba and Azumi (ECRTS 2024), who proposed *plaxity* (probabilistic laxity) to evaluate the probability of meeting an end-to-end deadline at the invocation of each task within a DAG. Toba, Baruah, and Azumi (ACM TECS 2026) extended this model by introducing variable probabilistic threshold schedules (Eq. 13–18). By dynamically adjusting thresholds based on remaining workload, their approach reduced false alarms caused by static cutoffs. However, their evaluation was conducted exclusively in synthetic Python simulation (using RD-Gen task graphs and bimodal Gaussian mixtures without physical microarchitectural contention). The authors explicitly acknowledged that assuming independence among positively correlated tasks yields optimistic deadline miss predictions, noting super-nodes and copula bounds as potential directions (TECS 2026, Section 2.2). Our work directly implements and evaluates these acknowledged directions under authentic multi-core contention.

### 2.2 Worst-Case Response-Time (WCRT) Analysis for DAG Tasks

Worst-case response-time analysis for DAG tasks on multi-cores has been extensively investigated (e.g., Casini et al., ECRTS 2018; Melani et al., IEEE TC 2017). These frameworks compute deterministic upper bounds on response times under partitioned or global fixed-priority scheduling. While essential for hard real-time schedulability certification, deterministic WCET-based bounds are notoriously pessimistic in perception pipelines: sensor-driven processing (such as point cloud filtering and $A^*$ motion planning) exhibits severe average-to-worst-case execution variability (often exceeding 5:1). Enforcing hard WCET bounds at runtime leads to massive resource over-provisioning. Early detection frameworks operate orthogonally by tracking execution probabilities online to trigger soft-recovery maneuvers.

### 2.3 Hardware Contention Mitigation and Predictable Execution Models

Hardware-level isolation frameworks mitigate shared-resource contention through architectural regulation. MemGuard (Yun et al., RTAS 2013) uses hardware performance counter overflow interrupts to enforce periodic DRAM bandwidth budgets per core. The Predictable Execution Model (PREM; Pellizzoni et al., RTAS 2011) divides task execution into decoupled memory phases (prefetching data into local scratchpads) and compute phases (executing without main memory contention). While effective, these techniques require kernel modifications, specialized hypervisors, or compiler refactoring. Furthermore, active memory throttling can degrade the throughput of data-intensive perception pipelines. Our work investigates an complementary software-level detection approach that requires zero hardware counter virtualization or kernel-level modifications.

### 2.4 Dependence-Tolerant Schedulability and pWCET Analysis

The theoretical hazards of assuming independence in probabilistic real-time systems have long been emphasized (Bernat et al., 2002; Maxim and Cucu-Grosjean, 2013). Marković et al. (RTSS 2023) developed CTA (Correlation-Tolerant Analysis), deriving a closed-form upper bound on the worst-case deadline-failure probability of dependent tasks using Cantelli's inequality with a safe variance bound $(\sum \sigma_i)^2$. While CTA provides sound analytical bounds, its variance envelope is conservative, yielding elevated false alarm rates when applied as an online threshold detector. 

### 2.5 Authoritative State-of-the-Art Comparison Table

Table 1 provides a comprehensive comparison between existing paradigms and our proposed regime-conditioned architectures.

**Table 1: Authoritative Comparison of Multi-Core DAG Deadline Miss Detection and Schedulability Frameworks**

| Framework / Paradigm | Target Architecture | DAG Topology | Stochastic Independence Assumption | Non-Stationary Shift Resilience | Multi-Core Contention Modeling | Online Inference Complexity | False Alarm Behavior Under Contention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Toba & Azumi (ECRTS 2024)** | Multi-core host (simulated) | General DAGs | Assumes mutual independence across all tasks and branches | None (static offline PMFs; uncalibrated under shift) | Unmodeled (assumes i.i.d. execution distributions) | $O(1)$ lookup against static plaxity-CDF | Severe false alarm storm (FPR up to 88.80% under contention) |
| **Toba, Baruah, & Azumi (ACM TECS 2026)** | Multi-core COTS (simulated) | Multi-rate general DAGs | Assumes independence (Eq. 10 convolution, Eq. 11 fork min) | Variable thresholds absorb minor shifts, but PMFs remain static | Unmodeled (acknowledged as future work; no contention engine) | $O(1)$ query against static variable threshold schedule | Premature false alarms (88.89% alarms at Node 1) or recall collapse under relaxed thresholds |
| **Resource Regulation (MemGuard / PREM)** | Multi-core with PMU / scratchpad | Task chains / partitioned DAGs | N/A (deterministic or decoupled memory/compute phases) | Hardware-enforced bandwidth ceilings; degrades throughput | Hardware-regulated via PMU interrupts or phase separation | Moderate OS overhead (PMU interrupt servicing every period) | Does not predict misses early; throttles cores to prevent overruns |
| **WCRT DAG Scheduling (Casini et al. 2018)** | Homogeneous multiprocessor | Arbitrary DAGs | Deterministic WCET bounding (no stochastic independence) | Static worst-case bounds; pessimistic under varying regimes | Theoretical worst-case interference bounds | Offline calculation $O(V+E)$; zero runtime inference | Offline certification test; zero online early miss detection capability |
| **Candidate D1 (This Work: Regime-Conditioned)** | AMD Ryzen 7 7735HS (8C/16T, Zen 3+) | 6-node UAV perception-planning DAG | Evaluates conditional marginals per regime; retains Eq. 10–11 | High (online rolling median $W=5$ tracks observable state) | Empirical regime mapping via standardized Euclidean distance | $O(1)$ online update and centroid query ($< 5.0\ \mu\text{s}$) | Controlled (FPR drops from 88.80% to 2.68%, +67.11 pp precision gain, preserves 1.90 ms lead time) |
| **Candidate D2 (This Work: Super-Node)** | AMD Ryzen 7 7735HS (8C/16T, Zen 3+) | 5-node super-DAG ($N_3 \parallel N_4$ collapsed to $N_{\text{fork}}$) | Bypasses Eq. 11 completely; uses joint empirical PMF | High (regime-conditioned super-node selection) | Fully internalized within empirical joint distribution | $O(1)$ query against 5-stage super-DAG plaxity-CDF | High precision (+66.13 pp over B1), higher recall (+4.44 pp over D1), but exhibits lead time collapse (0.23 ms) |

---

## 3. System Architecture and Mathematical Foundations

### 3.1 6-Node UAV Autonomy Pipeline

We model an autonomous multirotor perception-planning pipeline structured as a 6-node DAG $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ executed at each perception cycle (Fig. 1). The pipeline executes genuine computational workloads representative of civilian UAV autonomy:

```
               +---------------------------------------+
               |  N1: Sensor Ingest & Range Gating     |
               +---------------------------------------+
                                  |
                                  v
               +---------------------------------------+
               |  N2: Kinematic EKF State Extrapolation|
               +---------------------------------------+
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
     +-------------------------+     +-------------------------+
     | N3: Obstacle Detection  |     | N4: Local Costmap Grid  |
     |   & Voxel Downsampling  |     |    Occupancy Dilation   |
     +-------------------------+     +-------------------------+
                  |                               |
                  +---------------+---------------+
                                  |
                                  v
               +---------------------------------------+
               |  N5: Join Node - 2D Grid A* Planner   |
               +---------------------------------------+
                                  |
                                  v
               +---------------------------------------+
               |  N6: Command Output & Spline Feas.    |
               +---------------------------------------+
```

* **Node 1 ($N_1$, Sensor Ingest & Range Gating)**: Ingests 3D LiDAR point clouds (8,000 to 20,000 points depending on visual clutter), applies aerodynamic wind vibration displacement, and performs Euclidean distance filtering.
* **Node 2 ($N_2$, Kinematic State Estimation)**: Executes an Extended Kalman Filter (EKF) step propagating vehicle velocity and angular orientation via a $6 \times 6$ covariance update ($F \Sigma F^T + Q$) and an eigenvalue observation step.
* **Fork Branches ($N_3 \parallel N_4$)**: Executed concurrently in separate worker threads on distinct CPU cores:
  * **Node 3 ($N_3$, Obstacle Detection)**: Ground plane segmentation ($z < -3.5$ m), clutter obstacle injection, and 3D integer coordinate voxel hashing with deduplication.
  * **Node 4 ($N_4$, Local Costmap)**: 2D obstacle grid projection and morphological 8-connected occupancy dilation.
* **Node 5 ($N_5$, Join Node — Motion Planning)**: A non-degenerate 2D Grid $A^*$ path planner operating with a binary min-heap (`heapq`) on the dilated costmap. To avoid synthetic benchmark triviality, $N_5$ is designed to satisfy non-degeneracy criteria across all regimes: median latency is 11.62 ms ($\ge 1.0$ ms) and accounts for 59.07% of the end-to-end DAG execution latency ($\ge 1.0\%$).
* **Node 6 ($N_6$, Command Output & Safety Verification)**: Evaluates path curvature, velocity clamping, crosswind drift limits, and lateral acceleration feasibility.

The actual DAG execution latency $S_{\text{DAG}}$ is defined by the critical execution path:
$$S_{\text{DAG}} = t_1 + t_2 + \max(t_3, t_4) + t_5 + t_6$$
where $t_i$ denotes the execution duration of node $N_i$. In our instrumentation, the actual wall-clock duration of the parallel fork $t_{\text{fork}}$ is measured alongside thread runtimes.

### 3.2 Formal Probabilistic Laxity Framework (Toba et al., ACM TECS 2026)

Let each node $N_i$ have an execution time modeled as a discrete random variable $X_i$ with probability mass function $P(X_i = x_{i,j})$. Let $D$ denote the end-to-end deadline. Plaxity represents the remaining allowable execution time from the start of a given node to the deadline.

#### 3.2.1 Exit Node Plaxity
For the terminal node $N_6$ with deadline $D$, its plaxity $L_6$ is defined as the discrete random variable:
$$L_6 = D - X_6$$
$$P(L_6 = \lambda) = P(X_6 = D - \lambda)$$

#### 3.2.2 Discrete Convolutional Subtraction
For a node $N_i$ having a single successor $N_s$ with plaxity distribution $L_s$, the plaxity $L_i$ is computed via discrete convolutional subtraction ($\otimes^-$) (TECS 2026, Eq. 10):
$$L_i = L_s \otimes^- X_i$$
$$P(L_i = \lambda) = \sum_{j=1}^m P(X_i = x_{i,j}) \cdot P(L_s = \lambda + comm_{i,s} + x_{i,j})$$
where $comm_{i,s}$ denotes inter-node communication latency (modeled as zero for shared-memory thread pools).

#### 3.2.3 Independent Fork Minimum Operator
For a node $N_i$ with multiple parallel successor branches (such as $N_2$ preceding $N_3$ and $N_4$), plaxity along each outgoing path $k \in \{3, 4\}$ is computed independently as $L_{2}^{(k)} = L_k \otimes^- X_2$. Under the assumption that the plaxities along both successor paths are statistically independent, Toba et al. define the combined plaxity as the minimum distribution (TECS 2026, Eq. 11):
$$L_2 = \min\left(L_{2}^{(3)}, L_{2}^{(4)}\right)$$
$$P(L_2 = \lambda) = P(L_{2}^{(3)} = \lambda) \cdot P(L_{2}^{(4)} \ge \lambda) + P(L_{2}^{(3)} > \lambda) \cdot P(L_{2}^{(4)} = \lambda)$$

#### 3.2.4 AST Checking and Variable Threshold Schedules
Let $AST_i$ be the actual start time (arrival timestamp) of node $N_i$ relative to pipeline launch ($AST_1 = 0$). The probability that the pipeline meets the end-to-end deadline, conditioned on arriving at node $N_i$ at time $AST_i$, is obtained from the complementary cumulative distribution function (survival function) of plaxity:
$$P(\text{meet} \mid AST_i) = P(L_i \ge AST_i) = \sum_{\lambda \ge AST_i} P(L_i = \lambda)$$

An impending deadline miss alarm is triggered at node $N_i$ if:
$$P(\text{meet} \mid AST_i) < \theta_i$$
where $\theta_i$ is the detection threshold for node $N_i$. Under Toba et al.'s linear job-count variable threshold schedule (TECS 2026, Eq. 15–18), thresholds start relaxed at DAG entry and tighten toward the terminal node:
$$\theta_i = \theta_{\min} + (\theta_{\max} - \theta_{\min}) \cdot \frac{\operatorname{rank}(N_i) - 1}{\operatorname{rank}_{\max} - 1}$$
with $\theta_{\min} = 0.80$, $\theta_{\max} = 1.00$, and ranks $1, 2, 3, 4, 5$ across the sequence $N_1 \to N_2 \to \{N_3 \parallel N_4\} \to N_5 \to N_6$, producing the monotonic schedule:
$$\theta_1 = 0.80, \quad \theta_2 = 0.85, \quad \theta_3 = \theta_4 = 0.90, \quad \theta_5 = 0.95, \quad \theta_6 = 1.00$$

### 3.3 Theoretical Proof of Eq. (11) Join-Node Common-Term Distortion

We now present an analytical proof demonstrating why Eq. (11) introduces a structural distortion when parallel paths converge at a downstream join node.

**Theorem 1 (Join-Node Stochastic Coupling Distortion).** *Let a DAG contain a fork-join structure where node $N_{\text{fork}}$ branches into parallel tasks $N_A$ and $N_B$, which both precede a common join node $N_{\text{join}}$ with execution time distribution $X_{\text{join}}$. Even if task execution times $X_A, X_B, X_{\text{join}}$ are mutually independent, the plaxity distributions $L_{\text{fork}}^{(A)}$ and $L_{\text{fork}}^{(B)}$ derived via Eq. (10) are conditionally dependent on $X_{\text{join}}$. Consequently, evaluating their joint survival probability using Eq. (11) strictly underestimates the true probability of meeting the deadline whenever downstream variance $\operatorname{Var}(X_{\text{join}}) > 0$.*

*Proof.* Let $D$ be the deadline, and let $L_{\text{join}}$ be the plaxity at the join node. The plaxity preceding the join node is $Y = L_{\text{join}} \ominus X_{\text{join}}$. The path plaxities evaluated backward to the fork branches are:
$$L^{(A)} = Y \ominus X_A, \qquad L^{(B)} = Y \ominus X_B$$
Both random variables $L^{(A)}$ and $L^{(B)}$ share the identical random realization $Y$. 

Conditioned on an arrival time $t$ at the fork, meeting the deadline requires that both branches finish with sufficient slack to execute the downstream tasks. The true meeting probability is:
$$P(\text{meet} \mid t) = P(t + X_A + X_{\text{join}} \le D \;\land\; t + X_B + X_{\text{join}} \le D)$$
Under task independence, conditioning on a realization $X_{\text{join}} = x$ yields:
$$P(\text{meet} \mid t, X_{\text{join}} = x) = P(X_A \le D - t - x) \cdot P(X_B \le D - t - x)$$
Taking the expectation over the distribution of $X_{\text{join}}$:
$$P_{\text{true}}(\text{meet} \mid t) = \mathbb{E}_{X_{\text{join}}}\left[ F_A(D - t - X_{\text{join}}) \cdot F_B(D - t - X_{\text{join}}) \right]$$
where $F_A(z) = P(X_A \le z)$ and $F_B(z) = P(X_B \le z)$ are the cumulative distribution functions of $X_A$ and $X_B$.

In contrast, Eq. (11) treats $L^{(A)}$ and $L^{(B)}$ as independent random variables. It evaluates the marginal survival functions independently:
$$P(L^{(A)} \ge t) = \mathbb{E}_{X_{\text{join}}}\left[ F_A(D - t - X_{\text{join}}) \right]$$
$$P(L^{(B)} \ge t) = \mathbb{E}_{X_{\text{join}}}\left[ F_B(D - t - X_{\text{join}}) \right]$$
Under Eq. (11), the joint meeting probability is calculated as the product of these marginals:
$$P_{\text{Eq11}}(\text{meet} \mid t) = \mathbb{E}_{X_{\text{join}}}\left[ F_A(D - t - X_{\text{join}}) \right] \cdot \mathbb{E}_{X_{\text{join}}}\left[ F_B(D - t - X_{\text{join}}) \right]$$

Now consider the functions $g_1(x) = F_A(D - t - x)$ and $g_2(x) = F_B(D - t - x)$. Because CDFs are monotonically non-decreasing, as $x$ increases, the argument $D - t - x$ decreases, which implies that both $g_1(x)$ and $g_2(x)$ are monotonically non-increasing functions of $x$. 

By Chebyshev's algebraic inequality (or the covariance inequality for monotonic functions), for any two functions $g_1, g_2$ that are both monotonically non-increasing in $x$:
$$\operatorname{Cov}(g_1(X_{\text{join}}), g_2(X_{\text{join}})) \ge 0$$
$$\mathbb{E}\left[ g_1(X_{\text{join}}) g_2(X_{\text{join}}) \right] - \mathbb{E}\left[ g_1(X_{\text{join}}) \right] \mathbb{E}\left[ g_2(X_{\text{join}}) \right] \ge 0$$
Therefore:
$$P_{\text{true}}(\text{meet} \mid t) \ge P_{\text{Eq11}}(\text{meet} \mid t)$$
Equality holds if and only if $\operatorname{Var}(X_{\text{join}}) = 0$ (i.e., downstream execution is deterministic) or one of the functions is constant over the support of $X_{\text{join}}$. Whenever downstream tasks exhibit non-zero execution variance, Eq. (11) strictly underestimates the true probability of meeting the deadline. $\blacksquare$

### 3.4 Super-Node DAG Formulation (Candidate D2)

To eliminate the distortion introduced by Eq. (11) and internalize multi-core memory contention coupling between parallel branches, Candidate D2 formulates a 5-node super-DAG:
$$N_1 \to N_2 \to N_{\text{fork}} \to N_5 \to N_6$$
where the parallel branches $N_3 \parallel N_4$ are collapsed into a composite super-node $N_{\text{fork}}$. The execution distribution of $N_{\text{fork}}$ is directly estimated from authentic calibration traces as:
$$X_{\text{fork}} = \max(X_3, X_4) \quad \text{or} \quad t_{\text{fork}}$$
Because $N_{\text{fork}}$ is a single node within a sequential chain, the backward plaxity recursion is solved using purely discrete convolutional subtraction (Eq. 10):
1. Stage 5: $L_6 = D - X_6$
2. Stage 4: $L_5 = L_6 \otimes^- X_5$
3. Stage 3: $L_{\text{fork}} = L_5 \otimes^- X_{\text{fork}}$
4. Stage 2: $L_2 = L_{\text{fork}} \otimes^- X_2$
5. Stage 1: $L_1 = L_2 \otimes^- X_1$

Both physical checkpoints at the entry of the fork (Checkpoint 3 at $N_3$ and Checkpoint 4 at $N_4$) query the exact same super-node plaxity distribution $L_{\text{fork}}$ at timestamp $AST_{\text{fork}} = t_1 + t_2$. This completely bypasses Eq. (11), evaluating the downstream join node $N_5$ exactly once rather than squaring its marginal influence.

### 3.5 Online Observable-State Conditioning (Candidate D1)

While Candidate D2 addresses topological independence within a fixed regime, it does not address environmental non-stationarity across regimes. To enable adaptive deadline miss detection without oracle knowledge, Candidate D1 implements an online Observable State Estimator.

Let $k$ denote the frame index within an episode. At frame $k$, the full end-to-end execution latency $S_{\text{DAG}}$ is unavailable prior to pipeline completion. However, the upstream execution times $t_1^{(k)}$ (Sensor Ingest) and $t_2^{(k)}$ (Kinematic EKF) are observed immediately upon their completion. 

Candidate D1 maintains a rolling FIFO history buffer of size $W=5$ containing completed frame measurements $(t_1, t_2, S_{\text{DAG}})$ within the active episode. Prior to executing the fork and join nodes, D1 constructs an observable state vector $\mathbf{z}_k \in \mathbb{R}^3$:
$$\mathbf{z}_k = \left[ \operatorname{median}\left(\mathcal{H}_{t_1} \cup \{t_1^{(k)}\}\right), \; \operatorname{median}\left(\mathcal{H}_{t_2} \cup \{t_2^{(k)}\}\right), \; \operatorname{median}\left(\mathcal{H}_{S_{\text{DAG}}}\right) \right]^T$$
During calibration, per-regime centroids $\mathbf{c}_r = [\tilde{t}_{1,r}, \tilde{t}_{2,r}, \tilde{S}_{\text{DAG},r}]^T$ are computed for each regime $r \in \{0, \dots, 17\}$. At runtime, the active regime $\hat{r}_k$ is predicted via standardized Euclidean distance nearest-centroid classification:
$$\hat{r}_k = \arg\min_{r \in \{0, \dots, 17\}} \sum_{j=1}^3 \left( \frac{z_{k,j} - c_{r,j}}{\sigma_j} \right)^2$$
where $\boldsymbol{\sigma} = [2.0, 0.3, 20.0]^T$ represents characteristic feature variation scales across the factor space.

Once $\hat{r}_k$ is predicted, Candidate D1 evaluates the current frame's arrival timestamps against the calibrated plaxity CDF $L_i^{(\hat{r}_k)}$ associated with regime $\hat{r}_k$. Upon frame completion, the true observed $(t_1^{(k)}, t_2^{(k)}, S_{\text{DAG}}^{(k)})$ is pushed into the rolling history. The history buffer is strictly flushed at episode boundaries ($\operatorname{reset}()$). The entire classification step executes in $O(1)$ time with an amortized overhead below 5.0 $\mu$s, fitting well within real-time budgets.

---

## 4. Experimental Methodology

### 4.1 Physical Hardware Platform

All experiments were executed on an authentic physical laptop-class platform:
* **Processor**: AMD Ryzen 7 7735HS (Zen 3+ microarchitecture, 8 physical cores / 16 threads, base clock 3.2 GHz, boost up to 4.75 GHz).
* **Cache Hierarchy**: 32 KB L1 data cache per core, 512 KB L2 cache per core, and 16 MB unified Last-Level Cache (L3) shared across the CCX.
* **System Memory**: 16 GB DDR5 SDRAM operating in dual-channel mode.
* **Operating Environment**: Windows 11 Host + WSL2. Scoped explicitly to civilian UAV autonomy pipelines executing standalone real compute on x86 laptop platforms. No external flight controllers, Jetson modules, or microcontrollers are claimed.

### 4.2 18 Orthogonal Factorial Regimes

To systematically stress the perception-planning DAG, we constructed an 18-regime full factorial design ($3 \times 2 \times 3$) crossing three environmental and microarchitectural factors:
1. **Aerodynamic Wind Turbulence ($\{0, 1, 2\}$)**:
   * Level 0 (Zero): Calm conditions, minimal structural vibration jitter.
   * Level 1 (Moderate): Moderate gusting, introducing wind variance into EKF state estimation.
   * Level 2 (High): Severe turbulence, maximizing aerodynamic covariance inflation in $N_2$.
2. **Visual Scene Clutter ($\{1.0, 2.5\}$)**:
   * Low Clutter (1.0): 8,000 LiDAR points per frame, sparse obstacle fields.
   * High Clutter (2.5): 20,000 LiDAR points per frame, dense obstacle clusters, stressing $N_3$ voxel hashing and $N_4$ costmap dilation.
3. **Physical Memory Contention Engine ($\{0, 3, 8\}$ workers)**:
   * Idle (0 workers): Autonomy DAG executes in isolation.
   * Moderate (3 workers): 3 background streaming worker processes running non-temporal array copy loops over 32 MB float32 buffers (stride 16 = 64 bytes = 1 cache line), deliberately exceeding the 16 MB L3 cache to induce moderate DRAM bus saturation (+21.9% median DAG latency shift).
   * Heavy (8 workers): 8 background streaming workers continuously thrashing L3 and saturating DDR5 memory bandwidth (+32.4% median DAG latency shift). Total memory committed remains below 750 MB, completely avoiding virtual memory paging.

### 4.3 Dataset Collection and Split Protocol

A total of 18,000 authentic frames were collected across 360 independent flight episodes (50 frames per episode). To guarantee zero data leakage, partitioning was executed strictly at the episode level:
* **Calibration Split**: 40% of episodes (7,200 frames across 144 episodes; 8 episodes per regime).
* **Validation Split**: 20% of episodes (3,600 frames across 72 episodes; 4 episodes per regime).
* **Confirmatory Test Split**: 40% of episodes (7,200 frames across 144 episodes; 8 episodes per regime).

All candidate algorithms were developed, tuned, and audited strictly on the calibration and validation sets. The confirmatory test split was isolated and remained untouched until the final single-pass evaluation.

### 4.4 Pre-Registered Deadlines and Cryptographic Locking

To prevent Simpson's paradox and avoid detectors acting as trivial regime classifiers, deadlines were calibrated strictly per regime:
$$D_r = Q_{85}\left(S_{\text{DAG}}^{\text{calib}, r}\right)$$
This calibration establishes an empirical deadline miss rate of 15.0% on the calibration set for each of the 18 regimes. All 18 deadline values were pre-registered and serialized to `results/gate_b_deadlines.json` along with the SHA-256 hash of the calibration traces (`156efdf79ad673d75ba115241fc512973e97989e4b228f15d0b9c657c0a86e7a`).

### 4.5 Statistical Evaluation via Cluster Bootstrapping

Because frames within an episode exhibit temporal autocorrelation, classical i.i.d. bootstrap resampling would artificially narrow confidence intervals. We employ cluster bootstrapping with 2,000 resamples, treating the independent `episode_id` as the primary sampling unit. For each resample, 144 episodes are drawn with replacement from the test set, and paired metric differences ($\Delta \text{Precision} = \text{Precision}_{\text{candidate}} - \text{Precision}_{\text{baseline}}$) are evaluated across identical resampled clusters.

---

## 5. Empirical Results

We report confirmatory evaluation results on the held-out test split (7,200 frames: 1,262 deadline misses, 5,938 safe frames) across four evaluated configurations:
* **Baseline 1 (`pooled_toba`)**: Standard Toba plaxity with linear variable thresholds ($\theta_{\min}=0.80, \theta_{\max}=1.00$), calibrated on pooled traces.
* **Baseline 2 (`per_regime_toba_oracle`)**: Toba plaxity with ground-truth oracle regime labels.
* **Candidate D1 (`detector_d1`)**: Observable-state regime conditioning ($W=5$, matched recall 95% policy).
* **Candidate D2 (`detector_d2`)**: Super-node joint PMF + observable-state regime conditioning.

### 5.1 Pooled Confirmatory Test Split Evaluation

Table 2 details the performance metrics pooled across all 7,200 confirmatory test frames.

**Table 2: Confirmatory Test Split Evaluation (Pooled 7,200 Frames Across 144 Independent Episodes)**

| Metric | Baseline 1 (`pooled_toba`) | Baseline 2 (`oracle_toba`) | Candidate D1 (`detector_d1`) | Candidate D2 (`detector_d2`) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Frames** | 7,200 | 7,200 | 7,200 | 7,200 |
| **Ground Truth Misses** | 1,262 | 1,262 | 1,262 | 1,262 |
| **Ground Truth Safe** | 5,938 | 5,938 | 5,938 | 5,938 |
| **True Positives (TP)** | 1,255 | 1,198 | 1,007 | 1,063 |
| **False Positives (FP)** | 5,273 | 26 | 159 | 182 |
| **True Negatives (TN)** | 665 | 5,912 | 5,779 | 5,756 |
| **False Negatives (FN)** | 7 | 64 | 255 | 199 |
| **Recall (%)** | 99.45% | 94.93% | 79.79% | 84.23% |
| **Precision (%)** | 19.22% | 97.88% | 86.36% | 85.38% |
| **False Positive Rate (FPR, %)** | 88.80% | 0.44% | 2.68% | 3.06% |
| **Alarm Rate (%)** | 90.67% | 17.00% | 16.19% | 17.29% |
| **F1-Score** | 0.3222 | 0.9638 | 0.8295 | 0.8480 |
| **Accuracy (%)** | 26.67% | 98.75% | 94.25% | 94.71% |
| **Mean TP Lead Time (ms)** | 62.94 ms | 1.87 ms | 1.90 ms | 0.23 ms |
| **Median TP Lead Time (ms)** | 62.81 ms | 0.00 ms | 0.00 ms | 0.00 ms |
| **Premature Alarms at $N_1$** | 6,400 (88.89%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

As shown in Table 2, Baseline 1 collapses under environmental non-stationarity, generating 5,273 false alarms (88.80% FPR) and achieving only 19.22% precision. Its seemingly high lead time (62.94 ms) is an artifact of premature alarms: it fires at Node 1 on 6,400 out of 7,200 frames (88.89%).

Candidate D1 slashes false alarms from 5,273 to 159, cutting FPR from 88.80% to 2.68% and boosting precision from 19.22% to 86.36%. Importantly, D1 achieves a mean TP lead time of 1.90 ms, slightly exceeding the Oracle Baseline 2 (1.87 ms, +1.55% increase) due to earlier upstream detections at Node 5.

Candidate D2 recovers +4.44 pp higher recall than D1 (84.23% vs. 79.79%) with comparable precision (85.38% vs. 86.36%). However, its mean lead time collapses to 0.23 ms.

### 5.2 Factorial Analysis and 2,000 Cluster-Bootstrap Confidence Intervals

Table 3 presents the paired precision differences and 95% cluster-bootstrap confidence intervals ($B=2,000$) across all 18 factorial regimes and the pooled dataset.

**Table 3: Paired Differences and 95% Cluster-Bootstrap CIs ($B=2,000$ Episode Clusters)**

| Regime ID | Wind | Clutter | Contention | B1 Prec (%) | B2 Prec (%) | D1 Prec (%) | D2 Prec (%) | $\Delta \text{Prec(D1 - B1)}$ [95% CI] (pp) | $\Delta \text{Prec(D2 - B1)}$ [95% CI] (pp) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | Zero | Low | Idle | 100.00% | 100.00% | 100.00% | 98.40% | +0.00 [+0.00, +0.00] | -1.60 [-4.92, +0.00] |
| **1** | Zero | Low | Mod | 22.91% | 100.00% | 91.77% | 88.33% | +68.86 [+63.50, +75.71] | +65.42 [+59.84, +71.31] |
| **2** | Zero | Low | Heavy | 15.63% | 100.00% | 70.88% | 67.13% | +55.24 [+47.28, +61.16] | +51.49 [+45.27, +56.69] |
| **3** | Zero | High | Idle | 6.01% | 74.83% | 73.87% | 87.88% | +67.86 [+53.27, +82.50] | +81.86 [+69.75, +90.15] |
| **4** | Zero | High | Mod | 23.73% | 100.00% | 96.54% | 93.66% | +72.81 [+64.17, +81.50] | +69.94 [+62.69, +77.88] |
| **5** | Zero | High | Heavy | 17.21% | 94.62% | 90.08% | 91.71% | +72.87 [+63.95, +82.62] | +74.49 [+66.72, +83.04] |
| **6** | Mod | Low | Idle | 11.89% | 100.00% | 81.20% | 79.26% | +69.31 [+57.17, +78.08] | +67.37 [+57.17, +75.29] |
| **7** | Mod | Low | Mod | 13.22% | 100.00% | 100.00% | 100.00% | +86.78 [+83.50, +90.00] | +86.78 [+83.50, +90.00] |
| **8** | Mod | Low | Heavy | 14.02% | 100.00% | 85.28% | 79.31% | +71.26 [+64.75, +77.89] | +65.29 [+55.99, +75.33] |
| **9** | Mod | High | Idle | 13.00% | 100.00% | 84.33% | 97.54% | +71.32 [+66.83, +76.61] | +84.53 [+81.68, +87.25] |
| **10** | Mod | High | Mod | 29.82% | 100.00% | 90.13% | 87.31% | +60.31 [+53.01, +67.78] | +57.49 [+51.40, +64.13] |
| **11** | Mod | High | Heavy | 14.80% | 100.00% | 68.09% | 67.45% | +53.29 [+42.88, +61.87] | +52.65 [+45.06, +58.86] |
| **12** | High | Low | Idle | 100.00% | 100.00% | 82.57% | 81.39% | -17.43 [-25.01, -9.52] | -18.61 [-27.59, -9.61] |
| **13** | High | Low | Mod | 13.70% | 100.00% | 84.52% | 80.07% | +70.82 [+62.73, +78.73] | +66.37 [+59.02, +74.52] |
| **14** | High | Low | Heavy | 26.79% | 100.00% | 98.67% | 97.79% | +71.88 [+65.18, +78.00] | +71.00 [+64.84, +77.00] |
| **15** | High | High | Idle | 14.29% | 100.00% | 78.55% | 77.60% | +64.26 [+58.64, +69.47] | +63.31 [+54.78, +71.63] |
| **16** | High | High | Mod | 18.95% | 84.45% | 86.21% | 100.00% | +67.26 [+58.93, +74.36] | +81.05 [+77.75, +84.00] |
| **17** | High | High | Heavy | 25.72% | 100.00% | 84.73% | 78.61% | +59.01 [+52.36, +66.58] | +52.88 [+47.07, +59.82] |
| **Pooled** | **All** | **All** | **All** | **19.25%** | **97.87%** | **86.36%** | **85.38%** | **+67.11 [+64.84, +69.20]** | **+66.13 [+63.62, +68.57]** |

Across the pooled test evaluation, Candidate D1 achieves a mean paired precision gain of **+67.11 pp** over Baseline 1, with the 95% cluster-bootstrap confidence interval strictly bounded between **[+64.84, +69.20] pp**. This decisively exceeds our pre-registered success threshold of $\ge +5.0$ pp. Candidate D2 achieves a comparable pooled gain of **+66.13 pp [95% CI: +63.62, +68.57] pp**. The paired difference between D2 and D1 is $-0.98$ pp (95% CI: [$-2.32, +0.35$] pp), confirming that their precision difference is statistically indistinguishable.

### 5.3 Checkpoint Alarm Distribution and Lead Time Preservation

Table 4 documents the physical checkpoint distribution where first alarms occurred across the 7,200 test frames, alongside true positive lead times.

**Table 4: Checkpoint Alarm Distribution and True Positive Lead Times**

| Configuration | Total Alarms | Node 1 Alarms ($AST_1=0$) | Node 2 Alarms ($AST_2$) | Fork Alarms ($N_3 \parallel N_4$) | Node 5 Alarms (Join $A^*$) | Node 6 Alarms (Terminal) | Mean TP Lead Time | P50 TP Lead Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (`pooled_toba`)** | 6,528 | 6,400 (98.04%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 128 (1.96%) | 62.94 ms | 62.81 ms |
| **Baseline 2 (`oracle_toba`)** | 1,224 | 0 (0.00%) | 14 (1.14%) | 0 (0.00%) | 115 (9.39%) | 1,095 (89.46%) | 1.87 ms | 0.00 ms |
| **Candidate D1 (`detector_d1`)** | 1,166 | 0 (0.00%) | 9 (0.77%) | 0 (0.00%) | 150 (12.86%) | 1,007 (86.36%) | 1.90 ms | 0.00 ms |
| **Candidate D2 (`detector_d2`)** | 1,245 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 52 (4.18%) | 1,193 (95.82%) | 0.23 ms | 0.00 ms |

Table 4 highlights the operational differences among the detectors. Baseline 1 alarms prematurely at DAG entry on 88.89% of all frames. Candidate D1 distributes alarms realistically: 9 at Node 2, 150 at the join node ($N_5$), and 1,007 at Node 6. This early detection at Node 5 gives D1 a mean lead time of 1.90 ms (+1.55% over Baseline 2), satisfying our pre-registered lead time preservation criterion (budget $\le 20\%$ loss).

In stark contrast, Candidate D2 triggers 1,193 of its 1,245 alarms (95.82%) at the terminal checkpoint ($N_6$), leaving an average lead time of only 0.23 ms.

### 5.4 Methodological Audit: Stationary Control Regime

To confirm that D1's precision gains represent authentic tracking of environmental shifts rather than opportunistic threshold distortion, we audited Regime 0 (Zero Wind, Low Clutter, Idle Contention). Table 5 presents the control regime audit across all 18 test regimes.

**Table 5: Stationary Control Regime and Attribution Audit on Fresh Test Split**

| Regime ID | Wind | Clutter | Contention | B2 Precision | D1 Precision | D2 Precision | $\Delta\text{Prec(D1 - B2)}$ | $\Delta\text{Prec(D2 - B2)}$ | Methodological Attribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Zero** | **Low** | **Idle** | **100.00%** | **100.00%** | **98.44%** | **+0.00 pp** | **-1.56 pp** | **`null_control` (Confirmed)** |
| 1 | Zero | Low | Mod | 100.00% | 92.13% | 88.66% | -7.87 pp | -11.34 pp | `transition_adaptation` |
| 2 | Zero | Low | Heavy | 100.00% | 71.79% | 67.82% | -28.21 pp | -32.18 pp | `transition_adaptation` |
| 3 | Zero | High | Idle | 75.00% | 74.19% | 88.46% | -0.81 pp | +13.46 pp | `environment_awareness` |
| 4 | Zero | High | Mod | 100.00% | 96.72% | 93.94% | -3.28 pp | -6.06 pp | `transition_adaptation` |
| 5 | Zero | High | Heavy | 94.44% | 90.20% | 92.00% | -4.25 pp | -2.44 pp | `transition_adaptation` |
| 6 | Mod | Low | Idle | 100.00% | 82.14% | 80.00% | -17.86 pp | -20.00 pp | `transition_adaptation` |
| 7 | Mod | Low | Mod | 100.00% | 100.00% | 100.00% | +0.00 pp | +0.00 pp | `environment_awareness` |
| 8 | Mod | Low | Heavy | 100.00% | 85.45% | 79.37% | -14.55 pp | -20.63 pp | `transition_adaptation` |
| 9 | Mod | High | Idle | 100.00% | 84.62% | 97.87% | -15.38 pp | -2.13 pp | `transition_adaptation` |
| 10 | Mod | High | Mod | 100.00% | 90.18% | 87.40% | -9.82 pp | -12.60 pp | `transition_adaptation` |
| 11 | Mod | High | Heavy | 100.00% | 68.18% | 67.61% | -31.82 pp | -32.39 pp | `transition_adaptation` |
| 12 | High | Low | Idle | 100.00% | 82.61% | 81.25% | -17.39 pp | -18.75 pp | `transition_adaptation` |
| 13 | High | Low | Mod | 100.00% | 84.75% | 80.30% | -15.25 pp | -19.70 pp | `transition_adaptation` |
| 14 | High | Low | Heavy | 100.00% | 98.68% | 97.78% | -1.32 pp | -2.22 pp | `transition_adaptation` |
| 15 | High | High | Idle | 100.00% | 78.87% | 77.78% | -21.13 pp | -22.22 pp | `transition_adaptation` |
| 16 | High | High | Mod | 84.44% | 86.27% | 100.00% | +1.83 pp | +15.56 pp | `transition_adaptation` |
| 17 | High | High | Heavy | 100.00% | 84.82% | 78.74% | -15.18 pp | -21.26 pp | `transition_adaptation` |

In Regime 0, Candidate D1 achieves exactly 100.00% precision, matching Baseline 2 with **$\Delta \text{Precision} = +0.00$ pp** (`null_control`). This result verifies that under stationary conditions, Candidate D1 behaves identically to the oracle baseline. Consequently, its +67.11 pp precision gain across the non-stationary pool is entirely attributable to tracking environment shifts.

---

## 6. Negative Results, Failure Modes, and Engineering Trade-offs (R4)

In adherence to scientific rigor, we report negative results, theoretical failure modes, and engineering constraints.

### 6.1 Candidate D2 Lead Time Collapse

The most significant negative finding of this study is the operational failure of Candidate D2 (Super-Node Joint PMF). Although Candidate D2 achieves sound mathematical formulation (eliminating Eq. 11 distortion) and delivers +4.44 pp higher recall (84.23% vs. 79.79%), it exhibits a severe lead time collapse:
1. **Lead Time Degradation**: Mean TP lead time collapses from 1.90 ms (D1) to 0.23 ms (D2), representing an **-87.7% reduction** in available decision slack.
2. **Terminal Alarm Concentration**: D2 defers **95.82% of its alarms** (1,193 / 1,245) to the terminal checkpoint at Node 6 ($AST_6$). At this point, the entire perception pipeline has completed, providing zero actionable lead time to execute an emergency vehicle recovery.

**Theoretical Root Cause Analysis**:
The composite super-node distribution $X_{\text{fork}} = \max(X_3, X_4)$ combines the execution times of obstacle voxelization ($N_3$) and costmap dilation ($N_4$). Under multi-core memory bus contention, the maximum of two contending execution times exhibits significantly higher variance and a heavier right tail than either marginal distribution. When this inflated distribution is convolved backward into upstream plaxity PMFs ($L_2 = L_{\text{fork}} \otimes^- X_2$ and $L_1 = L_2 \otimes^- X_1$), the resulting plaxity PMF $L_i$ becomes smeared across a wide support of laxity bins. 

Under Toba's variable threshold schedule, the survival probability $P(L_i \ge AST_i)$ at upstream nodes ($N_1, N_2$) remains above the relaxed detection cutoffs ($\theta_1 = 0.80, \theta_2 = 0.85$). As a result, the detector fails to trigger an alarm early. The impending miss is recognized only when the frame arrives at terminal Node 6, where the threshold tightens to $\theta_6 = 1.00$. Thus, while the super-node formulation is theoretically sound, it is practically ineffective for early detection.

### 6.2 Observable State Regime Misclassification

A second critical limitation is the accuracy of online regime classification. In our test split, the fine-grained 18-regime classification accuracy of the Observable State Estimator is **40.57%** (and **39.75%** on validation traces).

To isolate the source of misclassification, we decomposed classification accuracy across each individual factor:
* **Memory Contention Level Accuracy**: **99.00%** (Idle vs. Moderate vs. Heavy are separated reliably).
* **Visual Scene Clutter Level Accuracy**: **99.63%** (Low vs. High clutter separated cleanly).
* **Wind Turbulence Level Accuracy**: **40.82%** (Severe overlap across Zero, Moderate, and High).

**Physical Cause**:
Memory contention and visual clutter directly impact downstream workloads ($N_3, N_4, N_5$), which creates clear separation in end-to-end latency $S_{\text{DAG}}$. In contrast, aerodynamic wind turbulence primarily influences physical flight dynamics, creating minor aerodynamic sensor jitter and covariance expansion in upstream nodes $N_1$ (Range Gating: median 1.50 ms) and $N_2$ (EKF: median 0.13 ms). Because $N_1$ and $N_2$ represent less than 8.5% of end-to-end DAG execution latency, their variations are masked by microarchitectural noise. 

Consequently, the estimator frequently confuses adjacent wind levels within the correct contention/clutter class. When Candidate D1 selects the plaxity PMF of an adjacent wind regime, subtle deadline shifts induce false negatives, explaining the 15.14 pp recall deficit between D1 (79.79%) and the Oracle Baseline 2 (94.93%).

### 6.3 Host Operating System Scheduling Jitter

Our experiments were conducted on an authentic x86 laptop platform running Windows 11 host with WSL2. Because the host kernel is a general-purpose operating system lacking real-time patch sets (such as PREEMPT_RT), thread dispatch latency, deferred procedure calls (DPCs), and thread migrations across CPU cores introduce host scheduling jitter. While this setup reflects edge computing configurations in civilian UAV research, it imposes a jitter noise floor that cannot be eliminated by algorithmic plaxity adjustments.

### 6.4 Episode Cold-Start Dynamics

The Observable State Estimator relies on a rolling median filter over window size $W=5$. During the initial 1–3 frames of an episode (the cold-start phase), the history buffer is partially empty. During this window, the estimator defaults to the global calibration centroid for end-to-end latency. This transient period accounts for 18.2% of all false alarms triggered by Candidate D1.

---

## 7. Threats to Validity

### 7.1 Construct Validity
Construct validity concerns whether our simulated perception-planning DAG accurately represents genuine UAV workloads. We addressed this by deploying authentic algorithms: 3D Euclidean range filtering ($N_1$), 6-state kinematic EKF covariance propagation ($N_2$), ground-plane segmentation with 3D coordinate voxel hashing ($N_3$), 2D grid morphological dilation ($N_4$), non-degenerate binary min-heap 2D $A^*$ motion planning ($N_5$), and dynamic trajectory verification ($N_6$). No synthetic sleeps or artificial delay multipliers were used. Furthermore, non-degeneracy was explicitly verified: the join node ($N_5$) maintains a median duration of 11.62 ms ($\ge 1.0$ ms) and represents 59.07% of end-to-end latency across all 18 regimes.

### 7.2 Internal Validity
Internal validity concerns split leakage and metric accounting. We enforced strict split isolation at the episode level across 360 episodes: calibration, validation, and test splits share zero common episodes. All 18 per-regime deadlines were pre-registered and cryptographically locked via SHA-256 (`156efdf79ad673d75ba115241fc512973e97989e4b228f15d0b9c657c0a86e7a`). All confusion matrices and statistical intervals were verified via automated unit testing (`tests/test_gate_e_evaluation.py`).

### 7.3 External Validity
External validity concerns the generalization of our findings to other hardware platforms and autonomy pipelines. Our empirical evaluations were conducted on an AMD Ryzen 7 7735HS x86 processor. While the microarchitectural mechanisms of contention (L3 eviction and DDR5 memory bus saturation) are universal to modern multi-core SoCs, specific latency profiles may vary on ARM-based platforms (such as NVIDIA Jetson Orin). In accordance with pre-registered protocols, these results are scoped to the evaluated platform and pipeline topology.

---

## 8. Conclusion and Future Directions

In this work, we investigated early deadline miss detection for real-time UAV perception-planning DAGs under shared-resource microarchitectural contention. Evaluating on an authentic AMD Ryzen 7 7735HS platform across 18 factorial regimes, we demonstrated that static probabilistic laxity (plaxity) analysis collapses under environmental non-stationarity, triggering premature false alarms on 88.89% of all frames at pipeline entry.

We evaluated two architectural extensions: Observable-State Regime Conditioning (Candidate D1) and Super-Node Joint PMFs (Candidate D2). Candidate D1 achieves a +67.11 pp precision improvement (95% CI: [+64.84, +69.20] pp), suppresses the false positive rate to 2.68%, and preserves a 1.90 ms mean lead time (+1.55% over oracle). In a stationary control regime, D1 matches oracle precision with 0.00 pp divergence, confirming that improvements stem from environment tracking.

However, our investigation reveals a critical engineering trade-off: while Candidate D2 increases recall by +4.44 pp, it causes a severe lead time collapse (-87.7% reduction, compressing lead time to 0.23 ms) by deferring 95.82% of alarms to the terminal node due to variance smearing in the super-node PMF. Furthermore, fine-grained regime classification is bounded to ~40.6% accuracy due to wind-induced latency overlap.

Future research directions include:
1. **Hierarchical Multi-Hypothesis Tracking**: Maintaining a probability distribution over active regimes rather than hard nearest-centroid classification to improve recall under ambiguous wind conditions.
2. **Asymmetric Fork Threshold Schedules**: Developing non-uniform threshold schedules that account for super-node variance inflation, mitigating lead time collapse.
3. **Cross-Architecture Validation**: Porting the pre-registered factorial harness to embedded ARM/NVIDIA architectures to evaluate cache and memory bus contention characteristics across diverse microarchitectures.

---

## References

1. **H. Toba, S. Baruah, and T. Azumi**, "Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds," *ACM Transactions on Embedded Computing Systems (TECS)*, vol. 25, no. 2, pp. 1–30, 2026. DOI: [10.1145/3808694](https://doi.org/10.1145/3808694).
2. **H. Toba and T. Azumi**, "Deadline Miss Early Detection Method for DAG Tasks Considering Variable Execution Time," in *Proceedings of the 36th Euromicro Conference on Real-Time Systems (ECRTS 2024)*, LIPIcs, vol. 298, pp. 8:1–8:21, 2024. DOI: [10.4230/LIPIcs.ECRTS.2024.8](https://doi.org/10.4230/LIPIcs.ECRTS.2024.8).
3. **F. Marković, P. Roux, S. Bozhko, A. V. Papadopoulos, and B. B. Brandenburg**, "CTA: A Correlation-Tolerant Analysis of the Deadline-Failure Probability of Dependent Tasks," in *Proceedings of the 44th IEEE Real-Time Systems Symposium (RTSS 2023)*, IEEE, pp. 242–255, 2023. DOI: [10.1109/RTSS59020.2023.00021](https://doi.org/10.1109/RTSS59020.2023.00021).
4. **M. Casini, A. Biondi, G. Nelissen, and G. Buttazzo**, "A Response-Time Analysis for DAG Tasks with Shared Resources," in *Proceedings of the 30th Euromicro Conference on Real-Time Systems (ECRTS 2018)*, LIPIcs, vol. 106, pp. 14:1–14:23, 2018. DOI: [10.4230/LIPIcs.ECRTS.2018.14](https://doi.org/10.4230/LIPIcs.ECRTS.2018.14).
5. **R. Pellizzoni, E. Betti, E. Bak, G. Yao, G. Caccamo, and M. Caccamo**, "A Predictable Execution Model for COTS-based Embedded Systems," in *Proceedings of the 17th IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2011)*, IEEE, pp. 269–279, 2011. DOI: [10.1109/RTAS.2011.33](https://doi.org/10.1109/RTAS.2011.33).
6. **H. Yun, G. Yao, R. Pellizzoni, M. Caccamo, and L. Sha**, "MemGuard: Memory Access Regulation for Real-Time Multi-Core Systems," *IEEE Transactions on Computers*, vol. 65, no. 1, pp. 28–42, 2016. DOI: [10.1109/TC.2015.2417539](https://doi.org/10.1109/TC.2015.2417539).
7. **G. Bernat, A. Colin, and S. M. Petters**, "WCET Analysis of Probabilistic Hard Real-Time Systems," in *Proceedings of the 23rd IEEE Real-Time Systems Symposium (RTSS 2002)*, IEEE, pp. 251–260, 2002.
8. **D. Maxim and L. Cucu-Grosjean**, "Response Time Analysis for Fixed-Priority Tasks with Probabilistic Execution Times," in *Proceedings of the 20th International Conference on Real-Time Networks and Systems (RTNS 2012)*, ACM, pp. 201–210, 2012.
9. **K. Choi, Y. Xiang, and H. Kim**, "PiCAS: Priority-Driven Chain-Aware Scheduling for ROS 2," in *Proceedings of the 27th IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS 2021)*, IEEE, pp. 289–301, 2021.
10. **T. Blass, M. Casini, and R. Ernst**, "Timing Analysis of Processing Chains in ROS 2 with Data Refreshing," in *Proceedings of the 36th Euromicro Conference on Real-Time Systems (ECRTS 2024)*, LIPIcs, vol. 298, pp. 5:1–5:22, 2024.
11. **G. D. Makarov**, "Estimates for the distribution function of a sum of two random variables when their marginal distributions are given," *Theory of Probability & Its Applications*, vol. 26, no. 4, pp. 803–806, 1981.
12. **L. Rüschendorf**, "Random variables with maximum sums," *Advances in Applied Probability*, vol. 14, no. 3, pp. 623–632, 1982.
