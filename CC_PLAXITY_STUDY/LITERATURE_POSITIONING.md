# Literature positioning for CC-plaxity (verified 7 Nov 2026)

Every entry below was checked against a primary record: publisher page, DBLP, DROPS, author or university page, or the full text.
Venue and title details come from those records. Claims about a paper's content are limited to what its abstract, record or
text states. **Correction to the earlier `FINAL_SUBMISSION_BUNDLE/LITERATURE_AUDIT.md`:** its item 7, a "Günzel, von der Brüggen,
Chen, *Pitfalls in Bounding Deadline Miss Probabilities*" paper, could not be found. The real paper with "pitfalls" in its title is
Chen, Ueter, von der Brüggen, Chen (DATE 2019), listed below. Items in that older audit that are not repeated here (e.g. 12, 13, 22)
were not re-verified and must not be cited without checking.

## 1. The base: early deadline-miss detection by plaxity
- **H. Toba, T. Azumi.** *Deadline Miss Early Detection Method for DAG Tasks Considering Variable Execution Time.* ECRTS 2024,
  LIPIcs 298, 8:1–8:21, doi:10.4230/LIPIcs.ECRTS.2024.8.
- **H. Toba, S. Baruah, T. Azumi.** *Enhancing Deadline Miss Early Detection for DAG Tasks with Variable Probabilistic Thresholds.*
  ACM TECS 2026, doi:10.1145/3808694. (Full text checked: §2.2 independence assumption, Eqs. 8–18, §4.2 thresholds, §7 future work.)

**Position.** CC-plaxity keeps their quantity, P(remaining time > D − elapsed), their checkpoints and their thresholds. It
replaces the independent, per-frame node PMFs with a model of the *cause* of dependence: one latent shared slowdown,
tracked across frames. This answers their §2.2 limitation (independence) and their §7 future-work item (runtime distribution updates).

## 2. Closest prior work: Markov / HMM models of execution times
- **B. Villalba Frias, L. Palopoli, L. Abeni, D. Fontanelli.** *Probabilistic Real-Time Guarantees: There Is Life Beyond the
  i.i.d. Assumption.* RTAS 2017, pp. 175–186 (outstanding paper). Computation times of consecutive jobs are modelled as a Markov
  process, and the probabilistic guarantees of reservation-based scheduling are adapted to it. Villalba Frias's 2018 PhD thesis adds
  HMM-based identification from traces (PROSIT).
- **A. Friebe, A. V. Papadopoulos, T. Nolte.** *Identification and Validation of Markov Models with Continuous Emission
  Distributions for Execution Times.* RTCSA 2020.
- **A. Friebe, F. Marković, A. V. Papadopoulos, T. Nolte.** *Adaptive Runtime Estimate of Task Execution Times using Bayesian
  Modeling.* RTCSA 2021. Emission distributions are adapted online with a Bayesian update, and a generalized likelihood ratio switches between clusters.
- **A. Friebe, F. Marković, A. V. Papadopoulos, T. Nolte.** *Continuous-Emission Markov Models for Real-Time Applications:
  Bounding Deadline Miss Probabilities.* RTAS 2023. HMM with Gaussian emissions, under a Constant Bandwidth Server, validated against
  Linux SCHED_DEADLINE. It compares against an independence assumption.
- **A. Friebe, F. Marković, A. V. Papadopoulos, T. Nolte.** *Efficiently Bounding Deadline Miss Probabilities of Markov Chain
  Real-Time Tasks.* Real-Time Systems, 2024.
- **A. Friebe.** *Probabilistic Analysis and Scheduling of Real-Time Systems.* PhD thesis, Mälardalen University, 2025.

**Position (be explicit in the paper; reviewers from this community will ask).** That HMMs describe execution-time
sequences better than i.i.d. models is **established**. It is not our contribution. What differs:

| | Markov/HMM timing line (Frias; Friebe et al.) | CC-plaxity |
|---|---|---|
| Object modelled | one task's sequence of job execution times | a DAG's *node* times within a frame **and** across frames |
| Dependence structure | across jobs, through the hidden state | across jobs (sticky HMM) **and** across nodes of the same frame, through one shared multiplicative factor with per-node loadings β_i |
| Output | offline bound on deadline-miss probability (CBS / reservation) | online, in-flight miss probability at DAG checkpoints (early detection, Toba's quantity) |
| Hidden state | discrete states with emission distributions | continuous log-slowdown σ on a 121-point grid; the emissions come from the same nominal node KDEs, scaled |
| Shift / extrapolation | runtime adaptation of emissions (Friebe 2021) | multiplicative extrapolation to *unseen* heavier contention, with a measured failure boundary (core oversubscription) and a self-check fallback |
| Hardware findings | — | DVFS non-monotonicity (idle is slowest under schedutil); one-factor breakdown under oversubscription |

## 3. Correlation-tolerant / correlation-aware offline analysis (CTA, CAA)
- **F. Marković, P. Roux, S. Bozhko, A. V. Papadopoulos, B. B. Brandenburg.** *CTA: A Correlation-Tolerant Analysis of the
  Deadline-Failure Probability of Dependent Tasks.* RTSS 2023, pp. 317–330, doi:10.1109/RTSS59052.2023.00035. A closed-form
  WCDFP upper bound from Cantelli's inequality. It needs only bounds on per-task means and standard deviations and is valid under arbitrary
  dependence. Fixed-priority scheduling. Parts of the proof are machine-checked in Coq.
- **F. Marković, G. von der Brüggen, M. Günzel, J.-J. Chen, B. B. Brandenburg.** *A Distribution-Agnostic and Correlation-Aware
  Analysis of Periodic Tasks* (CAA). RTSS 2024. It bounds the deadline-failure probability using a two-parameter stochastic task model
  inferred by nonparametric bootstrapping. Designed to dominate CTA.

**Position.** CTA and CAA answer a *design-time* question: a safe bound on deadline-failure probability **despite**
unknown dependence. CC-plaxity answers a *run-time* question: *this* frame's miss probability, **using** the dependence
(its cause, shared contention) as information. They are complementary. CAA/CTA can certify the long-run failure rate
offline; CC-plaxity decides online when to act. Our Proposition (per-frame elapsed-time scores cannot gain ranking from
better PMFs) shows that a better marginal or bound cannot replace runtime state. Note: CTA/CAA bounds are sound
under assumptions we do not make. CC-plaxity gives **no guarantee**, and its calibration fails beyond the oversubscription
boundary (G8). The self-check (CC-SC) is our mitigation, and it is empirical.

## 4. Probabilistic timing analysis and independence
- **R. I. Davis, L. Cucu-Grosjean.** *A Survey of Probabilistic Timing Analysis Techniques for Real-Time Systems.* LITES 6(1), 2019;
  and the companion *A Survey of Probabilistic Schedulability Analysis Techniques for Real-Time Systems*, LITES 2019.
- **L. Cucu-Grosjean.** *Independence – a misunderstood property of and for probabilistic real-time systems.* 2013 (HAL hal-00920504).
- **K.-H. Chen, N. Ueter, G. von der Brüggen, J.-J. Chen.** *Efficient Computation of Deadline-Miss Probability and Potential
  Pitfalls.* DATE 2019.
- **K.-H. Chen, M. Günzel, G. von der Brüggen, J.-J. Chen.** *Critical Instant for Probabilistic Timing Guarantees: Refuted and
  Revisited.* RTSS 2022.

## 5. Contention and its control
- **H. Yun, G. Yao, R. Pellizzoni, M. Caccamo, L. Sha.** *MemGuard: Memory Bandwidth Reservation System for Efficient Performance
  Isolation in Multi-core Platforms.* RTAS 2013, pp. 55–64.

**Position.** Regulation such as MemGuard removes the shared state that CC-plaxity exploits. On regulated platforms we
expect CC's advantage to shrink, the same boundary condition as the earlier UNO Q monitor traces with no shared state. Most
commodity embedded Linux deployments are unregulated, which is the setting tested here.

## 6. Consumers of early warning: degradation and deadline-miss-aware control
- **A. Soyyigit, S. Yao, H. Yun.** *Anytime-Lidar: Deadline-aware 3D Object Detection.* RTCSA 2022. A runtime/accuracy trade-off
  for LiDAR detection under deadlines, on nuScenes.
- **P. Pazzaglia, L. Pannocchi, A. Biondi, M. Di Natale.** *Beyond the Weakly Hard Model: Measuring the Performance Cost of
  Deadline Misses.* ECRTS 2018.
- **P. Pazzaglia, C. Mandrioli, M. Maggio, A. Cervin.** *DMAC: Deadline-Miss-Aware Control.* ECRTS 2019, LIPIcs 133, 1:1–1:24.

**Position.** These works say *what to do* about a (possible) miss. CC-plaxity supplies the *when*, earlier and with calibrated
probabilities. Our closed-loop experiment (G10a: halve points or voxels on alarm) is a minimal anytime-style degradation, not a
contribution of its own.

## 7. Calibration monitoring
- **I. Gibbs, E. Candès.** *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS 2021.

**Position.** CC-SC monitors CC's own probability-integral-transform (PIT) values and falls back when they leave range. This is a
calibration self-check in the spirit of conformal and online-calibration monitoring. It is a simple binomial rule, not a coverage
guarantee. The earlier bundle's per-frame ACI detector is a different use: there ACI adjusted thresholds, while here a
monitor switches models.

## What remains to search before submission
DBLP/Scholar sweeps for: "deadline miss prediction" + ROS 2 / Autoware; DNN or perception latency prediction under co-location
interference; mixed-criticality mode switches triggered by runtime monitors; latent-factor (copula/factor) models of
execution-time dependence. None of these were searched exhaustively here.

## Update 16 Nov 2026: web check of the window rule and new directions

- The window rule (WH-ACI) is NOT new as a scheduling idea. Bernat and Cayssials, "Guaranteed on-line weakly-hard real-time systems" (RTSS 2001): Bi-Modal Scheduler, normal mode is best effort, "panic mode" is entered when one more miss would break the constraint. Same counting idea. Also (m,k)-firm distance-based priority, and the SAW runtime monitor (2020) that counts faults against (m,K) and switches to safe mode. Must be cited; our rule should be described as a BMS-style rule applied to warnings.
- Still not found: (m,K) enforcement on top of a learned or conformal alarm; a guarantee stated on UNWARNED misses on a best-effort platform with no WCET analysis; latch-up of a monitor that acts on its own alarms. Closest ML work: SAOCP (interval regret, not a hard bound), performative risk control, selective conformal risk control, conformal with semi-bandit feedback.
- Stability under weakly-hard constraints with miss-handling strategies is established: Maggio et al. ECRTS 2020 (joint spectral radius), Vreman et al. (extended weakly-hard, hold/zero), Seidel, Maggio and Allgower 2026 (synthesis). A certificate that links our monitor to these is feasible.
- Exploratory (gen/cert_explore.py, results/cert_explore.csv): for the Addendum N pendulum the worst 20-frame block gain under at most 4 holds is 0.001 to 0.14 (below 1 means stable for every admissible miss pattern); certified for all tested plants and budgets. Without a bound the hold mode grows 1.22x to 2.72x per frame.
- PROBLEM found by the same script: in the Addendum O closed loop, WARNED frames still miss the deadline 10 to 16 percent of the time (halving the point cloud is not a timely fallback). The guarantee covers unwarned misses only, and the control study assumes a fallback that is always on time. A certificate needs a fallback with a guaranteed completion time. Not yet built.
