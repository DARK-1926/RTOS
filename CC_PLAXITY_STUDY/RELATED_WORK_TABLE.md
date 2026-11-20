# Systematic related-work comparison (18 Nov 2026)

*Each entry was checked against a primary record: publisher page, DROPS, Crossref, DBLP, or the paper's own text. Claims about
content are limited to what the abstract or text we read states. Entries marked "abstract only" were not read in full.
This complements `LITERATURE_POSITIONING.md` (7 and 16 Nov) and closes its "what remains to search" list.*

## Comparison table

How to read the columns:
- **When it warns:** at frame release (R), at checkpoints within the frame (C), or only after a budget is overrun (O).
- **Guarantee:** what is provably bounded, if anything.
- **Acts on its warnings:** whether the method changes the system's behaviour.
- **Valid when its own actions change the data:** whether its guarantee still holds once those actions shape the data it learns from.

| Work | Online? | When it warns | Model of timing | Guarantee | Window (m, K) | Acts on its warnings | Valid when its own actions change the data | Hardware evidence |
|---|---|---|---|---|---|---|---|---|
| Toba & Azumi, ECRTS 2024 | yes | C (start of each node) | per-node PMFs, convolved; independent | none (thresholds) | no | detection only | n/a | yes |
| Toba, Baruah & Azumi, ACM TECS 2026 | yes | C | as above, with variable probabilistic thresholds | none | no | detection only | n/a | yes |
| Peeck, Schlatow & Ernst, DATE 2021 | yes | O (segment deadline overrun, "temporal exception") | static segment budgets, tuned on traces | (m, k) holds **on the measured trace** used to set the budgets; not enforced online | yes, as a requirement | recovery or propagation; fallback to a safety layer | not addressed | yes (Autoware.Auto, DDS) |
| Bernat & Cayssials, RTSS 2001 (Bi-Modal Scheduler) | yes | R (panic mode when one more miss would break the constraint) | WCET-based schedulability | (m, k) by schedulability analysis | yes | switches to panic mode | n/a (no learning) | no |
| Wu et al., RV 2020 (LNCS), *Efficient System Verification with Multiple Weakly-Hard Constraints for Runtime Monitoring* | yes | O (counts faults) | none (verification of the controlled system) | properties hold if the environment meets (m, K); otherwise switch to a safe mode | yes | switch to a safe mode | n/a | no |
| Jiang, Yang, Fisher, Audsley & Dong, RTSS 2020 (Pythia-MCS) | yes | C (at the I/O access point, before the overrun) | execution time predicted from I/O behaviour | MC schedulability | no | early mode switch | not addressed | yes |
| Villalba Frias et al., RTAS 2017; Friebe et al., RTAS 2023, RTS 2024 | offline | — | Markov / HMM job execution times | bound on deadline-miss probability (reservation / CBS) | no | — | — | yes (Linux SCHED_DEADLINE) |
| Marković et al., RTSS 2023 (CTA); RTSS 2024 (CAA) | offline | — | moments / bootstrapped models, dependence-tolerant | bound on deadline-failure probability | no | — | — | — |
| Eisenklam et al., ECRTS 2026 (MPORA) | yes | (allocation, not warning) | in-kernel instruction-rate prediction + weighted conformal | conformal-based deadline constraints (form not read: abstract only) | no | re-allocates resources (receding horizon) | not stated in the abstract | yes |
| Huang et al., MLSys 2025 (Pitot) | prediction | — | matrix factorisation of interference + conformal quantile regression | probabilistic runtime bounds (marginal) | no | no | n/a | yes (24 devices) |
| Gibbs & Candès, NeurIPS 2021 (ACI) | yes | — | any | long-run miscoverage → α for any sequence | no | no | the bound is sequence-wise; feedback not discussed | — |
| Fannjiang et al., PNAS 2022 (feedback covariate shift) | — | — | any regressor | coverage when test inputs are *chosen* by the model | no | yes (design) | **yes, for one round of design** | — |
| Pazzaglia et al., ECRTS 2019 (DMAC); Maggio et al., ECRTS 2020; Vreman et al., ECRTS 2021 | offline design | — | given miss patterns | closed-loop stability or performance under weakly-hard miss models | yes (as assumption) | controller design | — | — |
| Soyyigit, Yao & Yun, RTCSA 2022 (Anytime-Lidar) | yes | R (deadline-aware scheduling of stages) | stage-time estimates | none stated (deadline-aware accuracy trade-off) | no | yes (skips work) | not addressed | yes (nuScenes) |
| **This work** | yes | **R and C** | cross-frame contention state (KP / KPF), any score | (i) long-run unwarned-miss rate ≤ α; (ii) **≤ m unwarned misses in every K frames**, minimal forcing; (iii) with an on-time fallback, **≤ m late frames in every K**, exact degradation form | **enforced online, by construction** | yes (degrade at release; level-2 fallback) | **yes, all three bounds are sequence-wise; latch-up named and fixed** | **3 platforms, 3 closed loops, pre-registered** |

## What the table shows

1. **Nobody else enforces (m, K) online on top of a learned or probabilistic predictor.**
   - The counting idea is old: Bernat & Cayssials (2001) and Wu et al. (2020).
   - Peeck et al. (2021) is the nearest systems paper: Autoware, weakly-hard, fallback to a safety layer. But its weakly-hard
     property is established on a measured trace when the segment budgets are tuned. It is not enforced at run time, and its
     detection happens after a budget overrun.
   - Our rule enforces the bound online, for any score, with a minimality proof.
2. **Early warning at release is rare.** Toba's detector cannot rank frames at release (our proposition). Pythia-MCS predicts
   before the overrun, but only from I/O behaviour and for mixed-criticality mode switches.
3. **Guarantees that survive the monitor's own interventions.**
   - The conformal literature recognises the problem: Fannjiang et al. (2022) for design feedback, and the performative-prediction
     line.
   - None of the timing papers above discuss it.
   - Our bounds are sequence-wise, so they hold under any feedback. The latch-up failure, and its fix, is the practical side of this.
4. **The control side is established.** DMAC and Maggio / Vreman give stability under weakly-hard miss models. Our contribution
   there is only to *supply* the weakly-hard property, on hardware, with a measured-on-time fallback.

## Must cite, and must state the difference explicitly

- **Peeck, Schlatow & Ernst (DATE 2021).** A reviewer from the Ernst group or the AUTOSAR community will raise it. Difference:
  - they detect by timeouts and check (m, k) offline, on traces;
  - we predict at release and enforce (m, K) online;
  - our bounds come with proofs that do not depend on the trace.
- **Bernat & Cayssials (RTSS 2001)** and **Wu et al. (RV 2020)** for the counting rule.
- **Pythia-MCS (RTSS 2020)** for prediction-triggered early mode switches.
- **MPORA (ECRTS 2026)** for conformal methods in real-time resource allocation. Read the full paper before the manuscript, to
  state the form of its guarantee correctly.
- **Fannjiang et al. (PNAS 2022)** for conformal validity under feedback.
- **Maggio et al. (ECRTS 2020)**, **Vreman et al. (ECRTS 2021)** and **DMAC (ECRTS 2019)** for stability under weakly-hard misses.

## Still not searched exhaustively

- The Ernst group's later work (2022–2026) on monitoring and weakly-hard event chains: there may be a direct successor to Peeck et
  al. **Highest priority before submission.**
- Hammadeh et al. (DATE 2017, TWCA with task dependencies), for offline weakly-hard analysis of chains.
- Safety architectures with timing fallbacks (Simplex-style), for the "fallback that is on time" assumption.

## Sources checked today

- Peeck, Schlatow, Ernst, DATE 2021: full text read (past.date-conference.com, proceedings archive 2021, paper 1497).
- Pythia-MCS: title page read (author page rtss20b.pdf). Venue RTSS 2020, pp. 38–50, doi:10.1109/RTSS49844.2020.00015 (researchr and
  proceedings TOC; DBLP lists only 2 of the 5 authors).
- Wu et al.: Crossref record for doi:10.1007/978-3-030-60508-7_28.
- MPORA: DROPS record doi:10.4230/LIPIcs.ECRTS.2026.17 (abstract only).
- Pitot: arXiv 2503.06428 and the MLSys 2025 listing.
- Fannjiang et al.: PNAS 119(43) e2204569119, PMC record.
- Maggio et al. ECRTS 2020 and Vreman et al. ECRTS 2021: DROPS records.
