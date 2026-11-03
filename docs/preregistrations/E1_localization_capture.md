# E1: Is injection localization a report, or salience capture?

Status: preregistered before any data was collected. Changes after data collection must be logged at the bottom with the reason.

## Question
Models can name which position received an injected concept vector (Hahami 2607.14111; Zou 2609.35108). That result is treated as introspective access, or at least as immune to the yes-bias confound. In every retrieved design the prompt asks about injection.

If the injected position also wins when the prompt asks something unrelated, the "report" may be the perturbed position capturing attention, with no dependence on being asked. This is the localization analogue of the factual-no control that Hahami used against binary detection.

As far as we could find (targeted search on 2026-11-04), this has not been tested. See `docs/literature/source_verification.md`.

## Design
Single forward pass, scored on first-token logits, no LLM judge.

**Stimulus.** A list of ten labelled tokens, labels 0-9, with neutral candidate words. The list text is identical across tasks and conditions.

**Injection.** At the residual output of layer l, at the candidate token's position: `h <- h + alpha * ||h|| * v`, with v unit norm and alpha relative to the activation norm (as in Zou). Prompt tokens only.

**Concept vectors.** Final-token residual for "Tell me about {c}." minus the mean over 50 baseline words, normalized (Lindsey; Zou).

**Tasks** (instruction text only differs; all answers are a single position label, and no task offers `none` in the primary analysis):

| Code | Instruction summary | Why |
|---|---|---|
| T_inj | Identify the position with an injected concept vector | The reported effect |
| T_pick | Pick one position at random | Purest capture test: no correct answer, no mention of anything unusual |
| T_fav | Which token do you find most interesting | Second no-answer task, guards against T_pick wording effects |
| T_odd | Which token does not belong | Generic anomaly, asked in ordinary language |
| T_fact | Which position holds the word {w} | Has a correct answer; competence check and interference test |
| T_inj_none | T_inj with `none` allowed | Secondary, closer replication of Zou |

**Vector conditions.** Concept; norm-matched random; impact-matched random (norm calibrated so mean Jensen-Shannon divergence of the next-token distribution matches the concept condition, following Ferrara eq. 7); and clean (no injection).

**Splits.** Concepts and token clusters are divided into disjoint calibration and test sets.
- (l, alpha) is chosen on calibration data only, by maximizing T_inj accuracy. This selection favours T_inj, so it is conservative against the capture hypothesis.
- Concepts are not screened in the primary analysis. A secondary analysis screens by calibration accuracy to mimic Zou.

## Estimand
For task t, concept c, cluster k and injected position j, compare logits on the injected and clean runs:

`m = logit(label_j) - mean_{i != j} logit(label_i)`
`L(t,c,k,j) = m_injected - m_clean`

Averaging over all ten positions cancels position and label priors. `L_t` is the mean over the test set. Sham check: with alpha = 0, L is exactly 0.

Capture share: `C_t = L_t / L_inj`, for t in {pick, fav, odd}.

Interference on T_fact: how often the model answers the injected position when the correct position is elsewhere, compared with the clean rate for that same wrong position.

Inference uses a two-way cluster bootstrap resampling concepts and clusters independently (2,000 replicates, percentile 95% intervals), with Holm correction across the three control tasks.

## Validity gates (checked first, per model)
- **V1 competence:** T_fact clean accuracy at least 70%. If this fails, the model cannot do list lookup and is excluded from inference.
- **V2 effect exists:** T_inj accuracy lower bound above 10% and `L_inj` lower bound above 0. If this fails, there is no localization effect to explain at this size. That is a finding about the model, not evidence for or against capture.

## Hypotheses and decision rules
**H1, capture.** The injected position gains on tasks that do not mention injection.
- *Capture:* `L_pick` lower bound above 0 and `C_pick` lower bound at least 0.5.
- *Task-gated:* `L_inj - L_pick` lower bound above 0 and `C_pick` upper bound below 0.25.
- *Mixed:* anything else. Reported as mixed, with no favoured reading.

The same rule is applied to T_fav. If T_pick and T_fav disagree, the result is called mixed.

**H2, generic anomaly.** `L_odd` is as large as `L_inj`, by the same capture rule with T_odd in place of T_pick.

**H3, control strength.** Impact-matched random directions give a larger `L_inj` than norm-matched ones, narrowing the gap to concept vectors that Zou uses to argue against an anomaly detector. Supported if the lower bound of (impact-matched minus norm-matched) is above 0. Reported together with the remaining gap to concept vectors.

**Prior expectation, stated so it can be wrong.** At the early-layer, high-strength settings Zou selected, we expect substantial capture (share at least 0.5). We are less sure at later layers.

## What each outcome does and does not show
- **Capture:** localization accuracy does not, by itself, evidence reporting on internal state. It does not refute Zou's head-level circuit, which could implement salience. It would mean that circuit needs a different reading.
- **Task-gated:** the effect depends on being asked, which rules out plain capture. It does not establish introspection or second-order computation (Singh et al.), and the instruction still tells the model that injection exists.
- **H2 supported:** a report that fires equally on an ordinary-language oddness question is hard to call specific to activations.

## Scope and tiers
- **Small tier, Colab CPU (12 GB, no GPU):** Qwen3-0.6B or nearest available as a pipeline pilot, then Qwen3-1.7B. Hahami and Zou find weak or absent effects near this size, so gate V2 may fail. That outcome is expected and is reported as such.
- **Large tier, code written, run later on a GPU:** Qwen3-4B, then Llama-3.1-8B and Gemma-3-12B if tokens are available, to match Zou's models.

Model identifiers and exact counts per tier are fixed in the config committed before each run. Pilot data are used only to check the pipeline and timing, and are excluded from all inference.

## Change log
All entries below were made before any data was collected.

2026-11-04, implementation choices the text above left open or that compute forces:
- **Positions per cell.** Pilot and small tiers run 5 and 1 positions per (concept, cluster) cell, balanced within clusters, because ten per cell does not fit CPU budgets. The large tier runs all ten. With fewer positions, priors cancel only through balance across cells, not within each cell. Small-tier estimates are weaker for this reason.
- **Random conditions.** Pilot and small tiers run random directions on T_inj only, which is what H3 needs. Both random conditions share one direction per concept.
- **Impact matching.** Jensen-Shannon divergence in nats on the next-token distribution at the last prompt token, since the design scores only the first token. Ferrara eq. 6 averages over teacher-forced positions; this is the single-position case.
- **Holm.** Applied separately to the `L_t > 0` family and the `L_inj - L_t > 0` family. Bounds on `C_t` are unadjusted.
- **Undefined share.** `C_t` is reported as undefined when `|L_inj|` is below 0.05 logits.
- **Power.** Small-tier counts (10 concepts, 5 clusters as first sized) give a thin bootstrap. Counts may be raised once the pilot gives measured timing, before small-tier data exist.

2026-11-04, after independent code review ([review](../reviews/E1_harness_review.md)), still before any data:
- **Correction to the entry above.** The small tier runs 2 positions per cell, not 1, so each concept sees both position parities. It uses 10 concepts and 10 clusters, about 1,082 prompts.
- **Sparse design.** At the small tier each concept meets 5 of the 10 clusters, in a cyclic design. Ten clusters still under-cover (about 90% for a nominal 95%); the output report states this, so small-tier intervals are read as optimistic.
- **Grid and tasks at the small tier.** Alpha is limited to {3, 6}, which covers Zou's selected range but not smaller strengths. T_inj_none is dropped. Calibration uses 2 positions per cell.
- **Concept vector read site.** The "." ending the user's message in the chat rendering, instead of the template's final token, which sits after the generation prompt and may carry little concept content at early layers.
- **Pilot stimuli.** The pilot uses its own concepts and clusters, disjoint from calibration and test, so test estimates stay unseen.
- **H3.** Reported as uninformative if the achieved impact match misses the target by more than 10% (5% at the large tier).
- **Capture share.** A negative `L_inj`, like a near-zero one, makes `C_t` undefined.
- **Selection tie-break.** Smallest alpha, then earliest layer.
- **Numerics.** float32 at all tiers. Non-finite values raise an error rather than producing a verdict.
- **Label mass.** Total probability on the labels under injection is reported per task. It is descriptive, with a warning below 0.5.
- **Screening.** The secondary analysis uses test concepts on calibration clusters, and is off at the small tier.
- **Prompt details.** Instruction comes before the list. `none` is matched as lowercase only.

2026-11-04, after the first dry run on Qwen3-0.6B, which measured label mass only and scored no trial:
- **Answer prefix.** Clean label mass was far too low to score first-token logits: T_inj 0.24, T_pick 0.56, T_fav 0.72, T_odd 0.02, T_fact 0.12, with top tokens such as "Position" and "The". The assistant turn is now prefilled with the identical text "Position number:\n" on all tasks except T_inj_none, which stays unprefixed because a position prefix rules out answering none. Hahami et al. also prefill. This forces a position answer, so the design cannot observe a preference to decline except through T_inj_none.
- **Prefix wording, final.** "Position number:\n" still left T_pick at 0.33 and T_odd at 0.22, so one clean-run sweep of four wordings compared minimum label mass, with a tie rule fixed in advance ([table](../../results/e1/pilot/prefix_sweep.txt)). Chosen: "Position number: ", minimum 0.9998. No further changes.
- **Pilot.** Ran on its own stimuli (2 concepts, 1 cluster) with `inference_allowed: false`. Its numbers are pipeline checks and enter no analysis. They were seen before the small-tier run, so anyone assessing the prior expectation above should know the pilot showed `L_pick` above `L_inj`. The prereg's decision rules, stimuli and counts were not changed after seeing them.
- **Threads.** The small tier sets `OMP_NUM_THREADS=2` in the environment, since the default used one of two cores. No effect on results beyond float rounding.
